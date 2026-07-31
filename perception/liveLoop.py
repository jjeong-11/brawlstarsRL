"""Prototype live perception loop.

Grabs frames from a source (screen capture, or a recorded video played
back in real time for testing), runs the perception stack on each tick,
and prints one status line per tick. This is the skeleton the future
action executor will plug into: each tick produces a `state` dict with
everything the policy needs.

Usage:
    python3 liveLoop.py --video testvideos/test_game1.mp4        # simulate live from a recording
    python3 liveLoop.py --screen                                 # capture the real screen (needs `pip install mss`)
    python3 liveLoop.py --video ... --save-preview               # also write annotated frames to debugOutput/live_preview.png

Rate control:
    The loop targets --fps (default 10) but measures its own processing
    time and throttles itself DOWN automatically when it can't keep up,
    then creeps back up when there's headroom -- so on a slow machine it
    degrades to fewer updates per second instead of falling behind and
    building delay. Frames are always grabbed fresh at tick time (old
    frames are dropped, never queued), so the data is never stale.

Stage scheduling:
    Not every detector runs every tick. Cheap color-mask stages (gas) run
    each tick; OCR-heavy stages (game state, player HUD, enemies/boxes)
    run at their own slower intervals and their last result is carried
    between refreshes. Intervals are set in STAGE_INTERVALS below.
"""

import argparse
import threading
import time
from pathlib import Path

import cv2
import numpy as np

# repo-root/debugOutput, independent of the current working directory
_DEBUG_DIR = Path(__file__).resolve().parent.parent / "debugOutput"

from .getAnchor import find_player_position, find_player_position_ex
from .getHealth import find_health_info
from .getAmmo import find_ammo_info, AmmoLocator
from .getCube import find_cube_info
from .getEnemies import find_entities, find_enemy_health_bars
from .getPickups import find_ground_cubes
from .getGas import gas_info
from .getSuper import find_super_info
from .getGameState import get_game_state


# Seconds between refreshes of each stage. Since the template-based digit
# reader replaced tesseract in the digit hot paths (~1ms vs ~100-300ms per
# read), the player/entity stages are cheap enough to refresh several
# times per second; game_state still uses tesseract for the "Brawlers
# left" line (it contains letters), so it stays at 1s.
STAGE_INTERVALS = {
    "game_state": 1.0,    # brawlers-left OCR + screen classification (tesseract)
    "player": 0.3,        # anchor + HP + ammo + cube counter
    "entities": 0.5,      # enemies + boxes
    "pickups": 0.25,      # ground cubes (color only)
    "gas": 0.0,           # every tick (color only, cheapest)
    "super": 0.2,         # super-charge ring (color only, cheap)
}


class VideoSource:
    """Plays a recording back as if it were a live feed: each grab returns

    the frame at the current wall-clock position, skipping past frames if
    the consumer is slow (exactly how a live capture behaves).
    """

    live = False  # a recording: a None grab means it truly ended -> stop

    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open {path}")
        self.start = time.time()
        self.fps = max(1.0, self.cap.get(cv2.CAP_PROP_FPS))
        self.duration = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.fps
        self._last = None      # last decoded frame (re-served to fast consumers)
        self._last_idx = -1

    def grab(self):
        elapsed = time.time() - self.start
        if elapsed > self.duration:
            return None
        target = int(elapsed * self.fps)

        # Fast consumer: the wall clock hasn't reached a new frame yet, so
        # re-serve the last one instead of forcing the decoder to re-seek
        # and re-decode it (the old per-grab CAP_PROP_POS_MSEC seek was a
        # surprisingly large per-tick cost).
        if self._last is not None and target <= self._last_idx:
            return self._last

        # Slow consumer: skip forward with cheap decode-only grab()s for
        # small gaps; only do a real seek for big jumps.
        cur = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        if target - cur > int(self.fps * 3):
            self.cap.set(cv2.CAP_PROP_POS_MSEC, elapsed * 1000)
        else:
            while cur < target and self.cap.grab():
                cur += 1
        ok, frame = self.cap.read()
        if not ok:
            return None
        self._last = frame
        self._last_idx = target
        return frame

    def close(self):
        self.cap.release()


class ScreenSource:
    """Captures the real screen via mss. `region` is (left, top, width,

    height); None captures the primary monitor. Install with `pip install mss`.
    """

    live = True  # a live feed: a None grab is transient -> skip, don't stop

    def __init__(self, region=None):
        import mss  # imported lazily so video mode works without it

        self.sct = mss.mss()
        self.region = (
            {"left": region[0], "top": region[1], "width": region[2], "height": region[3]}
            if region
            else self.sct.monitors[1]
        )

    def grab(self):
        shot = self.sct.grab(self.region)
        frame = np.array(shot)[:, :, :3]  # BGRA -> BGR
        return np.ascontiguousarray(frame)

    def close(self):
        self.sct.close()


class AdbScreencapSource:
    """Grab frames straight from a physical Android device via adb screencap.

    Dependency-free (just needs `adb` + a connected phone with USB debugging),
    but slow: `screencap -p` is ~150-400ms/frame over USB. Fine for testing and
    slow-tick play. For real-time, mirror the phone with scrcpy and point
    ScreenSource at the scrcpy window instead (see docs/ANDROID_CONTROL.md).
    """

    live = True  # a live feed: a None grab is transient -> skip, don't stop

    def __init__(self, serial=None, retries=3, async_capture=True, max_age=5.0):
        import shutil
        if shutil.which("adb") is None:
            raise RuntimeError("`adb` not found on PATH — connect your phone first.")
        self.base = ["adb"] + (["-s", serial] if serial else [])
        self.retries = retries
        # Raw mode (`screencap` without -p) skips the on-device PNG encode,
        # which is the bulk of the per-frame cost -- typically 2-3x faster
        # end-to-end. If the device's raw header doesn't parse (format
        # differences across Android versions), fall back to PNG for good.
        self.use_raw = True

        # Background capture: a thread continuously pulls frames so grab() hands
        # back the most recent one instantly, overlapping the ~300ms screencap
        # with perception/action. async_capture=False restores blocking capture.
        self._async = async_capture
        self._max_age = max_age            # frames older than this -> grab() None
        self._frame = None
        self._frame_time = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._paused = threading.Event()   # when set, the bg thread stops grabbing
        self._thread = None
        if async_capture:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def pause(self):
        """Stop the background capture (frees the USB/adb channel for taps, e.g.
        while pressing menu buttons between matches). grab() falls back to a
        synchronous capture so callers still get frames."""
        self._paused.set()

    def resume(self):
        self._paused.clear()

    @staticmethod
    def _decode_raw(raw):
        """Decodes `screencap` raw output: a 12-byte (pre-P) or 16-byte
        (Android P+) little-endian header (width, height, pixel format,
        [colorspace]) followed by RGBA_8888 pixels."""
        if len(raw) < 16:
            return None
        w, h = np.frombuffer(raw[:8], np.uint32).astype(np.int64)
        if w <= 0 or h <= 0 or w > 10000 or h > 10000:
            return None
        npix = int(w * h * 4)
        for header in (12, 16):
            if len(raw) >= header + npix:
                rgba = np.frombuffer(raw, np.uint8, count=npix, offset=header)
                rgba = rgba.reshape(int(h), int(w), 4)
                return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        return None

    def _capture_once(self):
        import subprocess
        # A single screencap can come back empty over USB; retry a few times
        # before giving up so one hiccup doesn't look like the feed ending.
        for _ in range(self.retries):
            if self.use_raw:
                try:
                    raw = subprocess.run(self.base + ["exec-out", "screencap"],
                                         capture_output=True, timeout=5).stdout
                except Exception:
                    raw = b""
                if raw:
                    img = self._decode_raw(raw)
                    if img is not None:
                        return img
                    self.use_raw = False  # unrecognized layout: stick to PNG
            try:
                raw = subprocess.run(self.base + ["exec-out", "screencap", "-p"],
                                     capture_output=True, timeout=5).stdout
            except Exception:
                raw = b""
            if raw:
                img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    return img
        return None  # transient miss; live consumers skip and keep going

    def _loop(self):
        # Background thread: keep the latest frame fresh while the main loop works.
        while not self._stop.is_set():
            if self._paused.is_set():
                time.sleep(0.05)      # paused: leave the USB channel free for taps
                continue
            img = self._capture_once()
            if img is not None:
                with self._lock:
                    self._frame = img
                    self._frame_time = time.time()
            else:
                time.sleep(0.05)      # brief backoff after a failed capture

    def grab(self):
        # Synchronous when not async, or while paused (so menu navigation still
        # gets frames but without the background thread hogging USB).
        if not self._async or self._paused.is_set():
            return self._capture_once()
        # Hand back the most recent frame instantly (no blocking on screencap).
        deadline = time.time() + 5.0
        while True:
            with self._lock:
                frame, ftime = self._frame, self._frame_time
            if frame is not None:
                # Return None if capture has stalled (phone unplugged/asleep) so
                # we don't train on a frozen frame; the env just skips the tick.
                return frame if (time.time() - ftime) <= self._max_age else None
            if time.time() >= deadline:
                return None           # never got a first frame
            time.sleep(0.05)

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


class LivePerception:
    """Runs the perception stack with per-stage refresh intervals and

    simple temporal smoothing of the game state (a single flickered
    reading -- e.g. the brawlers-left counter briefly obscured by an
    effect, observed in real footage -- doesn't change the reported state
    until it repeats on the next refresh).
    """

    # HP reads above this are OCR garbage (a damage popup merging into the
    # digit line produced "25075" from a real 5075 on footage). Max
    # brawler HP with full cube boosts stays well under this.
    HP_SANITY_CAP = 20000

    def __init__(self, async_game_state: bool = True):
        self.last_run = {name: 0.0 for name in STAGE_INTERVALS}
        self.cache = {
            "game_state": {"state": "unknown", "brawlers_left": None, "rank": None},
            "anchor": None,
            "hp": None,
            "ammo": None,
            "ammo_known": False,
            "anchor_fresh": False,
            "hud_cubes": None,
            "enemies": [],
            "boxes": [],
            "cubes": [],
            "in_gas": False,
            "gas_sides": (0.0, 0.0, 0.0, 0.0),   # coverage left/right/above/below player
            "gas_safe": (0.0, 0.0),              # unit vector away from the gas
            "gas_frac": 0.0,
            "super_charge": 0.0,
        }
        self._pending_state = None   # for game-state smoothing
        self._pending_anchor = None  # for anchor-jump confirmation
        # Learns where the ammo row sits relative to the anchor, so ammo stops
        # depending on the HP digit line parsing. See perception/getAmmo.py.
        self._ammo_locator = AmmoLocator()
        self._anchor_time = 0.0
        self._pending_hp = None      # for large-HP-change confirmation
        self._pending_cubes = None   # for large-cube-jump confirmation

        # ASYNC GAME STATE: get_game_state is the one stage still built on
        # tesseract (the "Brawlers left" line contains letters), and each
        # call costs ~100-300ms -- run synchronously it froze one tick per
        # second and was the single biggest source of tick-time spikes.
        # It's coarse, slow-changing episode-control info, so it runs in a
        # background worker instead: ticks submit a frame when the stage is
        # due and harvest the finished result whenever it lands. Both
        # tesseract (a subprocess) and OpenCV release the GIL, so the
        # worker doesn't slow the main loop down. The very first call runs
        # synchronously so the loop starts with a real state, and
        # `async_game_state=False` restores the fully synchronous behavior.
        self._gs_async = async_game_state
        self._gs_lock = threading.Lock()
        self._gs_thread = None
        self._gs_result = None       # (raw_state_dict, duration_seconds)
        self._gs_ran_once = False

    def close(self):
        """Wait for the background game-state OCR worker to finish (optional).

        The worker is non-daemon, so the interpreter already joins it on exit;
        this lets a caller join it explicitly for a graceful shutdown.
        """
        t = self._gs_thread
        if t is not None and t.is_alive():
            t.join(timeout=3.0)

    def _due(self, name, now):
        if now - self.last_run[name] >= STAGE_INTERVALS[name]:
            self.last_run[name] = now
            return True
        return False

    def _apply_game_state(self, raw):
        """Temporal smoothing: require the same NEW state twice in a row
        before switching; brawlers_left updates immediately while in_match."""
        c = self.cache
        if raw["state"] == c["game_state"]["state"]:
            c["game_state"] = raw
            self._pending_state = None
        elif self._pending_state is not None and raw["state"] == self._pending_state["state"]:
            c["game_state"] = raw
            self._pending_state = None
        else:
            self._pending_state = raw

    def _gs_worker(self, frame):
        t0 = time.time()
        raw = get_game_state(frame)
        with self._gs_lock:
            self._gs_result = (raw, time.time() - t0)

    def tick(self, frame):
        now = time.time()
        timings = {}
        c = self.cache

        # One shared HSV conversion per tick -- previously every stage
        # (anchor, entities, pickups, game state, gas) re-converted the
        # same full frame on its own.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if self._gs_async and self._gs_ran_once:
            # Harvest a finished background result on ANY tick it lands.
            with self._gs_lock:
                result, self._gs_result = self._gs_result, None
            if result is not None:
                raw, duration = result
                timings["game_state"] = duration  # worker time, off-thread
                self._apply_game_state(raw)
            # Submit a new job when due (frame copied: the worker outlives
            # this tick and sources may reuse their frame buffer).
            if self._due("game_state", now) and (
                    self._gs_thread is None or not self._gs_thread.is_alive()):
                # non-daemon: the interpreter joins this one-shot worker at exit
                # instead of killing it mid-OCR (which aborted with a core dump).
                self._gs_thread = threading.Thread(
                    target=self._gs_worker, args=(frame.copy(),), daemon=False)
                self._gs_thread.start()
        elif self._due("game_state", now):
            t0 = time.time()
            raw = get_game_state(frame, hsv=hsv)
            timings["game_state"] = time.time() - t0
            self._apply_game_state(raw)
            self._gs_ran_once = True

        in_match = c["game_state"]["state"] == "in_match"

        # Reset per-life values when we die, a match ends, or a new one loads.
        if not in_match and c["game_state"]["state"] in ("loading", "match_end", "defeated"):
            c["hud_cubes"] = None
            c["hp"] = None
            c["anchor"] = None
            c["super_charge"] = 0.0

        if in_match and self._due("player", now):
            t0 = time.time()
            # Seed with the previous anchor: the player barely moves between
            # frames, so his own last position discriminates far better than
            # "nearest the screen centre", which an enemy standing inboard of
            # him satisfies just as well. See find_player_position_ex.
            ax, ay, ar, verified = find_player_position_ex(
                frame, hsv=hsv, prior=(c["anchor"][0], c["anchor"][1]) if c["anchor"] else None)
            raw_anchor = (ax, ay, ar)

            # getAnchor now reports radius 0 when it could not VERIFY the
            # player (no readable HP number above a ring) rather than
            # guessing. Treat that as "no detection": coast on the previous
            # anchor so spatial consumers still have a position, but mark the
            # tick as not-fresh so the HUD readers below skip it.
            #
            # That distinction matters. Measured on 87 in-match frames: with a
            # freshly verified anchor the HP readout parses 84% of the time;
            # with a coasted one, 13% -- because the player has moved and every
            # HUD search window is positioned relative to him. A coasted anchor
            # is good enough to steer by and useless to read HUD from.
            anchor_fresh = verified and ar > 0
            c["anchor_fresh"] = anchor_fresh

            w = frame.shape[1]
            prev = c["anchor"]
            if not anchor_fresh:
                pass                      # keep c["anchor"] as-is (coast)
            elif prev is not None and now - self._anchor_time < 4.0:
                jump = ((raw_anchor[0] - prev[0]) ** 2 + (raw_anchor[1] - prev[1]) ** 2) ** 0.5
                if jump > w * 0.18:
                    if (self._pending_anchor is not None
                            and ((raw_anchor[0] - self._pending_anchor[0]) ** 2
                                 + (raw_anchor[1] - self._pending_anchor[1]) ** 2) ** 0.5 < w * 0.1):
                        c["anchor"] = raw_anchor  # confirmed twice
                        self._pending_anchor = None
                    else:
                        self._pending_anchor = raw_anchor  # hold, keep prev
                else:
                    c["anchor"] = raw_anchor
                    self._pending_anchor = None
            else:
                c["anchor"] = raw_anchor
            if anchor_fresh:
                self._anchor_time = now

            # HUD readers run only on a freshly verified anchor (see above).
            #
            # ONE flag for all three of them. They were previously guarded
            # individually and the cube reader was missed, which crashed live
            # training on the first tick: getCube unpacks the anchor without a
            # None check, and c["anchor"] starts as None and stays None until
            # the first VERIFIED detection. Anything added here must sit inside
            # this flag too.
            hud_readable = anchor_fresh and c["anchor"] is not None

            health = (find_health_info(frame, c["anchor"]) if hud_readable
                      else {"bounding_box": None, "current_health": None})
            new_hp = health["current_health"]
            # HP PERSISTENCE with pending-confirmation: keep the last good
            # value through unreadable frames; accept small changes
            # immediately; a LARGE change in either direction must repeat
            # on the next refresh before it's believed. (A one-directional
            # cap was tried first and failed: a single low misread during
            # spawn fade-in got accepted and then blocked the real value
            # forever.)
            if new_hp is not None and new_hp <= self.HP_SANITY_CAP:
                if c["hp"] is None or abs(new_hp - c["hp"]) <= max(1500, c["hp"] * 0.4):
                    c["hp"] = new_hp
                    self._pending_hp = None
                elif (self._pending_hp is not None
                        and abs(new_hp - self._pending_hp) <= max(1500, self._pending_hp * 0.4)):
                    c["hp"] = new_hp  # big change confirmed twice
                    self._pending_hp = None
                else:
                    self._pending_hp = new_hp
            # Ammo runs on ANY anchor, not just a freshly verified one.
            #
            # It used to sit behind `hud_readable` alongside HP and cubes, which
            # meant it inherited the whole anchor -> health -> ammo chain and
            # landed on ~15% of frames. But the ammo row is at a fixed offset
            # from the player sprite, so once `_ammo_locator` has learned that
            # offset the HP digits no longer have to parse -- a coasted anchor
            # is enough. HP and cubes still need the verified anchor, because
            # they are read by OCR at a position that must be right to the
            # pixel; a pip row only has to be found, and it is found by colour.
            ammo_anchor = c["anchor"] if c["anchor"] is not None else None
            ammo_info = (find_ammo_info(frame, ammo_anchor,
                                        health_info=health if hud_readable else None,
                                        locator=self._ammo_locator)
                         if ammo_anchor is not None
                         else {"bounding_box": None, "ammo_count": 0,
                               "detected": False})
            # Carry the count forward across frames where the bar could not be
            # located, and publish whether THIS frame actually read it. A stale
            # count is a far better estimate than the 0 that a failed read
            # returns -- and consumers that gate on "out of ammo" need to know
            # the difference (see getAmmo.find_ammo_info).
            if ammo_info["detected"]:
                c["ammo"] = ammo_info["ammo_count"]
            c["ammo_known"] = bool(ammo_info["detected"])

            # CUBE PERSISTENCE: cubes only ever increase until you die, so
            # a readable count is remembered and never overwritten by a
            # None (the HUD counter OCR is flaky) or by a lower misread.
            # Increases of 1-3 (normal pickups) are accepted immediately;
            # bigger jumps need to repeat on the next refresh (a raw
            # monotonic rule let a single "38" misread stick forever).
            # NOT gated on hud_readable, unlike HP and ammo above.
            #
            # Those two read small features at a precise offset from the
            # anchor, so a rough anchor lands them on the wrong pixels. The
            # cube badge is much larger, and find_cube_info OCRs it and returns
            # None when it cannot parse — so a bad anchor makes it fail
            # harmlessly rather than report a wrong number.
            #
            # Gating it cost every single read: measured over 87 in-match
            # frames, 6 cube counts parsed using any anchor and 0 using only
            # verified ones. cube_pickup is the reward that drives collecting
            # cubes at all, so that silently switched the behaviour off.
            new_cubes = (find_cube_info(frame, c["anchor"])["cube_count"]
                         if c["anchor"] is not None else None)
            if new_cubes is not None and new_cubes <= 20:
                prev_cubes = c["hud_cubes"] or 0
                if new_cubes > prev_cubes:
                    if new_cubes - prev_cubes <= 3:
                        c["hud_cubes"] = new_cubes
                        self._pending_cubes = None
                    elif self._pending_cubes == new_cubes:
                        c["hud_cubes"] = new_cubes
                        self._pending_cubes = None
                    else:
                        self._pending_cubes = new_cubes
            timings["player"] = time.time() - t0

        # The entity and pickup stages both need the full-screen red-bar
        # scan; when both are due on the same tick, run it once and share.
        entities_due = in_match and self._due("entities", now)
        pickups_due = in_match and self._due("pickups", now)
        bars = None
        if entities_due or pickups_due:
            t0 = time.time()
            bars = find_enemy_health_bars(frame, hsv=hsv)
            timings["bars"] = time.time() - t0

        if entities_due:
            t0 = time.time()
            ents = find_entities(frame, player_pos=c["anchor"], hsv=hsv, bars=bars)
            c["enemies"] = ents["enemies"]
            c["boxes"] = ents["boxes"]
            timings["entities"] = time.time() - t0

        if pickups_due:
            t0 = time.time()
            exclude = []
            if c["anchor"] is not None:
                exclude.append(c["anchor"])
            exclude += [(e["center"][0], e["center"][1], e["radius"]) for e in c["enemies"]]
            exclude += [(b["center"][0], b["center"][1], b["radius"]) for b in c["boxes"]]
            c["cubes"] = find_ground_cubes(frame, exclude_positions=exclude,
                                           hsv=hsv, bars=bars)
            timings["pickups"] = time.time() - t0

        if in_match and self._due("gas", now):
            t0 = time.time()
            gi = gas_info(frame, c["anchor"], hsv=hsv)   # anchor may be None: frame center
            c["in_gas"] = gi["in_gas"]
            c["gas_sides"] = gi["sides"]
            c["gas_safe"] = gi["safe_vector"]
            c["gas_frac"] = gi["frac"]
            timings["gas"] = time.time() - t0

        # SUPER charge (fixed HUD button; no anchor needed).
        if in_match and self._due("super", now):
            t0 = time.time()
            c["super_charge"] = find_super_info(frame)["charge"]
            timings["super"] = time.time() - t0

        return dict(c), timings


def draw_overlay(frame, state):
    out = frame.copy()
    if state["anchor"]:
        x, y, r = state["anchor"]
        cv2.circle(out, (x, y), r, (255, 0, 0), 3)
    for e in state["enemies"]:
        hx, hy, hw, hh = e["health"]["bounding_box"]
        cv2.rectangle(out, (hx, hy), (hx + hw, hy + hh), (0, 0, 255), 2)
    for b in state["boxes"]:
        bx, by, bw, bh = b["health"]["bounding_box"]
        cv2.rectangle(out, (bx, by), (bx + bw, by + bh), (0, 165, 255), 2)
    for cu in state["cubes"]:
        cx, cy, cw, ch = cu["bounding_box"]
        cv2.rectangle(out, (cx, cy), (cx + cw, cy + ch), (0, 255, 0), 2)
    gs = state["game_state"]
    label = (f"{gs['state']} left:{gs['brawlers_left']} HP:{state['hp']} "
             f"ammo:{state['ammo']} gas:{state['in_gas']} super:{state['super_charge']*100:.0f}%")
    cv2.putText(out, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    src_group = ap.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--video", help="simulate live feed from a recording")
    src_group.add_argument("--screen", action="store_true", help="capture the real screen (requires mss)")
    ap.add_argument("--region", type=int, nargs=4, metavar=("LEFT", "TOP", "W", "H"),
                    help="screen region to capture (with --screen)")
    ap.add_argument("--fps", type=float, default=10.0, help="target ticks per second (default 10)")
    ap.add_argument("--min-fps", type=float, default=1.0, help="floor for adaptive throttling")
    ap.add_argument("--save-preview", action="store_true",
                    help="write annotated frame to debugOutput/live_preview.png each second")
    ap.add_argument("--show", action="store_true", help="cv2.imshow window (needs a display)")
    ap.add_argument("--duration", type=float, default=None, help="stop after N seconds")
    args = ap.parse_args()

    source = VideoSource(args.video) if args.video else ScreenSource(args.region)
    perception = LivePerception()

    target_fps = args.fps
    current_fps = target_fps
    loop_times = []
    started = time.time()
    last_preview = 0.0
    ticks = 0
    misses = 0

    try:
        while True:
            if args.duration and time.time() - started > args.duration:
                break
            tick_start = time.time()

            frame = source.grab()
            if frame is None:
                # Recording ended -> stop. Live feed hiccup -> skip and retry
                # (up to a long streak, in case the device really disconnected).
                if not getattr(source, "live", False):
                    print("recording ended")
                    break
                misses += 1
                if misses > 90:
                    print("live source: too many dropped frames, stopping")
                    break
                time.sleep(0.05)
                continue
            misses = 0

            state, timings = perception.tick(frame)
            ticks += 1

            gs = state["game_state"]
            timing_str = " ".join(f"{k}:{v*1000:.0f}ms" for k, v in timings.items())
            print(
                f"[{time.time()-started:6.1f}s @{current_fps:4.1f}fps] "
                f"{gs['state']}"
                + (f" left:{gs['brawlers_left']}" if gs["brawlers_left"] else "")
                + (f" rank:{gs['rank']}" if gs.get("rank") else "")
                + (f" | HP:{state['hp']} ammo:{state['ammo']} cubes:{state['hud_cubes']}"
                   f" super:{state['super_charge']*100:.0f}%"
                   f" enemies:{len(state['enemies'])} boxes:{len(state['boxes'])}"
                   f" ground_cubes:{len(state['cubes'])} gas:{state['in_gas']}"
                   if gs["state"] == "in_match" else "")
                + (f"  ({timing_str})" if timing_str else ""),
                flush=True,
            )

            if args.save_preview and time.time() - last_preview >= 1.0:
                _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(_DEBUG_DIR / "live_preview.png"), draw_overlay(frame, state))
                last_preview = time.time()
            if args.show:
                cv2.imshow("live", cv2.resize(draw_overlay(frame, state), None, fx=0.5, fy=0.5))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # --- adaptive rate control ---
            loop_time = time.time() - tick_start
            loop_times.append(loop_time)
            if len(loop_times) > 10:
                loop_times.pop(0)
            avg = sum(loop_times) / len(loop_times)
            budget = 1.0 / current_fps
            if avg > budget * 0.95 and current_fps > args.min_fps:
                current_fps = max(args.min_fps, current_fps * 0.8)
                print(f"  !! processing avg {avg*1000:.0f}ms > budget, throttling to {current_fps:.1f}fps", flush=True)
            elif avg < budget * 0.7 and current_fps < target_fps:
                # Recover quickly once there's headroom -- the old 1.1x ramp
                # with a 0.5x threshold took ~30s to climb back after one
                # slow stretch (e.g. the startup frames).
                current_fps = min(target_fps, current_fps * 1.25)

            sleep_for = (1.0 / current_fps) - (time.time() - tick_start)
            if sleep_for > 0:
                time.sleep(sleep_for)
    finally:
        source.close()
        elapsed = time.time() - started
        print(f"\n{ticks} ticks in {elapsed:.1f}s = {ticks/max(0.01,elapsed):.1f} effective fps")


if __name__ == "__main__":
    main()
