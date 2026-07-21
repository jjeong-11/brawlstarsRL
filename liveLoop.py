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
import time

import cv2
import numpy as np

from getAnchor import find_player_position
from getHealth import find_health_info
from getAmmo import find_ammo_info
from getCube import find_cube_info
from getEnemies import find_entities
from getPickups import find_ground_cubes
from getGas import is_player_in_gas
from getGameState import get_game_state


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
}


class VideoSource:
    """Plays a recording back as if it were a live feed: each grab returns

    the frame at the current wall-clock position, skipping past frames if
    the consumer is slow (exactly how a live capture behaves).
    """

    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open {path}")
        self.start = time.time()
        self.duration = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(
            1.0, self.cap.get(cv2.CAP_PROP_FPS)
        )

    def grab(self):
        elapsed = time.time() - self.start
        if elapsed > self.duration:
            return None
        self.cap.set(cv2.CAP_PROP_POS_MSEC, elapsed * 1000)
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        self.cap.release()


class ScreenSource:
    """Captures the real screen via mss. `region` is (left, top, width,

    height); None captures the primary monitor. Install with `pip install mss`.
    """

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

    def __init__(self):
        self.last_run = {name: 0.0 for name in STAGE_INTERVALS}
        self.cache = {
            "game_state": {"state": "unknown", "brawlers_left": None, "rank": None},
            "anchor": None,
            "hp": None,
            "ammo": None,
            "hud_cubes": None,
            "enemies": [],
            "boxes": [],
            "cubes": [],
            "in_gas": False,
        }
        self._pending_state = None   # for game-state smoothing
        self._pending_anchor = None  # for anchor-jump confirmation
        self._anchor_time = 0.0
        self._pending_hp = None      # for large-HP-change confirmation
        self._pending_cubes = None   # for large-cube-jump confirmation

    def _due(self, name, now):
        if now - self.last_run[name] >= STAGE_INTERVALS[name]:
            self.last_run[name] = now
            return True
        return False

    def tick(self, frame):
        now = time.time()
        timings = {}
        c = self.cache

        if self._due("game_state", now):
            t0 = time.time()
            raw = get_game_state(frame)
            timings["game_state"] = time.time() - t0
            # Smoothing: require the same NEW state twice in a row before
            # switching; brawlers_left updates immediately while in_match.
            if raw["state"] == c["game_state"]["state"]:
                c["game_state"] = raw
                self._pending_state = None
            elif self._pending_state is not None and raw["state"] == self._pending_state["state"]:
                c["game_state"] = raw
                self._pending_state = None
            else:
                self._pending_state = raw

        in_match = c["game_state"]["state"] == "in_match"

        # Reset per-life values when a match ends / a new one loads.
        if not in_match and c["game_state"]["state"] in ("loading", "match_end"):
            c["hud_cubes"] = None
            c["hp"] = None
            c["anchor"] = None

        if in_match and self._due("player", now):
            t0 = time.time()
            raw_anchor = find_player_position(frame)

            # ANCHOR PERSISTENCE: the player can't teleport. If the fresh
            # anchor jumped far from the recent one (mid-combat chaos
            # frames produce these), keep the old anchor unless the jump
            # repeats on the next refresh too.
            w = frame.shape[1]
            prev = c["anchor"]
            if prev is not None and now - self._anchor_time < 4.0:
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
            self._anchor_time = now

            health = find_health_info(frame, c["anchor"])
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
            # Pass health through so ammo doesn't redo the digit search.
            c["ammo"] = find_ammo_info(frame, c["anchor"], health_info=health)["ammo_count"]

            # CUBE PERSISTENCE: cubes only ever increase until you die, so
            # a readable count is remembered and never overwritten by a
            # None (the HUD counter OCR is flaky) or by a lower misread.
            # Increases of 1-3 (normal pickups) are accepted immediately;
            # bigger jumps need to repeat on the next refresh (a raw
            # monotonic rule let a single "38" misread stick forever).
            new_cubes = find_cube_info(frame, c["anchor"])["cube_count"]
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

        if in_match and self._due("entities", now):
            t0 = time.time()
            ents = find_entities(frame, player_pos=c["anchor"])
            c["enemies"] = ents["enemies"]
            c["boxes"] = ents["boxes"]
            timings["entities"] = time.time() - t0

        if in_match and self._due("pickups", now):
            t0 = time.time()
            exclude = []
            if c["anchor"] is not None:
                exclude.append(c["anchor"])
            exclude += [(e["center"][0], e["center"][1], e["radius"]) for e in c["enemies"]]
            exclude += [(b["center"][0], b["center"][1], b["radius"]) for b in c["boxes"]]
            c["cubes"] = find_ground_cubes(frame, exclude_positions=exclude)
            timings["pickups"] = time.time() - t0

        if in_match and c["anchor"] is not None and self._due("gas", now):
            t0 = time.time()
            c["in_gas"] = is_player_in_gas(frame, c["anchor"])
            timings["gas"] = time.time() - t0

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
    label = f"{gs['state']} left:{gs['brawlers_left']} HP:{state['hp']} ammo:{state['ammo']} gas:{state['in_gas']}"
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

    try:
        while True:
            if args.duration and time.time() - started > args.duration:
                break
            tick_start = time.time()

            frame = source.grab()
            if frame is None:
                print("source ended")
                break

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
                   f" enemies:{len(state['enemies'])} boxes:{len(state['boxes'])}"
                   f" ground_cubes:{len(state['cubes'])} gas:{state['in_gas']}"
                   if gs["state"] == "in_match" else "")
                + (f"  ({timing_str})" if timing_str else ""),
                flush=True,
            )

            if args.save_preview and time.time() - last_preview >= 1.0:
                cv2.imwrite("debugOutput/live_preview.png", draw_overlay(frame, state))
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
