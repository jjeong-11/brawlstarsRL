"""
rl/actions.py
=============

What the policy can do, and how a chosen action becomes real touch input on the
Android device running Brawl Stars.

Action space (tap-based, NO aiming) — ``MultiDiscrete([16, 3, 2, 2])``:

    head 0  heading  : 0-15, clockwise from screen-right (22.5 deg per step)
    head 1  distance : 0 near · 1 mid · 2 far — how far to commit
    head 2  attack   : 0 no · 1 tap   (auto-aims at the nearest enemy)
    head 3  super    : 0 no · 1 tap   (auto-aims at the nearest enemy)

The policy chooses *where* to go, and the planner in ``rl/path_planner.py``
latches that destination in world space and routes to it with A* over the
terrain grid until it is reached. Attack and super remain independent taps, and
unlike the movement heads they are honoured on EVERY step — the agent can keep
shooting while walking a committed route.

WHY POLAR AND NOT A GRID
    An earlier version used a 15x15 player-centred grid: 225 movement actions,
    of which the corner cells were 1.41x farther than the edge cells for no
    reason and the ~9 cells around the centre were indistinguishable tiny
    nudges. Polar spends its resolution where it means something — 16 evenly
    spaced headings and an explicit "how far do I commit" — for 19 logits
    instead of 30, and every combination is distinct and reachable.

The ``ActionExecutor`` is the boundary to the device:
    * LoggingExecutor — no hardware; records intents (dry runs / tests).
    * AdbExecutor     — functional: shells out to `adb` to tap/swipe the on-screen
                        controls. See docs/ANDROID_CONTROL.md for setup, coordinate
                        calibration, and lower-latency alternatives.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .path_planner import WaypointPlanner, PathPlannerConfig

try:
    from gymnasium import spaces
except Exception:
    spaces = None  # gym optional at import time

N_HEADINGS = PathPlannerConfig().n_headings
N_DISTANCES = len(PathPlannerConfig().distances)

# Human-readable compass labels for the 16 headings, purely for logging.
_COMPASS = ("E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW",
            "W", "WNW", "NW", "NNW", "N", "NNE", "NE", "ENE")
_DIST_LABEL = ("near", "mid", "far")


def make_action_space():
    nvec = [N_HEADINGS, N_DISTANCES, 2, 2]
    return spaces.MultiDiscrete(nvec) if spaces else {"type": "MultiDiscrete", "nvec": nvec}


class Intent:
    """A decoded, human-readable action: a move direction + attack/super taps."""

    __slots__ = ("move_label", "move", "fire_attack", "fire_super", "raw")

    def __init__(self, move_label, move, fire_attack, fire_super, raw):
        self.move_label = move_label     # e.g. "NE/far" or "NE/far(committed)"
        self.move = move                 # (dx, dy) unit vector
        self.fire_attack = fire_attack   # bool (plain tap, auto-aim)
        self.fire_super = fire_super     # bool (plain tap, auto-aim)
        self.raw = raw

    def __repr__(self):
        parts = [self.move_label]
        if self.fire_attack:
            parts.append("attack")
        if self.fire_super:
            parts.append("SUPER")
        return f"Intent({'+'.join(parts)})"


def decode_action(action, state=None, frame_size=(1280, 720),
                  planner: Optional[WaypointPlanner] = None,
                  terrain=None, camera_delta=(0.0, 0.0), gas_grid=None) -> Intent:
    """Decode ``[heading, distance, attack, super]`` into an Intent.

    `terrain`, `gas_grid` and `camera_delta` come from the env's per-tick
    perception; when omitted the planner degrades to direct steering with no
    world latching, which is fine for unit tests but not for real play.

    `gas_grid` is passed separately from `terrain` on purpose: gas detection is
    map-independent (it is an engine overlay, not map art) while terrain needs a
    matching colour profile, so on an uncalibrated map gas is available when
    terrain is not. Folding it into `terrain` would throw that away.
    """
    if not hasattr(action, "__len__") or len(action) < 4:
        raise ValueError("expected polar action [heading, distance, attack, super]")
    heading, dist, attack, super_ = (int(action[0]), int(action[1]),
                                     int(action[2]), int(action[3]))
    heading %= N_HEADINGS
    dist = max(0, min(N_DISTANCES - 1, dist))

    planner = planner or WaypointPlanner()
    move = planner.plan(heading, dist, state, frame_size, terrain=terrain,
                        camera_delta=camera_delta, gas_grid=gas_grid)

    status = planner.status()
    label = f"{_COMPASS[heading]}/{_DIST_LABEL[dist]}"
    if status.escaping_gas:
        label += "(GAS-ESCAPE)"     # the plan was overridden to flee the cloud
    elif status.active and not status.replanned:
        label += "(committed)"      # the heads above were ignored this step
    if status.blocked:
        label += "(blocked)"
    return Intent(label, move, attack == 1, super_ == 1, (heading, dist, attack, super_))


class ActionExecutor:
    """Interface: consume an action, drive the device."""

    def apply(self, action, state=None, frame_size=(1280, 720), planner=None,
              terrain=None, camera_delta=(0.0, 0.0), gas_grid=None) -> Intent:
        intent = decode_action(action, state=state, frame_size=frame_size,
                               planner=planner, terrain=terrain,
                               camera_delta=camera_delta, gas_grid=gas_grid)
        self._execute(intent)
        return intent

    def _execute(self, intent: Intent) -> None:  # pragma: no cover
        raise NotImplementedError

    def close(self) -> None:
        pass


class LoggingExecutor(ActionExecutor):
    """No hardware: records/prints intents. For wiring and tests."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.history = []

    def _execute(self, intent: Intent) -> None:
        self.history.append(intent)
        if self.verbose:
            print(intent)

    def tap_norm(self, fx: float, fy: float) -> None:
        self.history.append(("tap_norm", fx, fy))
        if self.verbose:
            print("tap_norm", fx, fy)


@dataclass
class Controls:
    """On-screen control positions in DEVICE landscape pixels (calibrate once).

    move_center : center of the left movement joystick
    move_radius : how far to drag the stick from center (bigger = faster move)
    attack_btn  : center of the attack button
    super_btn   : center of the super button (the skull button getSuper reads)
    hold_ms     : movement drag duration per tick (a tap won't move; a held drag does)
    """
    move_center: Tuple[int, int]
    attack_btn: Tuple[int, int]
    super_btn: Tuple[int, int]
    move_radius: int = 170
    hold_ms: int = 200

    @staticmethod
    def from_screen(width: int, height: int) -> "Controls":
        """A first guess from the landscape resolution. CALIBRATE before trusting.

        Fractions are rough Brawl Stars defaults: movement stick bottom-left,
        attack bottom-right, super just up-left of attack.
        """
        w, h = max(width, height), min(width, height)  # force landscape
        return Controls(
            move_center=(int(0.16 * w), int(0.80 * h)),
            attack_btn=(int(0.88 * w), int(0.82 * h)),
            super_btn=(int(0.80 * w), int(0.66 * h)),
            move_radius=int(0.10 * w),
            # Was 200, which silently capped the whole loop at 5 fps and
            # accumulated action latency behind it. 100 matches the default
            # 0.1s tick; tuned_for_tick() adjusts it if you change --tick-seconds.
            hold_ms=100,
        )

    # A swipe shorter than this is liable to be interpreted as a tap/fling
    # rather than a drag, so the joystick never deflects.
    MIN_HOLD_MS = 30

    def tuned_for_tick(self, tick_seconds: float) -> "Controls":
        """Copy with `hold_ms` clamped so movement swipes TILE instead of QUEUE.

        `input swipe ... <hold_ms>` BLOCKS the device shell for hold_ms. Issue
        one every tick and the sustainable rate is capped at 1000/hold_ms fps —
        at the 200ms default, 5 fps, no matter what the loop asks for.

        Worse, the overflow is invisible for a while. Commands go down a
        persistent `adb shell` pipe, so a backlog first fills the 64KB stdin
        buffer (costing nothing measurable) and only starts blocking once it is
        full. Measured: at a 0.1s tick, `act` profiled at 1.1ms while silently
        running 2x oversubscribed and accumulating latency; dropping to a 0.05s
        tick made it 4x, saturated the buffer, and `act` jumped to 329ms — 84%
        of the step. The loop got SLOWER by asking for more.

        The latency matters more than the throughput: a growing backlog means
        the game executes an action many steps after the policy chose it, so the
        reward gets attributed to the wrong action.

        Setting hold_ms to the tick period gives a ~100% duty cycle — the finger
        is down almost continuously, movement stays smooth, and nothing queues.
        """
        from dataclasses import replace
        if not tick_seconds or tick_seconds <= 0:
            return self
        budget = max(self.MIN_HOLD_MS, int(tick_seconds * 1000))
        if self.hold_ms <= budget:
            return self
        print(f"NOTE: hold_ms {self.hold_ms}ms exceeds the {1000 * tick_seconds:.0f}ms "
              f"tick, which would queue swipes and cap the loop at "
              f"{1000 / self.hold_ms:.1f} fps. Using {budget}ms.\n"
              f"      For genuinely continuous movement use --sendevent, which "
              f"holds the touch instead of re-swiping every tick.")
        return replace(self, hold_ms=budget)

    @classmethod
    def from_json(cls, path: str) -> "Controls":
        """Load calibrated coordinates saved earlier (see save())."""
        with open(path) as f:
            d = json.load(f)
        return cls(
            move_center=tuple(d["move_center"]),
            attack_btn=tuple(d["attack_btn"]),
            super_btn=tuple(d["super_btn"]),
            move_radius=int(d.get("move_radius", 170)),
            hold_ms=int(d.get("hold_ms", 200)),
        )

    def save(self, path: str) -> None:
        """Persist these coordinates so watch/train can reuse them."""
        with open(path, "w") as f:
            json.dump({
                "move_center": list(self.move_center),
                "attack_btn": list(self.attack_btn),
                "super_btn": list(self.super_btn),
                "move_radius": self.move_radius,
                "hold_ms": self.hold_ms,
            }, f, indent=2)


def intent_to_touches(intent: Intent, c: Controls) -> List[tuple]:
    """PURE mapping: Intent -> ordered touch ops. Unit-testable without a device.

    Returns a list of ("tap", x, y) and ("swipe", x0, y0, x1, y1, ms) tuples.
    Movement is a held drag on the joystick; attack/super are taps (auto-aim).
    """
    ops: List[tuple] = []
    if intent.move != (0.0, 0.0):
        cx, cy = c.move_center
        tx = int(cx + intent.move[0] * c.move_radius)
        ty = int(cy + intent.move[1] * c.move_radius)
        ops.append(("swipe", int(cx), int(cy), tx, ty, c.hold_ms))
    if intent.fire_attack:
        ops.append(("tap", int(c.attack_btn[0]), int(c.attack_btn[1])))
    if intent.fire_super:
        ops.append(("tap", int(c.super_btn[0]), int(c.super_btn[1])))
    return ops


class AdbExecutor(ActionExecutor):
    """Drives a real Android device (e.g. a Pixel 10) over `adb`.

    Sends taps/swipes to the on-screen controls. Uses a PERSISTENT `adb shell`
    (commands piped to one long-lived shell) to avoid paying adb's process-spawn
    cost on every action; falls back to one-shot `adb` if that shell dies.

    Coordinates are in the device's landscape touch pixels and MUST be calibrated
    for your phone (see docs/ANDROID_CONTROL.md). If `controls` is omitted, they're
    guessed from `adb shell wm size` — a starting point, not a substitute for
    calibration.

    Movement is CONTINUOUS: the joystick touch is pressed once (`input motionevent
    DOWN`), kept held and only MOVEd to change direction, and UPed to stop. This is
    what prevents the start/stop "bursty" motion that re-swiping every tick causes
    (a swipe lifts the finger each time, and the brawler stops during the capture
    gap). Set `hold_movement=False` to fall back to per-tick swipes.

    Movement mode (`hold_movement`):
      * False (default) — each move is one `input swipe` from the joystick center
        out to the direction. Reliable everywhere, but motion is BURSTY: the finger
        lifts at the end of each swipe and again during the capture gap.
      * True — tries to hold the joystick via `input motionevent` DOWN/MOVE/UP.
        DON'T rely on this: separate `adb input` calls are independent injections,
        so on most devices the DOWN/MOVE aren't linked into one gesture and the
        touch doesn't persist -> the brawler barely moves. Kept only for devices
        where it happens to work.

    For genuinely SMOOTH continuous movement (and true simultaneous move+shoot),
    pure `adb` is not enough — use scrcpy control (persistent touch socket). See
    docs/ANDROID_CONTROL.md.
    """

    def __init__(self, serial: Optional[str] = None, controls: Optional[Controls] = None,
                 persistent: bool = True, hold_movement: bool = False):
        if shutil.which("adb") is None:
            raise RuntimeError("`adb` not found on PATH. Install platform-tools and "
                               "connect your phone (USB debugging on): `adb devices`.")
        self.serial = serial
        self.screen_size = self._detect_size()          # (w, h) landscape
        self.controls = controls or Controls.from_screen(*self.screen_size)
        self.hold_movement = hold_movement
        self._move_down = False                          # is the joystick touch held?
        self._move_pos = None                            # last held (x, y)
        self._shell = None
        if persistent:
            self._open_shell()

    # -- adb plumbing --
    def _base(self):
        return ["adb"] + (["-s", self.serial] if self.serial else [])

    def _detect_size(self) -> Tuple[int, int]:
        try:
            out = subprocess.run(self._base() + ["shell", "wm", "size"],
                                 capture_output=True, text=True, timeout=5).stdout
            # "Physical size: 1080x2400" (may also show "Override size:")
            line = [l for l in out.splitlines() if "size:" in l][-1]
            wxh = line.split(":")[-1].strip().lower().split("x")
            w, h = int(wxh[0]), int(wxh[1])
            return (max(w, h), min(w, h))               # landscape
        except Exception:
            return (2400, 1080)                          # safe fallback

    def _open_shell(self):
        try:
            self._shell = subprocess.Popen(
                self._base() + ["shell"], stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
        except Exception:
            self._shell = None

    def _send(self, line: str):
        """Send one shell command line, via the persistent shell if alive."""
        if self._shell is not None and self._shell.poll() is None:
            try:
                self._shell.stdin.write(line + "\n")
                self._shell.stdin.flush()
                return
            except Exception:
                self._shell = None  # fall through to one-shot
        subprocess.run(self._base() + ["shell"] + line.split(), check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _tap(self, x, y):
        self._send(f"input tap {int(x)} {int(y)}")

    def _swipe(self, x0, y0, x1, y1, ms):
        self._send(f"input swipe {int(x0)} {int(y0)} {int(x1)} {int(y1)} {int(ms)}")

    def _motionevent(self, kind, x, y):
        # kind is DOWN / MOVE / UP. The pointer stays down between DOWN and UP,
        # even across separate `input` invocations -> continuous joystick hold.
        self._send(f"input motionevent {kind} {int(x)} {int(y)}")

    def _release_move(self):
        if self._move_down:
            x, y = self._move_pos or self.controls.move_center
            self._motionevent("UP", x, y)
            self._move_down = False
            self._move_pos = None

    # -- intent -> touches --
    def _execute(self, intent: Intent) -> None:
        c = self.controls
        if not self.hold_movement:
            # Legacy per-tick swipes (bursty). Kept as a fallback.
            for op in intent_to_touches(intent, c):
                if op[0] == "tap":
                    self._tap(op[1], op[2])
                else:
                    self._swipe(op[1], op[2], op[3], op[4], op[5])
            return

        # Continuous movement: press the stick at its CENTER, then hold and drag
        # toward the direction. Pressing DOWN directly at the deflected offset makes
        # a *floating* joystick recenter there -> zero deflection -> the brawler
        # barely moves. So always DOWN at center first, then MOVE out to the offset
        # (and keep MOVEing to redirect while held). UP to stop.
        if intent.move != (0.0, 0.0):
            cx, cy = c.move_center
            tx = int(cx + intent.move[0] * c.move_radius)
            ty = int(cy + intent.move[1] * c.move_radius)
            if not self._move_down:
                self._motionevent("DOWN", cx, cy)     # grab the stick at its center
                self._move_down = True
            self._motionevent("MOVE", tx, ty)          # deflect toward the direction
            self._move_pos = (tx, ty)
        else:
            self._release_move()

        # Attack / super are taps (auto-aim). See the multitouch caveat above.
        if intent.fire_attack:
            self._tap(*c.attack_btn)
        if intent.fire_super:
            self._tap(*c.super_btn)

    def tap_norm(self, fx: float, fy: float) -> None:
        """Tap at a position given as fractions of the landscape screen.

        Used by env._navigate_to_match to press menu buttons (Exit / Play Again).
        Uses a ONE-SHOT `adb shell input tap` (exactly like a manual tap) instead of
        the persistent shell — more reliable for these infrequent, timing-sensitive
        menu presses.
        """
        w, h = self.screen_size
        x, y = int(fx * w), int(fy * h)
        subprocess.run(self._base() + ["shell", "input", "tap", str(x), str(y)],
                       check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def calibrate(self):
        """Tap each configured control once so you can verify placement on-device."""
        c = self.controls
        print(f"screen {self.screen_size} | controls {c}")
        for name, pt in (("attack", c.attack_btn), ("super", c.super_btn)):
            print("tap", name, pt)
            self._tap(*pt)
        print("swipe move (right)")
        self._swipe(c.move_center[0], c.move_center[1],
                    c.move_center[0] + c.move_radius, c.move_center[1], c.hold_ms)

    def close(self):
        try:
            self._release_move()      # don't leave the joystick held down
        except Exception:
            pass
        if self._shell is not None:
            try:
                self._shell.stdin.close()
                self._shell.terminate()
            except Exception:
                pass
