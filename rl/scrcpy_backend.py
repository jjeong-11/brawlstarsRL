"""
rl/scrcpy_backend.py
====================

scrcpy-based control + capture — the real fix for smooth movement.

Pure `adb input` can't hold a touch across calls, so movement is always bursty and
move+shoot can't truly overlap. scrcpy keeps a persistent control socket to the
phone, so a touch can be pressed once and *held* (continuous joystick), and
multiple fingers (distinct touch_ids) can be down at the same time (move AND shoot).
It also streams video, so capture is fast and low-latency instead of ~300ms
`screencap` grabs.

Requires:  pip install scrcpy-client      (bundles the scrcpy server)
           a phone connected via `adb` (USB debugging on).

Usage (see also train_rl.py --scrcpy):

    from rl.scrcpy_backend import make_scrcpy_env
    from rl.actions import Controls
    env = make_scrcpy_env(serial="57230DLCR000M4",
                          controls=Controls.from_json("controls.json"),
                          tick_seconds=0.05)

NOTE: this talks to a third-party library and a real device, neither of which can
be exercised in the dev sandbox — the touch *logic* is unit-tested with a mock,
but if a call signature is off on your scrcpy-client version, it'll be a small
tweak in ScrcpyBackend / ScrcpyExecutor.
"""

from __future__ import annotations

import threading
from typing import Optional

from .actions import ActionExecutor, Controls, Intent

try:
    import scrcpy  # pip install scrcpy-client
except Exception:  # keep this module importable without the lib installed
    scrcpy = None

# scrcpy touch actions (same integer values the scrcpy protocol uses). Defined
# locally so the executor logic is testable without the library present.
ACTION_DOWN, ACTION_UP, ACTION_MOVE = 0, 1, 2


class ScrcpyBackend:
    """One shared scrcpy client: streams frames and sends control events."""

    def __init__(self, serial: Optional[str] = None, max_fps: int = 15,
                 bitrate: int = 8_000_000):
        if scrcpy is None:
            raise RuntimeError("scrcpy-client not installed. Run: pip install scrcpy-client")
        # max_width=0 -> no downscale, so touch coords match device pixels
        # (and your controls.json, calibrated on a full-res screencap).
        self.client = scrcpy.Client(device=serial, max_width=0,
                                    max_fps=max_fps, bitrate=bitrate, block_frame=False)
        self._frame = None
        self._lock = threading.Lock()
        self.client.add_listener(scrcpy.EVENT_FRAME, self._on_frame)
        self.client.start(threaded=True, daemon_threaded=True)

    def _on_frame(self, frame):
        if frame is not None:                 # frame is a BGR numpy array
            with self._lock:
                self._frame = frame

    def latest_frame(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def touch(self, x, y, action, touch_id):
        try:
            self.client.control.touch(int(x), int(y), action, touch_id)
        except OSError as e:
            raise RuntimeError(
                "scrcpy control socket is dead. The scrcpy-client pip package "
                "bundles an old scrcpy-server that's incompatible with this phone's "
                "Android version (its video stream drops and kills control). Use the "
                "adb path (drop --scrcpy), or the modern-scrcpy control-only client."
            ) from e

    def resolution(self):
        return getattr(self.client, "resolution", None)

    def stop(self):
        try:
            self.client.stop()
        except Exception:
            pass


class ScrcpySource:
    """Frame source backed by the scrcpy video stream (fast, low-latency)."""

    live = True

    def __init__(self, backend: ScrcpyBackend):
        self.backend = backend

    def grab(self):
        return self.backend.latest_frame()   # None until the first frame arrives

    def close(self):
        pass                                  # backend is owned/closed by the executor


class ScrcpyExecutor(ActionExecutor):
    """Held-touch movement + true multitouch attack/super, over scrcpy.

    Movement uses touch_id 0, held DOWN and MOVEd across ticks (continuous). Attack
    and super use their own touch_ids, so they fire WITHOUT interrupting movement.
    """

    MOVE_ID, ATTACK_ID, SUPER_ID = 0, 1, 2

    def __init__(self, backend: ScrcpyBackend, controls: Controls):
        self.backend = backend
        self.controls = controls
        self._move_down = False

    def _tap(self, x, y, touch_id):
        self.backend.touch(x, y, ACTION_DOWN, touch_id)
        self.backend.touch(x, y, ACTION_UP, touch_id)

    def _release_move(self):
        if self._move_down:
            cx, cy = self.controls.move_center
            self.backend.touch(cx, cy, ACTION_UP, self.MOVE_ID)
            self._move_down = False

    def _execute(self, intent: Intent) -> None:
        c = self.controls
        # Movement: press at the joystick CENTER, hold, and MOVE out to the
        # direction (a held, continuous deflection). UP to stop.
        if intent.move != (0.0, 0.0):
            cx, cy = c.move_center
            tx = int(cx + intent.move[0] * c.move_radius)
            ty = int(cy + intent.move[1] * c.move_radius)
            if not self._move_down:
                self.backend.touch(cx, cy, ACTION_DOWN, self.MOVE_ID)
                self._move_down = True
            self.backend.touch(tx, ty, ACTION_MOVE, self.MOVE_ID)
        else:
            self._release_move()

        # Attack / super on separate touch_ids -> simultaneous with movement.
        if intent.fire_attack:
            ax, ay = c.attack_btn
            self._tap(ax, ay, self.ATTACK_ID)
        if intent.fire_super:
            sx, sy = c.super_btn
            self._tap(sx, sy, self.SUPER_ID)

    def calibrate(self):
        c = self.controls
        print(f"controls {c}")
        ax, ay = c.attack_btn
        self._tap(ax, ay, self.ATTACK_ID)
        sx, sy = c.super_btn
        self._tap(sx, sy, self.SUPER_ID)
        # a brief held drag right, then release
        cx, cy = c.move_center
        self.backend.touch(cx, cy, ACTION_DOWN, self.MOVE_ID)
        self.backend.touch(cx + c.move_radius, cy, ACTION_MOVE, self.MOVE_ID)
        self.backend.touch(cx + c.move_radius, cy, ACTION_UP, self.MOVE_ID)

    def close(self):
        self._release_move()
        self.backend.stop()


def make_scrcpy_env(serial: Optional[str] = None, controls: Optional[Controls] = None,
                    reward_config=None, tick_seconds: float = 0.05, **kwargs):
    """Live env using scrcpy for BOTH capture and control (the smooth path)."""
    from .env import BrawlStarsEnv
    backend = ScrcpyBackend(serial)
    if controls is None:
        res = backend.resolution() or (2400, 1080)
        controls = Controls.from_screen(res[0], res[1])
    source = ScrcpySource(backend)
    executor = ScrcpyExecutor(backend, controls)
    return BrawlStarsEnv(source_factory=lambda: source, executor=executor,
                         reward_config=reward_config, tick_seconds=tick_seconds, **kwargs)
