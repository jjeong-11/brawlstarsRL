"""
rl/sendevent_backend.py
=======================

Smooth movement + real multitouch on the phone using RAW touch events — no scrcpy,
no PyAV, no ffmpeg. Just `adb`.

`adb shell input` can't hold a touch across calls (each is an independent gesture),
which is why movement was bursty / didn't move. `sendevent` writes raw events
straight to the touch digitizer, and the kernel keeps a contact "down" across
separate calls (type-B multitouch, using ABS_MT_SLOT + tracking IDs). So we get:

  * continuous movement  — press the joystick once (slot 0) and just keep MOVEing,
  * simultaneous attack  — tap on a DIFFERENT slot without releasing the joystick.

Calibrated from `adb shell getevent -pl` for this Pixel's panel:
    device /dev/input/event2  "focal_ts"
    ABS_MT_POSITION_X : 0..10799     (portrait short axis)
    ABS_MT_POSITION_Y : 0..24239     (portrait long axis)

The digitizer reports coordinates in the panel's fixed PORTRAIT orientation, while
your controls.json is in LANDSCAPE display pixels — so we swap axes and (maybe)
flip. Which flip is correct depends on which way the phone is rotated; run
``SendeventExecutor(...).calibrate()`` once and it will tell you (see below).
"""

from __future__ import annotations

import shutil
import subprocess
import time
from typing import Optional

from .actions import ActionExecutor, Controls, Intent

# --- device profile (from getevent -pl) --------------------------------------
# Defaults measured on one Pixel. `detect_touch_device()` overrides them from
# the phone at startup — writing raw events to the WRONG /dev/input node either
# does nothing or drives some other sensor, and with the wrong ABS ranges every
# touch lands in the wrong place, both of which fail silently.
TOUCH_DEVICE = "/dev/input/event2"
ABS_X_MAX = 10799     # portrait short axis  <- landscape HEIGHT
ABS_Y_MAX = 24239     # portrait long axis   <- landscape WIDTH


def detect_touch_device(serial: Optional[str] = None):
    """Find the touchscreen and its coordinate ranges via `getevent -pl`.

    Returns (device_path, abs_x_max, abs_y_max), falling back to the measured
    Pixel defaults if anything cannot be parsed. The touchscreen is the node
    advertising ABS_MT_POSITION_X/Y — that is what makes it a multitouch
    digitizer, as opposed to buttons, sensors or a virtual keyboard.
    """
    base = ["adb"] + (["-s", serial] if serial else [])
    try:
        out = subprocess.run(base + ["shell", "getevent", "-pl"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return TOUCH_DEVICE, ABS_X_MAX, ABS_Y_MAX

    def axis_max(line):
        # "    ABS_MT_POSITION_X   : value 0, min 0, max 10799, fuzz 0, flat 0"
        for part in line.split(","):
            part = part.strip()
            if part.startswith("max "):
                try:
                    return int(part.split()[1])
                except (IndexError, ValueError):
                    return None
        return None

    device = None
    axes = {}
    for line in out.splitlines():
        stripped = line.strip()
        if stripped.startswith("add device"):
            if device and "x" in axes and "y" in axes:
                break                       # the previous device was the digitizer
            device = stripped.split(":")[-1].strip()   # "add device 1: /dev/input/event2"
            axes = {}
        elif "ABS_MT_POSITION_X" in stripped:
            v = axis_max(stripped)
            if v:
                axes["x"] = v
        elif "ABS_MT_POSITION_Y" in stripped:
            v = axis_max(stripped)
            if v:
                axes["y"] = v

    if device and "x" in axes and "y" in axes:
        return device, axes["x"], axes["y"]
    return TOUCH_DEVICE, ABS_X_MAX, ABS_Y_MAX

# Landscape display size your controls.json / screencap are in.
LAND_W, LAND_H = 2424, 1080

# evdev type/code numbers (Linux input; same on Android).
EV_SYN, EV_KEY, EV_ABS = 0, 1, 3
SYN_REPORT = 0
BTN_TOUCH = 330                # 0x14a
ABS_MT_SLOT = 47               # 0x2f
ABS_MT_TRACKING_ID = 57        # 0x39
ABS_MT_POSITION_X = 53         # 0x35
ABS_MT_POSITION_Y = 54         # 0x36


class SendeventExecutor(ActionExecutor):
    """Raw-multitouch executor: held movement (slot 0) + tap attack/super (slots 1/2)."""

    MOVE_SLOT, ATTACK_SLOT, SUPER_SLOT = 0, 1, 2

    def __init__(self, serial: Optional[str] = None, controls: Optional[Controls] = None,
                 flip_short: bool = False, flip_long: bool = True,
                 device: Optional[str] = None, persistent: bool = True):
        if shutil.which("adb") is None:
            raise RuntimeError("`adb` not found on PATH — connect your phone first.")
        self.controls = controls or Controls.from_screen(LAND_W, LAND_H)
        self.flip_short = flip_short     # mirror along the short axis (landscape Y)
        self.flip_long = flip_long       # mirror along the long axis (landscape X)

        # Ask the phone which node is the digitizer and what its ranges are,
        # rather than trusting constants measured on one Pixel. Both failure
        # modes here are SILENT: the wrong node swallows the events, and wrong
        # ABS ranges put every touch in the wrong place. `device=None` means
        # auto-detect; pass an explicit path to override.
        self.abs_x_max, self.abs_y_max = ABS_X_MAX, ABS_Y_MAX
        if device is None:
            self.device, self.abs_x_max, self.abs_y_max = detect_touch_device(serial)
            print(f"[sendevent] touch device {self.device} "
                  f"(ABS_MT_POSITION_X max {self.abs_x_max}, Y max {self.abs_y_max})")
        else:
            self.device = device
        self._base = ["adb"] + (["-s", serial] if serial else [])
        self._active = {}                # slot -> tracking id (currently down)
        self._next_id = 1
        self._move_down = False
        self._warned = False             # print a device error at most once
        # Aimed presses are held for one tick and released at the start of
        # the next (see _aimed_press). Slots waiting for that release:
        self._pending_release = []
        # Per-tick batching (see _emit). None = not batching, send at once.
        self._batch = None
        self._persistent = persistent
        self._shell = None
        self._checked = False            # first write goes one-shot, to see errors

    # -- coordinate transform: landscape display px -> raw digitizer --
    def to_raw(self, lx, ly):
        fs = ly / LAND_H                 # short-axis fraction (from landscape height)
        fl = lx / LAND_W                 # long-axis fraction  (from landscape width)
        if self.flip_short:
            fs = 1.0 - fs
        if self.flip_long:
            fl = 1.0 - fl
        return int(round(fs * self.abs_x_max)), int(round(fl * self.abs_y_max))

    # -- raw event plumbing --
    #
    # WHY BATCHED, AND WHY A PERSISTENT SHELL. Every `_emit` used to be its own
    # `adb shell` subprocess: ~20-50ms of process spawn and USB round trip
    # EACH, and a tick that moves, attacks and supers made five of them (a tap
    # is a down AND an up). That is 100-250ms of blocking per tick on a loop
    # whose budget is 100ms -- the act stage alone could not keep up, the
    # same failure the README documents for `hold_ms`.
    #
    # Now `_execute` collects one tick's events and sends them as ONE line to
    # a long-lived `adb shell` (the trick AdbExecutor already uses), so a
    # tick costs one pipe write regardless of how many fingers move. The very
    # first write still goes one-shot with stderr captured, because that is
    # the only way to surface an SELinux/permission denial -- after that the
    # device has accepted our writes and the pipe can be fire-and-forget.
    def _emit(self, lines):
        if self._batch is not None:
            self._batch.extend(lines)
            return
        self._send(lines)

    def _send(self, lines):
        if not lines:
            return
        cmd = "; ".join(lines)
        if self._checked and self._persistent:
            if self._shell is None or self._shell.poll() is not None:
                self._open_shell()
            if self._shell is not None:
                try:
                    self._shell.stdin.write(cmd + "\n")
                    self._shell.stdin.flush()
                    return
                except Exception:
                    self._shell = None
        try:
            r = subprocess.run(self._base + ["shell", cmd],
                               capture_output=True, text=True, timeout=5)
            err = (r.stderr or "").strip()
            if err and not self._warned:
                print("[sendevent] device rejected the write:", err.splitlines()[0])
                print("            -> raw touch is likely blocked on this phone.")
                self._warned = True
            self._checked = self._checked or not err
        except Exception as e:
            if not self._warned:
                print("[sendevent] failed:", e)
                self._warned = True

    def _open_shell(self):
        try:
            self._shell = subprocess.Popen(
                self._base + ["shell"], stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
        except Exception:
            self._shell = None

    def _ev(self, typ, code, val):
        return f"sendevent {self.device} {typ} {code} {val}"

    def _down(self, slot, lx, ly):
        rx, ry = self.to_raw(lx, ly)
        lines = [self._ev(EV_ABS, ABS_MT_SLOT, slot),
                 self._ev(EV_ABS, ABS_MT_TRACKING_ID, self._next_id)]
        if not self._active:             # first finger touching -> BTN_TOUCH down
            lines.append(self._ev(EV_KEY, BTN_TOUCH, 1))
        lines += [self._ev(EV_ABS, ABS_MT_POSITION_X, rx),
                  self._ev(EV_ABS, ABS_MT_POSITION_Y, ry),
                  self._ev(EV_SYN, SYN_REPORT, 0)]
        self._active[slot] = self._next_id
        self._next_id += 1
        self._emit(lines)

    def _move(self, slot, lx, ly):
        rx, ry = self.to_raw(lx, ly)
        self._emit([self._ev(EV_ABS, ABS_MT_SLOT, slot),
                    self._ev(EV_ABS, ABS_MT_POSITION_X, rx),
                    self._ev(EV_ABS, ABS_MT_POSITION_Y, ry),
                    self._ev(EV_SYN, SYN_REPORT, 0)])

    def _up(self, slot):
        if slot not in self._active:
            return
        del self._active[slot]
        lines = [self._ev(EV_ABS, ABS_MT_SLOT, slot),
                 self._ev(EV_ABS, ABS_MT_TRACKING_ID, -1)]
        if not self._active:             # last finger lifted -> BTN_TOUCH up
            lines.append(self._ev(EV_KEY, BTN_TOUCH, 0))
        lines.append(self._ev(EV_SYN, SYN_REPORT, 0))
        self._emit(lines)

    def _tap(self, slot, lx, ly):
        self._down(slot, lx, ly)
        self._up(slot)

    # -- intent -> touches --
    def _execute(self, intent: Intent) -> None:
        c = self.controls
        self._batch = []
        try:
            # Release last tick's aimed presses FIRST: lifting the finger is
            # what fires an aimed shot, and the slot must be free before this
            # tick can press it again.
            for slot in self._pending_release:
                self._up(slot)
            self._pending_release = []

            if intent.move != (0.0, 0.0):
                cx, cy = c.move_center
                tx = int(cx + intent.move[0] * c.move_radius)
                ty = int(cy + intent.move[1] * c.move_radius)
                if not self._move_down:
                    self._down(self.MOVE_SLOT, cx, cy)   # press at joystick center
                    self._move_down = True
                self._move(self.MOVE_SLOT, tx, ty)        # hold + drag to direction
            elif self._move_down:
                self._up(self.MOVE_SLOT)
                self._move_down = False

            # A plain tap is the game's auto-aim; an aim vector is a drag.
            # Ignoring the aim here used to turn every manual-aim brawler (and
            # every retargeted shot from rl/combat.py) back into auto-aim on
            # this backend, silently.
            if intent.fire_attack:
                if getattr(intent, "aim", None) is None:
                    self._tap(self.ATTACK_SLOT, *c.attack_btn)
                else:
                    self._aimed_press(self.ATTACK_SLOT, c.attack_btn,
                                      intent.aim, getattr(intent, "aim_frac", 1.0))
            if intent.fire_super:
                if getattr(intent, "aim", None) is None:
                    self._tap(self.SUPER_SLOT, *c.super_btn)
                else:
                    self._aimed_press(self.SUPER_SLOT, c.super_btn,
                                      intent.aim, getattr(intent, "aim_frac", 1.0))
        finally:
            lines, self._batch = self._batch, None
            self._send(lines)

    def _aimed_press(self, slot, button, aim, frac: float) -> None:
        """Press a fire button, drag out along `aim`, release NEXT tick.

        The in-game gesture is press, drag to swing the aim indicator, lift to
        fire. Sent as one burst, DOWN/MOVE/UP land within a millisecond of each
        other and the game can read that as a tap -- an auto-aimed shot, the
        silent mode switch `Controls.aim_min_radius` exists to prevent. Holding
        the drag for one tick (the release is queued for the start of the next
        `_execute`) gives the game whole frames with the aim indicator out,
        with no sleeping anywhere in the loop.

        Two MOVEs rather than one: a single jump straight to the end point is
        the least drag-like thing a finger can do.
        """
        c = self.controls
        bx, by = int(button[0]), int(button[1])
        r = max(c.aim_min_radius, int(c.aim_radius * max(c.aim_min_frac,
                                                         min(1.0, float(frac or 1.0)))))
        tx = max(1, min(LAND_W - 2, int(bx + aim[0] * r)))
        ty = max(1, min(LAND_H - 2, int(by + aim[1] * r)))
        if slot in self._active:          # still held from a previous aim
            self._up(slot)
        self._down(slot, bx, by)
        self._move(slot, (bx + tx) // 2, (by + ty) // 2)
        self._move(slot, tx, ty)
        self._pending_release.append(slot)

    # -- one-time orientation calibration --
    def calibrate(self):
        """Figure out which flip is right. Taps the attack button under BOTH
        landscape orientations, ~1.5s apart. Watch the phone: whichever tap lands
        ON the attack button (bottom-right) is your orientation. Tell me "A" or "B"
        and I'll set the default (or set flip_short/flip_long yourself).
        """
        c = self.controls
        for label, fs, fl in (("A", False, True), ("B", True, False)):
            self.flip_short, self.flip_long = fs, fl
            rx, ry = self.to_raw(*c.attack_btn)
            print(f"  orientation {label}: flip_short={fs} flip_long={fl} "
                  f"-> tapping attack at raw ({rx},{ry})")
            self._tap(self.ATTACK_SLOT, *c.attack_btn)
            time.sleep(1.5)
        print("Which one hit the attack button? Set flip_short/flip_long to that row.")

    def close(self):
        try:
            for slot in self._pending_release:
                self._up(slot)
            self._pending_release = []
            if self._move_down:
                self._up(self.MOVE_SLOT)
                self._move_down = False
        except Exception:
            pass
        if self._shell is not None:
            try:
                self._shell.stdin.close()
                self._shell.terminate()
            except Exception:
                pass
            self._shell = None


def make_sendevent_env(serial: Optional[str] = None, controls: Optional[Controls] = None,
                       reward_config=None, tick_seconds: float = 0.1,
                       flip_short: bool = False, flip_long: bool = True,
                       device: Optional[str] = None, **kwargs):
    """Live env: raw-multitouch control (smooth) + adb screencap capture."""
    from .env import BrawlStarsEnv
    from perception.liveLoop import AdbScreencapSource
    executor = SendeventExecutor(serial=serial, controls=controls,
                                 flip_short=flip_short, flip_long=flip_long,
                                 device=device)
    return BrawlStarsEnv(source_factory=lambda: AdbScreencapSource(serial),
                         executor=executor, reward_config=reward_config,
                         tick_seconds=tick_seconds, **kwargs)
