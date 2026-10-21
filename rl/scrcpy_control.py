"""
rl/scrcpy_control.py
====================

SMOOTH, HELD touch input through the official scrcpy server.

WHY
---
`adb shell input swipe` is the only touch path adb offers on a recent Pixel
(raw `sendevent` is blocked), and every swipe lifts the finger when it ends:
the brawler walks for `hold_ms`, stops, waits for the next tick, and walks
again. Separate `input motionevent` calls are not linked into one gesture on
most devices either. So movement on the adb backend is bursty by
construction, and the planner's stuck detection reads the stop-start as
"barely moving".

scrcpy's server, started over adb, keeps a control socket open and injects
touch events into the input system directly. A finger put down stays down
until we lift it, MOVE redirects it, and several fingers can be down at once
(movement held while attack and super are tapped).

WHAT YOU NEED
-------------
The scrcpy desktop app, which ships the server jar:

    brew install scrcpy            # macOS
    sudo apt install scrcpy        # Debian/Ubuntu (2.1 or newer)

The jar is found next to the `scrcpy` binary (or set SCRCPY_SERVER_PATH), and
its version is read from `scrcpy --version` (or set SCRCPY_VERSION) -- the
server refuses a client that announces any other version.

Only the control channel is opened (video=false audio=false): nothing is
captured or mirrored, and frames still come from screencap.

PROTOCOL (scrcpy >= 2.1, checked against the server source)
-----------------------------------------------------------
  * push the jar to /data/local/tmp, `adb forward tcp:<port>
    localabstract:scrcpy_<scid>`, run it with app_process
  * connect to the forwarded port; the server sends one dummy byte once it is
    really listening (until then adb accepts and immediately closes)
  * INJECT_TOUCH_EVENT, 32 bytes, big-endian:
        u8 type=2 | u8 action | u64 pointer_id | i32 x | i32 y |
        u16 screen_w | u16 screen_h | u16 pressure | u32 action_button |
        u32 buttons
  * anything the server sends back (device name, clipboard) is read and
    discarded so its socket never backs up.

    python -m rl.scrcpy_control --serial <SERIAL> --selftest
"""

from __future__ import annotations

import os
import pathlib
import random
import re
import shutil
import socket
import struct
import subprocess
import threading
import time
from typing import List, Optional, Tuple

from .actions import ActionExecutor, Controls, Intent

# MotionEvent actions
ACTION_DOWN, ACTION_UP, ACTION_MOVE = 0, 1, 2
TYPE_INJECT_TOUCH_EVENT = 2
_DEVICE_JAR = "/data/local/tmp/scrcpy-server-brawlbot.jar"


# --------------------------------------------------------------------------- #
def find_server() -> Tuple[pathlib.Path, str]:
    """(path to scrcpy-server, its version) or raise with what to install."""
    version = os.environ.get("SCRCPY_VERSION")
    exe = shutil.which("scrcpy")
    if version is None and exe:
        try:
            out = subprocess.run([exe, "--version"], capture_output=True, text=True,
                                 timeout=10).stdout
            m = re.search(r"scrcpy\s+v?(\d+\.\d+(?:\.\d+)?)", out)
            version = m.group(1) if m else None
        except Exception:
            version = None

    cands: List[pathlib.Path] = []
    if os.environ.get("SCRCPY_SERVER_PATH"):
        cands.append(pathlib.Path(os.environ["SCRCPY_SERVER_PATH"]))
    if exe:
        real = pathlib.Path(os.path.realpath(exe))
        cands += [real.parent.parent / "share" / "scrcpy" / "scrcpy-server",
                  real.parent / "scrcpy-server"]
    cands += [pathlib.Path(p) for p in ("/opt/homebrew/share/scrcpy/scrcpy-server",
                                        "/usr/local/share/scrcpy/scrcpy-server",
                                        "/usr/share/scrcpy/scrcpy-server")]
    jar = next((p for p in cands if p.is_file()), None)
    if jar is None or version is None:
        raise RuntimeError(
            "scrcpy not found. Install it (macOS: `brew install scrcpy`), or set "
            "SCRCPY_SERVER_PATH and SCRCPY_VERSION. Looked for the server at: "
            + ", ".join(str(p) for p in cands))
    parts = [int(x) for x in version.split(".")[:2]]
    if parts < [2, 1]:
        raise RuntimeError(f"scrcpy {version} is too old: this needs 2.1 or newer "
                           f"(control-only sessions and the 32-byte touch event)")
    return jar, version


def encode_touch(action: int, pointer_id: int, x: int, y: int,
                 screen: Tuple[int, int], pressure: float) -> bytes:
    """One INJECT_TOUCH_EVENT message (32 bytes)."""
    p = max(0, min(0xFFFF, int(round(pressure * 0xFFFF))))
    return struct.pack(">BBqiiHHHII", TYPE_INJECT_TOUCH_EVENT, action, pointer_id,
                       int(x), int(y), int(screen[0]), int(screen[1]), p, 0, 0)


class ScrcpyControl:
    """The control socket of a scrcpy server session: inject touches."""

    def __init__(self, serial: Optional[str] = None, screen_size=(2424, 1080),
                 server: Optional[Tuple[pathlib.Path, str]] = None,
                 connect_timeout: float = 8.0, log=print):
        self.serial = serial
        # Older servers map positions through the size the client believes the
        # screen has and drop events whose size does not match their own
        # (computed with dimensions rounded down to a multiple of 8). Newer
        # ones inject raw coordinates when no video is running. Sending the
        # real landscape size, rounded the same way, satisfies both.
        w, h = int(screen_size[0]), int(screen_size[1])
        self.screen = (w & ~7, h & ~7)
        self._sx = self.screen[0] / max(1, w)
        self._sy = self.screen[1] / max(1, h)
        self.log = log
        self.jar, self.version = server or find_server()
        self.scid = random.randrange(1, 0x7FFFFFFF)
        self.port = _free_port()
        self._sock: Optional[socket.socket] = None
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._down = set()
        self._server_log: List[str] = []
        self._start(connect_timeout)

    # -- adb plumbing ---------------------------------------------------- #
    def _adb(self, *args, timeout=20):
        cmd = ["adb"] + (["-s", self.serial] if self.serial else []) + list(args)
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    def _start(self, timeout: float) -> None:
        if shutil.which("adb") is None:
            raise RuntimeError("`adb` not found on PATH")
        r = self._adb("push", str(self.jar), _DEVICE_JAR, timeout=60)
        if r.returncode != 0:
            raise RuntimeError(f"adb push of the scrcpy server failed: {r.stderr.strip()}")
        name = f"scrcpy_{self.scid:08x}"
        r = self._adb("forward", f"tcp:{self.port}", f"localabstract:{name}")
        if r.returncode != 0:
            raise RuntimeError(f"adb forward failed: {r.stderr.strip()}")
        args = [f"CLASSPATH={_DEVICE_JAR}", "app_process", "/",
                "com.genymobile.scrcpy.Server", self.version,
                f"scid={self.scid:08x}", "log_level=info", "tunnel_forward=true",
                "video=false", "audio=false", "control=true", "cleanup=true"]
        cmd = ["adb"] + (["-s", self.serial] if self.serial else []) + ["shell"] + args
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True)
        threading.Thread(target=self._pump_server_log, daemon=True).start()

        deadline = time.time() + timeout
        last_err = None
        while time.time() < deadline:
            if self._proc.poll() is not None:
                break
            try:
                s = socket.create_connection(("127.0.0.1", self.port), timeout=2.0)
                s.settimeout(2.0)
                b = s.recv(1)                 # the dummy byte: server really there
                if b:
                    s.settimeout(None)
                    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    self._sock = s
                    threading.Thread(target=self._drain, daemon=True).start()
                    self.log(f"scrcpy {self.version}: control connected "
                             f"(port {self.port}, screen {self.screen[0]}x{self.screen[1]})")
                    return
                s.close()
            except OSError as e:
                last_err = e
            time.sleep(0.2)
        self.close()
        tail = " | ".join(self._server_log[-6:]) or str(last_err)
        raise RuntimeError(f"could not connect to the scrcpy server: {tail}")

    def _pump_server_log(self) -> None:
        proc = self._proc
        try:
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    self._server_log.append(line)
                    del self._server_log[:-50]
                    if "ERROR" in line or "Exception" in line:
                        self.log(f"scrcpy server: {line}")
        except Exception:
            pass

    def _drain(self) -> None:
        """Read and discard device messages (device name, clipboard...)."""
        s = self._sock
        try:
            while s is not None and s.recv(4096):
                pass
        except OSError:
            pass

    # -- touches --------------------------------------------------------- #
    def touch(self, action: int, pointer_id: int, x: float, y: float) -> None:
        if self._sock is None:
            return
        msg = encode_touch(action, pointer_id, int(round(x * self._sx)),
                           int(round(y * self._sy)), self.screen,
                           0.0 if action == ACTION_UP else 1.0)
        with self._lock:
            try:
                self._sock.sendall(msg)
            except OSError as e:
                self.log(f"scrcpy control socket lost: {e}")
                self._sock = None
                return
        if action == ACTION_DOWN:
            self._down.add(pointer_id)
        elif action == ACTION_UP:
            self._down.discard(pointer_id)

    def down(self, pid, x, y):
        self.touch(ACTION_DOWN, pid, x, y)

    def move(self, pid, x, y):
        self.touch(ACTION_MOVE, pid, x, y)

    def up(self, pid, x, y):
        self.touch(ACTION_UP, pid, x, y)

    @property
    def alive(self) -> bool:
        return self._sock is not None and self._proc is not None and self._proc.poll() is None

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass
        try:
            self._adb("forward", "--remove", f"tcp:{self.port}", timeout=5)
        except Exception:
            pass


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# --------------------------------------------------------------------------- #
class ScrcpyExecutor(ActionExecutor):
    """Held movement + separate fire fingers, over scrcpy.

    Movement: one finger goes down at the stick's centre and STAYS down; each
    tick only MOVEs it toward the new direction, and it lifts when the brawler
    should stop. Attack and super are their own fingers, so firing never
    interrupts movement. Taps and aimed drags run on a short timer thread
    (a press needs a few tens of ms between DOWN and UP to register), so the
    tick loop is never blocked.
    """

    MOVE_ID, ATTACK_ID, SUPER_ID, MENU_ID = 11, 12, 13, 14
    PRESS_S = 0.045         # DOWN -> UP for a tap
    AIM_S = 0.035           # DOWN -> MOVE and MOVE -> UP for an aimed shot

    def __init__(self, serial: Optional[str] = None, controls: Optional[Controls] = None,
                 log=print):
        self.serial = serial
        self.screen_size = _detect_size(serial)
        self.controls = controls or Controls.from_screen(*self.screen_size)
        self.ctl = ScrcpyControl(serial, self.screen_size, log=log)
        self._move_down = False
        self._move_pos = None
        self._busy = {self.ATTACK_ID: 0.0, self.SUPER_ID: 0.0}

    # -- movement -------------------------------------------------------- #
    def _execute(self, intent: Intent) -> None:
        c = self.controls
        if intent.move != (0.0, 0.0):
            cx, cy = c.move_center
            tx = cx + intent.move[0] * c.move_radius
            ty = cy + intent.move[1] * c.move_radius
            if not self._move_down:
                # Down at the CENTRE, then out: pressing down at the offset
                # makes a floating stick re-centre there (no deflection).
                self.ctl.down(self.MOVE_ID, cx, cy)
                self._move_down = True
            self.ctl.move(self.MOVE_ID, tx, ty)
            self._move_pos = (tx, ty)
        else:
            self._release_move()

        if intent.fire_attack:
            self._press(self.ATTACK_ID, c.attack_btn, intent.aim, intent.aim_frac)
        if intent.fire_super:
            self._press(self.SUPER_ID, c.super_btn, intent.aim, intent.aim_frac)

    def _release_move(self) -> None:
        if self._move_down:
            x, y = self._move_pos or self.controls.move_center
            self.ctl.up(self.MOVE_ID, x, y)
            self._move_down = False
            self._move_pos = None

    # -- fire buttons ---------------------------------------------------- #
    def _press(self, pid, button, aim=None, frac: float = 1.0) -> None:
        now = time.time()
        if self._busy.get(pid, 0.0) > now:
            return                      # previous press on this finger not finished
        c = self.controls
        bx, by = float(button[0]), float(button[1])
        if aim is None:
            self._busy[pid] = now + self.PRESS_S + 0.01
            self.ctl.down(pid, bx, by)
            threading.Timer(self.PRESS_S, self.ctl.up, (pid, bx, by)).start()
            return
        r = max(c.aim_min_radius, c.aim_radius * max(c.aim_min_frac, min(1.0, frac)))
        w, h = self.screen_size
        tx = max(1.0, min(w - 2.0, bx + aim[0] * r))
        ty = max(1.0, min(h - 2.0, by + aim[1] * r))
        self._busy[pid] = now + 2 * self.AIM_S + 0.01
        self.ctl.down(pid, bx, by)
        threading.Timer(self.AIM_S, self.ctl.move, (pid, tx, ty)).start()
        threading.Timer(2 * self.AIM_S, self.ctl.up, (pid, tx, ty)).start()

    # -- the rest of the executor interface ------------------------------ #
    def tap_norm(self, fx: float, fy: float) -> None:
        w, h = self.screen_size
        x, y = fx * w, fy * h
        self.ctl.down(self.MENU_ID, x, y)
        time.sleep(self.PRESS_S)
        self.ctl.up(self.MENU_ID, x, y)

    def calibrate(self) -> None:
        """Tap attack and super, then hold the stick right for a second."""
        c = self.controls
        print(f"screen {self.screen_size} | controls {c}")
        for name, pid, pt in (("attack", self.ATTACK_ID, c.attack_btn),
                              ("super", self.SUPER_ID, c.super_btn)):
            print("tap", name, pt)
            self.ctl.down(pid, *pt)
            time.sleep(self.PRESS_S)
            self.ctl.up(pid, *pt)
            time.sleep(0.4)
        print("hold move right for 1 s")
        cx, cy = c.move_center
        self.ctl.down(self.MOVE_ID, cx, cy)
        for i in range(1, 11):
            self.ctl.move(self.MOVE_ID, cx + c.move_radius * i / 10, cy)
            time.sleep(0.02)
        time.sleep(0.8)
        self.ctl.up(self.MOVE_ID, cx + c.move_radius, cy)

    def close(self) -> None:
        try:
            self._release_move()
            for pid in list(self.ctl._down):
                self.ctl.up(pid, *self.controls.attack_btn)
        except Exception:
            pass
        self.ctl.close()


def _detect_size(serial: Optional[str]) -> Tuple[int, int]:
    """Landscape (w, h) from `wm size` (override size wins if set)."""
    try:
        cmd = ["adb"] + (["-s", serial] if serial else []) + ["shell", "wm", "size"]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout
        line = [l for l in out.splitlines() if "size:" in l][-1]
        a, b = (int(v) for v in line.split(":")[-1].strip().lower().split("x"))
        return (max(a, b), min(a, b))
    except Exception:
        return (2424, 1080)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    ap = argparse.ArgumentParser(description="scrcpy touch control self-test")
    ap.add_argument("--serial", default=None)
    ap.add_argument("--controls", default="controls.json")
    ap.add_argument("--selftest", action="store_true",
                    help="tap attack and super, then hold the joystick right for 1 s")
    ap.add_argument("--walk", type=float, default=0.0, metavar="SECONDS",
                    help="also walk a square, SECONDS per side, with the stick held")
    a = ap.parse_args()
    ctrl = Controls.from_json(a.controls) if pathlib.Path(a.controls).exists() else None
    ex = ScrcpyExecutor(a.serial, ctrl)
    try:
        if a.selftest:
            ex.calibrate()
        if a.walk > 0:
            c = ex.controls
            cx, cy = c.move_center
            ex.ctl.down(ex.MOVE_ID, cx, cy)
            for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
                t_end = time.time() + a.walk
                while time.time() < t_end:
                    ex.ctl.move(ex.MOVE_ID, cx + dx * c.move_radius, cy + dy * c.move_radius)
                    time.sleep(0.1)
            ex.ctl.up(ex.MOVE_ID, cx, cy)
        print("done")
    finally:
        ex.close()
