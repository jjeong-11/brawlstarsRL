"""rl/scrcpy_control.py against a fake scrcpy server (no phone, no adb)."""

import pathlib
import socket
import struct
import sys
import threading
import time
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rl import scrcpy_control as sc  # noqa: E402
from rl.actions import Controls, Intent  # noqa: E402


def test_touch_event_layout_matches_server_reader():
    # ControlMessageReader.parseInjectTouchEvent: u8 action, i64 pointer id,
    # i32 x, i32 y, u16 w, u16 h, u16 pressure, i32 action_button, i32 buttons
    msg = sc.encode_touch(sc.ACTION_MOVE, 11, 300, 850, (2424, 1080), 1.0)
    assert len(msg) == 32
    t, a, pid, x, y, w, h, p, ab, b = struct.unpack(">BBqiiHHHII", msg)
    assert (t, a, pid, x, y, w, h, p, ab, b) == (2, 2, 11, 300, 850, 2424, 1080, 0xFFFF, 0, 0)
    assert struct.unpack(">BBqiiHHHII", sc.encode_touch(sc.ACTION_UP, 11, 1, 2, (8, 8), 0.0))[7] == 0


class _FakeServer:
    """Listens like the adb-forwarded port: dummy byte, then records bytes."""

    def __init__(self):
        self.srv = socket.socket()
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(1)
        self.port = self.srv.getsockname()[1]
        self.data = bytearray()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        c, _ = self.srv.accept()
        c.sendall(b"\x00")
        c.sendall(b"Pixel 10".ljust(64, b"\x00"))      # device meta: must be drained
        while True:
            chunk = c.recv(4096)
            if not chunk:
                break
            self.data += chunk


def _events(raw):
    return [struct.unpack(">BBqiiHHHII", bytes(raw[i:i + 32])) for i in range(0, len(raw) - 31, 32)]


def test_executor_holds_the_stick_and_taps_with_separate_fingers(monkeypatch):
    fake = _FakeServer()
    monkeypatch.setattr(sc, "_free_port", lambda: fake.port)
    monkeypatch.setattr(sc.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(sc.ScrcpyControl, "_adb",
                        lambda self, *a, **k: types.SimpleNamespace(returncode=0, stdout="", stderr=""))

    class _Proc:
        stdout = iter(())

        def poll(self):
            return None

        def terminate(self):
            pass

    monkeypatch.setattr(sc.subprocess, "Popen", lambda *a, **k: _Proc())
    monkeypatch.setattr(sc, "_detect_size", lambda serial: (2424, 1080))
    monkeypatch.setattr(sc, "find_server", lambda: (pathlib.Path("/x/scrcpy-server"), "3.3.1"))

    ctl = Controls(move_center=(300, 850), attack_btn=(2186, 874), super_btn=(1776, 806),
                   move_radius=180)
    ex = sc.ScrcpyExecutor(None, ctl, log=lambda *a: None)
    right = Intent("E", (1.0, 0.0), False, False, None)
    down_ = Intent("S", (0.0, 1.0), True, False, None)
    stop = Intent("-", (0.0, 0.0), False, False, None)
    ex._execute(right)
    ex._execute(right)
    ex._execute(down_)
    time.sleep(0.15)                 # let the attack tap's UP timer fire
    ex._execute(stop)
    ex.close()
    time.sleep(0.1)

    ev = _events(fake.data)
    mv = [(e[1], e[3], e[4]) for e in ev if e[2] == ex.MOVE_ID]
    # ONE down at the centre, MOVEs while held, ONE up: no lift between ticks.
    assert mv[0] == (sc.ACTION_DOWN, 300, 850)
    assert [a for a, _, _ in mv].count(sc.ACTION_DOWN) == 1
    assert [a for a, _, _ in mv].count(sc.ACTION_UP) == 1 and mv[-1][0] == sc.ACTION_UP
    assert (sc.ACTION_MOVE, 480, 850) in mv and (sc.ACTION_MOVE, 300, 1030) in mv
    at = [(e[1], e[3], e[4]) for e in ev if e[2] == ex.ATTACK_ID]
    assert at == [(sc.ACTION_DOWN, 2186, 874), (sc.ACTION_UP, 2186, 874)]
    assert all(e[5:7] == (2424, 1080) for e in ev)
