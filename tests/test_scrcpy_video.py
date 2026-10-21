"""rl/scrcpy_video.py against a fake server streaming real H.264 (no phone)."""

import pathlib
import socket
import sys
import threading
import time
import types

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

av = pytest.importorskip("av")
from rl import scrcpy_video as sv  # noqa: E402

W, H, N = 640, 288, 40


def _h264_frames():
    """N frames, frame i a flat grey of level 5*i, as one Annex-B stream per frame."""
    enc = av.CodecContext.create("libx264", "w")
    enc.width, enc.height, enc.pix_fmt, enc.framerate = W, H, "yuv420p", 30
    enc.options = {"tune": "zerolatency", "preset": "ultrafast"}
    chunks = []
    for i in range(N):
        img = np.full((H, W, 3), 5 * i, np.uint8)
        chunks.append(b"".join(bytes(p) for p in enc.encode(
            av.VideoFrame.from_ndarray(img, format="bgr24"))))
    return chunks


class _FakeServer:
    """The adb-forwarded port with raw_stream=true: H.264 bytes, no headers."""

    def __init__(self, chunks, interval=0.01):
        self.srv = socket.socket()
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(1)
        self.port = self.srv.getsockname()[1]
        self.chunks, self.interval = chunks, interval
        self.done = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        c, _ = self.srv.accept()
        for ch in self.chunks:
            c.sendall(ch)
            time.sleep(self.interval)
        self.done.set()
        time.sleep(1.0)
        c.close()


def test_decodes_the_stream_and_hands_out_the_newest_frame(monkeypatch):
    fake = _FakeServer(_h264_frames())
    monkeypatch.setattr(sv, "_free_port", lambda: fake.port)
    monkeypatch.setattr(sv.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(sv.ScrcpyVideoSource, "_adb",
                        lambda self, *a, **k: types.SimpleNamespace(returncode=0, stdout="", stderr=""))

    class _Proc:
        stdout = iter(())

        def poll(self):
            return None

        def terminate(self):
            pass

    monkeypatch.setattr(sv.subprocess, "Popen", lambda *a, **k: _Proc())
    src = sv.ScrcpyVideoSource(server=(pathlib.Path("/x"), "3.3.1"), log=lambda *a: None)
    assert fake.done.wait(5)
    time.sleep(0.3)
    f = src.grab()
    src.close()
    assert f is not None and f.shape == (H, W, 3)
    # Every frame but the last is emitted (the parser waits for the next start
    # code), and grab() returns the newest decoded one, not the first.
    assert src.frames >= N - 2
    assert abs(float(f.mean()) - 5 * (N - 2)) <= 6
    assert src.decode_errors == 0
