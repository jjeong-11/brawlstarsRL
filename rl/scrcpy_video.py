"""
rl/scrcpy_video.py
==================

Low-latency frames from the phone over scrcpy's H.264 stream.

WHY
---
`adb exec-out screencap` costs 150-400 ms per frame (README, "Finding out
why the loop is slow"). It is the single largest delay between the game
changing and the bot reacting -- larger than all of perception. scrcpy's
server encodes the screen with the phone's hardware H.264 encoder and
streams it; decoding on the computer takes a few ms per frame, so a frame is
typically tens of ms old when perception sees it. PylaAI captures the same
way (scrcpy + PyAV).

HOW
---
The server runs with `raw_stream=true`: no device name, codec header or
per-packet header -- just an Annex-B H.264 byte stream. That is the one
framing that has not changed across scrcpy 2.x-3.x releases (the header
format has), and PyAV's parser splits it into packets by itself. A decoder
thread keeps only the newest frame; `grab()` hands it out instantly.

The cost of skipping the per-packet header: the parser emits a frame only
when the next one starts, so frames are one frame-interval late (33 ms at
30 fps). Raise `max_fps` to shrink that.

    pip install av              # PyAV
    brew install scrcpy         # the server jar (see rl/scrcpy_control.py)

    python -m rl.scrcpy_video --serial <SERIAL> --seconds 10   # measures fps and frame age
"""

from __future__ import annotations

import random
import shutil
import socket
import subprocess
import threading
import time
from typing import Optional

from .scrcpy_control import _DEVICE_JAR, _free_port, find_server


class ScrcpyVideoSource:
    """A frame source (`grab()` -> newest BGR frame) backed by scrcpy video."""

    live = True   # a None grab is transient: the caller skips and keeps going

    def __init__(self, serial: Optional[str] = None, max_fps: int = 30,
                 bit_rate: int = 8_000_000, max_size: int = 0,
                 max_age: float = 1.0, connect_timeout: float = 10.0,
                 server=None, log=print):
        try:
            import av  # noqa: F401
        except ImportError as e:
            raise RuntimeError("PyAV is needed for scrcpy capture: pip install av") from e
        self.serial = serial
        self.log = log
        self.max_age = max_age
        self.jar, self.version = server or find_server()
        self.scid = random.randrange(1, 0x7FFFFFFF)
        self.port = _free_port()
        self._sock: Optional[socket.socket] = None
        self._proc: Optional[subprocess.Popen] = None
        self._frame = None
        self._t = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.frames = 0
        self.decode_errors = 0
        self._start(max_fps, bit_rate, max_size, connect_timeout)

    # ------------------------------------------------------------------ #
    def _adb(self, *args, timeout=20):
        cmd = ["adb"] + (["-s", self.serial] if self.serial else []) + list(args)
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    def _start(self, max_fps, bit_rate, max_size, timeout) -> None:
        if shutil.which("adb") is None:
            raise RuntimeError("`adb` not found on PATH")
        r = self._adb("push", str(self.jar), _DEVICE_JAR, timeout=60)
        if r.returncode != 0:
            raise RuntimeError(f"adb push of the scrcpy server failed: {r.stderr.strip()}")
        r = self._adb("forward", f"tcp:{self.port}", f"localabstract:scrcpy_{self.scid:08x}")
        if r.returncode != 0:
            raise RuntimeError(f"adb forward failed: {r.stderr.strip()}")
        args = [f"CLASSPATH={_DEVICE_JAR}", "app_process", "/",
                "com.genymobile.scrcpy.Server", self.version,
                f"scid={self.scid:08x}", "log_level=info", "tunnel_forward=true",
                "video=true", "audio=false", "control=false", "cleanup=true",
                "raw_stream=true", f"max_fps={int(max_fps)}",
                f"video_bit_rate={int(bit_rate)}", f"max_size={int(max_size)}"]
        cmd = ["adb"] + (["-s", self.serial] if self.serial else []) + ["shell"] + args
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True)
        self._server_log = []
        threading.Thread(target=self._pump_log, daemon=True).start()

        # raw_stream sends no dummy byte, so "connected" means "data arrived":
        # until the server listens, adb accepts and immediately closes.
        deadline = time.time() + timeout
        first = b""
        while time.time() < deadline and not first:
            if self._proc.poll() is not None:
                break
            try:
                s = socket.create_connection(("127.0.0.1", self.port), timeout=2.0)
                s.settimeout(3.0)
                first = s.recv(1 << 16)
                if first:
                    s.settimeout(None)
                    self._sock = s
                    break
                s.close()
            except OSError:
                pass
            time.sleep(0.2)
        if self._sock is None:
            self.close()
            raise RuntimeError("no video from the scrcpy server: "
                               + (" | ".join(self._server_log[-6:]) or "timed out"))
        self.log(f"scrcpy {self.version}: video stream connected (port {self.port})")
        threading.Thread(target=self._decode_loop, args=(first,), daemon=True).start()

    def _pump_log(self) -> None:
        try:
            for line in self._proc.stdout:
                line = line.rstrip()
                if line:
                    self._server_log.append(line)
                    del self._server_log[:-50]
        except Exception:
            pass

    def _decode_loop(self, first: bytes) -> None:
        import av
        codec = av.CodecContext.create("h264", "r")
        try:
            codec.flags2 |= getattr(av.codec.context.Flags2, "fast", 0)
        except Exception:
            pass
        data = first
        while not self._stop.is_set():
            if data:
                try:
                    for pkt in codec.parse(data):
                        for fr in codec.decode(pkt):
                            img = fr.to_ndarray(format="bgr24")
                            with self._lock:
                                self._frame, self._t = img, time.time()
                                self.frames += 1
                except Exception:
                    self.decode_errors += 1      # corrupt packet: skip it
            try:
                data = self._sock.recv(1 << 17) if self._sock is not None else b""
            except OSError:
                data = b""
            if not data:
                break
        self.log("scrcpy video stream ended")

    # ------------------------------------------------------------------ #
    def grab(self):
        """The newest decoded frame (BGR, device resolution), or None."""
        with self._lock:
            frame, t = self._frame, self._t
        if frame is None or time.time() - t > self.max_age:
            return None
        return frame

    @property
    def frame_age(self) -> Optional[float]:
        with self._lock:
            return None if self._frame is None else time.time() - self._t

    def pause(self) -> None:      # menus tap over a separate channel; nothing to free
        pass

    def resume(self) -> None:
        pass

    def close(self) -> None:
        self._stop.set()
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


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    ap = argparse.ArgumentParser(description="scrcpy capture: measure fps and frame age")
    ap.add_argument("--serial", default=None)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--max-fps", type=int, default=30)
    ap.add_argument("--save", default=None, help="write the last frame to this .png")
    a = ap.parse_args()
    src = ScrcpyVideoSource(a.serial, max_fps=a.max_fps)
    try:
        t_end = time.time() + a.seconds
        ages = []
        while time.time() < t_end:
            time.sleep(0.1)
            age = src.frame_age
            if age is not None:
                ages.append(age)
        f = src.grab()
        ages.sort()
        if ages:
            print(f"{src.frames / a.seconds:.1f} fps decoded, frame age median "
                  f"{1000 * ages[len(ages) // 2]:.0f} ms (age since decode, "
                  f"not glass-to-glass), size {None if f is None else f.shape}")
        else:
            print("no frames decoded")
        if a.save and f is not None:
            import cv2
            cv2.imwrite(a.save, f)
            print("saved", a.save)
    finally:
        src.close()
