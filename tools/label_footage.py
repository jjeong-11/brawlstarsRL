"""
tools/label_footage.py
======================

Action labels from gameplay footage: an inverse-dynamics model that needs no
training data, because Brawl Stars hands us the hard part.

    python tools/label_footage.py match.mp4 --out labels.jsonl
    python tools/label_footage.py match.mp4 --out labels.jsonl --step 2   # every 2nd frame

WHY IT WORKS
------------
VPT (Baker et al. 2022) labelled 70k hours of video with an IDM trained on
2k hours of labelled play. Here most of that is free:

  * MOVEMENT. The camera follows the player. World content scrolls on
    screen by the camera shift `c` (rl/camera_tracker.py, phase correlation),
    and the player's sprite moves on screen by `d` (normally ~0; non-zero
    where the camera clamps at a map edge). The player's motion through the
    world is therefore `d - c` -- exact up to tracking error, at any camera
    behaviour. Its direction is the joystick direction; its length says
    whether the stick was pushed at all.
  * ATTACK. The ammo bar loses a segment (when it reads on both frames).
  * SUPER. The super charge goes from full to near empty.

What it does NOT recover yet: the aim direction of a shot, and anything
off-screen. Recording with Android's "Show taps" developer option draws every
touch, joystick and aim drags included -- reading those circles is the next
step once a sample recording exists (docs/PLAN_pro_imitation.md, section 5).

OUTPUT
------
One JSON object per labelled frame:
    frame, t, rect, cam (content shift, px/frame), player (screen px or null),
    motion (player world motion, px/s), speed_tps (tiles/s), moving (bool),
    move (unit vector or null), heading16 (0..15, 0 = right, clockwise; or
    null), attack (bool), super (bool), ok (camera estimate trusted)
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

from rl.camera_tracker import CameraTracker  # noqa: E402


@dataclass
class LabelConfig:
    fps: float = 30.0
    tiles_across: float = 17.0       # across the play rect (CombatConfig default)
    # Moving if the smoothed world speed is above this many tiles/s. Brawler
    # speeds are ~2.0-2.9 tiles/s; tracking jitter on a still player is well
    # under 0.3.
    move_threshold_tps: float = 0.7
    smooth_frames: int = 3            # moving-average window over motion
    super_full: float = 0.95
    super_spent: float = 0.25


class FootageLabeler:
    """Feed frames in order; get one label per frame.

    `perceive(frame) -> dict` is optional and supplies `player_pos`,
    `ammo_known`, `ammo_count`, `super_charge` (rl/state.GameState fields).
    Without it, the player is assumed centred and only movement is labelled.
    """

    def __init__(self, config: Optional[LabelConfig] = None,
                 perceive: Optional[Callable] = None):
        self.cfg = config or LabelConfig()
        self.perceive = perceive
        self.cam = CameraTracker()
        self._prev_player = None
        self._prev_ammo = None
        self._prev_super = None
        self._motion = deque(maxlen=max(1, self.cfg.smooth_frames))
        self.n = 0

    def reset(self) -> None:
        self.cam.reset()
        self._prev_player = self._prev_ammo = self._prev_super = None
        self._motion.clear()

    def step(self, frame: np.ndarray, rect=None, dt: Optional[float] = None) -> dict:
        cfg = self.cfg
        dt = dt if dt else 1.0 / cfg.fps
        if rect is None:
            from perception.getTerrain import play_rect
            rect = play_rect(frame)
        cx, cy, ok = self.cam.update(frame, rect)

        st = self.perceive(frame) if self.perceive is not None else {}
        player = st.get("player_pos")
        d = (0.0, 0.0)
        if player is not None and self._prev_player is not None:
            d = (player[0] - self._prev_player[0], player[1] - self._prev_player[1])
            # A sprite "jump" is a re-detection, not motion.
            if math.hypot(*d) > 0.25 * rect[2]:
                d = (0.0, 0.0)
        self._prev_player = player if player is not None else self._prev_player

        if ok:
            self._motion.append(((d[0] - cx) / dt, (d[1] - cy) / dt))
        mx = sum(m[0] for m in self._motion) / max(1, len(self._motion))
        my = sum(m[1] for m in self._motion) / max(1, len(self._motion))
        tile_px = rect[2] / cfg.tiles_across
        speed_tps = math.hypot(mx, my) / max(tile_px, 1e-6)
        moving = ok and speed_tps >= cfg.move_threshold_tps
        move = heading = None
        if moving:
            n = math.hypot(mx, my)
            move = (mx / n, my / n)
            heading = int(round(math.atan2(move[1], move[0]) / (2 * math.pi / 16))) % 16

        attack = False
        if st.get("ammo_known"):
            a = st.get("ammo_count")
            if self._prev_ammo is not None and a is not None and a < self._prev_ammo:
                attack = True
            self._prev_ammo = a
        sup = False
        s = st.get("super_charge")
        if s is not None:
            if self._prev_super is not None and self._prev_super >= cfg.super_full \
                    and s <= cfg.super_spent:
                sup = True
            self._prev_super = s

        self.n += 1
        return {
            "frame": self.n - 1, "rect": list(map(int, rect)),
            "cam": [round(cx, 2), round(cy, 2)],
            "player": list(map(int, player)) if player is not None else None,
            "motion": [round(mx, 1), round(my, 1)], "speed_tps": round(speed_tps, 2),
            "moving": bool(moving), "move": [round(v, 3) for v in move] if move else None,
            "heading16": heading, "attack": attack, "super": sup, "ok": bool(ok),
        }


def _live_perceiver():
    """rl/state-style fields from the project's live perception stack."""
    from perception.liveLoop import LivePerception
    from rl.state import adapt_live_state
    lp = LivePerception()
    tick = [0]

    def perceive(frame):
        live, _ = lp.tick(frame)
        s = adapt_live_state(live, tick=tick[0], kills_this_tick=0,
                             frame_size=(frame.shape[1], frame.shape[0]))
        tick[0] += 1
        return {"player_pos": s.player_pos, "ammo_known": s.ammo_known,
                "ammo_count": s.ammo_count, "super_charge": s.super_charge}
    return perceive


def main(argv=None) -> int:
    import cv2
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("video")
    ap.add_argument("--out", default=None, help="labels.jsonl (default: next to the video)")
    ap.add_argument("--step", type=int, default=1, help="label every Nth frame")
    ap.add_argument("--no-perception", action="store_true",
                    help="movement only, player assumed centred (much faster)")
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args(argv)

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        print(f"cannot open {a.video}")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    lab = FootageLabeler(LabelConfig(fps=fps / max(1, a.step)),
                         perceive=None if a.no_perception else _live_perceiver())
    out = pathlib.Path(a.out) if a.out else pathlib.Path(a.video).with_suffix(".labels.jsonl")
    n = moving = attacks = supers = 0
    with open(out, "w") as f:
        idx = 0
        while True:
            okf, frame = cap.read()
            if not okf:
                break
            if idx % a.step == 0:
                rec = lab.step(frame)
                rec["video_frame"] = idx
                rec["t"] = round(idx / fps, 3)
                f.write(json.dumps(rec) + "\n")
                n += 1
                moving += rec["moving"]
                attacks += rec["attack"]
                supers += rec["super"]
                if a.max_frames and n >= a.max_frames:
                    break
            idx += 1
    print(f"{n} frames labelled -> {out}  moving {moving / max(1, n):.0%}  "
          f"attacks {attacks}  supers {supers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
