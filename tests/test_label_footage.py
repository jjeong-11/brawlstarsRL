"""tools/label_footage.py on synthetic footage with a known trajectory."""

import math
import pathlib
import sys

import cv2
import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.label_footage import FootageLabeler, LabelConfig  # noqa: E402

FPS = 30.0
W, H = 1920, 1080
TILE = W / 17.0
SPEED = 2.4 * TILE / FPS            # px per frame at Shelly's speed


def _world():
    img = cv2.imread(str(ROOT / "media" / "fixtures" / "showdown.png"))
    if img is None:
        pytest.skip("media/fixtures/showdown.png missing")
    play = img[:, 252:252 + W]
    row = np.hstack([play, cv2.flip(play, 1), play])
    return np.vstack([row, cv2.flip(row, 0), row])      # 5760 x 3240, seamless-ish


def _script():
    """(dx, dy) world motion per frame of the player, plus clamp flags."""
    steps = []
    for k in range(16):                                  # every heading, 0.5 s each
        a = 2 * math.pi * k / 16
        steps += [((math.cos(a) * SPEED, math.sin(a) * SPEED), False)] * 15
        steps += [((0.0, 0.0), False)] * 6               # stand still 0.2 s
    # Camera clamped (map edge): the window stops, the sprite walks right.
    steps += [((SPEED, 0.0), True)] * 20
    return steps


def test_movement_labels_recover_the_true_heading():
    world = _world()
    steps = _script()
    cam = np.array([1900.0, 1100.0])     # window top-left in the world
    sprite = np.array([W / 2, H / 2])    # player's position on screen
    state = {"player_pos": tuple(sprite)}
    lab = FootageLabeler(LabelConfig(fps=FPS), perceive=lambda f: dict(state))

    errs, still_moving, clamp_errs = [], 0, []
    labels = []
    for i, ((dx, dy), clamped) in enumerate([((0.0, 0.0), False)] + steps):
        if clamped:
            sprite = sprite + (dx, dy)
        else:
            cam = cam + (dx, dy)
        x0, y0 = int(round(cam[0])), int(round(cam[1]))
        frame = np.ascontiguousarray(world[y0:y0 + H, x0:x0 + W])
        state["player_pos"] = (float(sprite[0]), float(sprite[1]))
        rec = lab.step(frame, rect=(0, 0, W, H))
        labels.append(rec)
        if i < 4:
            continue                                   # tracker + smoothing warm-up
        # Settled = the whole smoothing window (3 frames) agrees. `i` indexes
        # [pad] + steps, so the step that produced this frame is steps[i - 1].
        window = [math.hypot(*steps[j][0]) for j in range(max(0, i - 3), i)]
        if all(w > 0 for w in window):                 # settled, moving frames
            if rec["move"] is None:
                errs.append(180.0)
                continue
            a_true = math.atan2(dy, dx)
            a_lab = math.atan2(rec["move"][1], rec["move"][0])
            e = abs(math.degrees(math.atan2(math.sin(a_lab - a_true), math.cos(a_lab - a_true))))
            (clamp_errs if clamped else errs).append(e)
        elif all(w == 0 for w in window) and rec["moving"]:
            still_moving += 1

    errs = np.array(errs)
    print(f"\n  moving frames {len(errs)}: median {np.median(errs):.1f} deg, "
          f"p95 {np.percentile(errs, 95):.1f}, within 30 deg {np.mean(errs <= 30):.0%}; "
          f"still frames read as moving: {still_moving}; "
          f"clamped-camera median {np.median(clamp_errs):.1f} deg")
    assert np.mean(errs <= 30) >= 0.9
    assert np.median(errs) <= 10
    assert still_moving <= 3
    assert np.median(clamp_errs) <= 10              # sprite motion covers the clamp
    speeds = [r["speed_tps"] for r in labels[10:20]]
    assert 1.8 <= float(np.median(speeds)) <= 3.0    # ~2.4 tiles/s recovered
