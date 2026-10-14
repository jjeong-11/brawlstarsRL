"""Perception speed-ups must not change a single output.

Every optimisation in this pass was justified as "same answer, less time".
These tests hold them to the first half of that claim on the committed
fixtures; the second half is printed rather than asserted, because wall time
on a CI box says nothing about the phone loop.
"""

from __future__ import annotations

import glob
import time

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")


def _frames(repo_root):
    paths = sorted(glob.glob(str(repo_root / "media" / "fixtures" / "*.png")))
    paths += sorted(glob.glob(str(repo_root / "media" / "gasphotos" / "*.jpg")))
    frames = [cv2.imread(p) for p in paths]
    frames = [f for f in frames if f is not None]
    if not frames:
        pytest.skip("no fixture frames")
    return frames


def test_hue_rotated_red_mask_is_bit_identical(repo_root):
    from perception import getEnemies as GE
    for f in _frames(repo_root):
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        two_pass = cv2.bitwise_or(
            cv2.inRange(hsv, GE._ENEMY_RED_LOWER1, GE._ENEMY_RED_UPPER1),
            cv2.inRange(hsv, GE._ENEMY_RED_LOWER2, GE._ENEMY_RED_UPPER2))
        assert np.array_equal(GE._enemy_red_mask(hsv), two_pass)


def test_split_cube_masks_give_the_same_cubes(repo_root):
    from perception.getEnemies import find_enemy_health_bars
    from perception.getPickups import cube_masks, find_ground_cubes
    for f in _frames(repo_root):
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        bars = find_enemy_health_bars(f, hsv=hsv)
        a = find_ground_cubes(f, hsv=hsv, bars=bars)
        b = find_ground_cubes(f, hsv=hsv, bars=bars, masks=cube_masks(hsv))
        assert a == b


def test_anchor_with_precomputed_green_is_identical(repo_root):
    from perception.getAnchor import find_player_position_full, green_mask_full
    for f in _frames(repo_root):
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        a = find_player_position_full(f, hsv=hsv)
        b = find_player_position_full(f, hsv=hsv, green_full=green_mask_full(hsv))
        assert a == b


def test_parallel_perception_matches_serial(repo_root, monkeypatch):
    """The threaded LivePerception must produce exactly the serial live dict."""
    import perception.liveLoop as LL
    monkeypatch.setattr(LL, "STAGE_INTERVALS", {k: 0.0 for k in LL.STAGE_INTERVALS})
    monkeypatch.setattr(LL, "get_game_state",
                        lambda frame, hsv=None: {"state": "in_match",
                                                 "brawlers_left": 9, "rank": None})
    frames = [f for f in _frames(repo_root) if f.shape[1] > 2000] or _frames(repo_root)

    def run(parallel):
        lp = LL.LivePerception(async_game_state=False, parallel=parallel)
        outs = []
        t0 = time.perf_counter()
        for i in range(2 * len(frames)):
            live, _ = lp.tick(frames[i % len(frames)])
            outs.append({k: repr(v) for k, v in live.items()})
        dt = (time.perf_counter() - t0) / (2 * len(frames))
        lp.close()
        return outs, dt

    serial, ts = run(False)
    parallel, tp = run(True)
    assert serial == parallel
    print(f"  all stages every tick: serial {1000 * ts:.1f} ms, "
          f"parallel {1000 * tp:.1f} ms")


def test_world_map_derived_layers_are_cached_and_read_only():
    from rl.world_map import WorldMap
    wm = WorldMap()
    wm.set_geometry((0, 0, 1920, 1080), (48, 27))
    a = wm.occupancy
    assert a is wm.occupancy, "occupancy rebuilt on every read"
    with pytest.raises(ValueError):
        a[0, 0] = True                      # read-only: a mutation must fail loudly
    wm.update(None, None, (5.0, 0.0))
    assert wm.occupancy is not a, "cache survived an update"
