"""Arena identification against the reference layouts."""

import cv2
import numpy as np
import pytest

from perception.localize import Localizer, LocalizerConfig
from perception.mapdb import database


@pytest.fixture(scope="module")
def db():
    d = database()
    if len(d) < 10:
        pytest.skip("no map database")
    return d


def _synthetic(gm, side, scale, rng, noise=0.08):
    """A fake accumulated occupancy map cropped out of a real arena."""
    ref = gm.occupancy
    rh, rw = ref.shape
    ty, tx = int(rng.integers(0, rh - side)), int(rng.integers(0, rw - side))
    crop = ref[ty:ty + side, tx:tx + side].astype(np.float32)
    live = cv2.resize(crop, (int(side * scale), int(side * scale)),
                      interpolation=cv2.INTER_NEAREST)
    flip = rng.random(live.shape) < noise
    return (np.where(flip, 1.0 - live, live) > 0.5), (tx, ty)


def _run(loc, live, seen, cfg, grow=None):
    """Feed the localiser until it locks, simulating exploration.

    The map must GROW between attempts. A search is only retried once the
    explored region has expanded by `retry_growth` -- re-running it on identical
    evidence is deterministic and therefore pointless -- so a test that holds
    the patch fixed would never get a second attempt, and never lock.
    """
    for step in range(cfg.attempt_every * 40):
        cur, cur_seen = (grow(step) if grow else (live, seen))
        pose = loc.update(cur, cur_seen)
        if pose is not None:
            return pose
    return None


@pytest.mark.slow
def test_locates_a_large_enough_patch(db):
    """With more than the measured 12x12 threshold explored, it should lock."""
    cfg = LocalizerConfig()
    rng = np.random.default_rng(0)
    side = cfg.min_patch_cells + 4
    ok = wrong = trials = 0
    for gm in list(db)[:12]:
        if min(gm.occupancy.shape) <= side:
            continue
        trials += 1
        live, (tx, ty) = _synthetic(gm, side, 3.0, rng)

        # Reveal the patch outward from the middle, the way a real agent
        # uncovers ground as it walks.
        h, w = live.shape

        def grow(step, live=live, h=h, w=w):
            frac = min(1.0, 0.5 + 0.015 * step)
            seen = np.zeros((h, w), bool)
            y0, y1 = int(h * (1 - frac) / 2), int(h - h * (1 - frac) / 2)
            x0, x1 = int(w * (1 - frac) / 2), int(w - w * (1 - frac) / 2)
            seen[y0:y1, x0:x1] = True
            return live, seen

        pose = _run(Localizer(db=db), live, np.ones_like(live, bool), cfg, grow=grow)
        if pose is None:
            continue
        if pose.game_map.name == gm.name and np.hypot(pose.origin[0] - tx,
                                                      pose.origin[1] - ty) <= 2.0:
            ok += 1
        else:
            wrong += 1
    assert wrong == 0, f"{wrong} WRONG locks -- unrecoverable, must never happen"
    assert ok >= 0.8 * trials, f"only located {ok}/{trials}"


def test_score_thresholds_bracket_the_measured_gap(db):
    """The lock thresholds must sit inside the measured correct/wrong overlap.

    Measured (see LocalizerConfig): correct identifications score 0.649-0.983,
    wrong ones 0.627-0.691. A guard against someone "tuning" these later without
    redoing the measurement -- lowering min_lock_confidence below 0.70 would
    start accepting the wrong matches outright.
    """
    cfg = LocalizerConfig()
    assert cfg.min_lock_confidence > 0.70, (
        "below the worst measured WRONG match (0.691) -- single attempts would "
        "start locking onto the wrong arena")
    assert cfg.lock_outright > cfg.min_lock_confidence
    assert cfg.lock_outright >= 0.90, (
        "a one-attempt lock has to clear every wrong match seen, not just most")
    assert cfg.vote_frames >= 2, (
        "below lock_outright, agreement across independent observations is the "
        "only thing separating correct from wrong")


def test_refuses_a_patch_that_is_too_small(db):
    """THE IMPORTANT ONE.

    Below the measured threshold a single viewport scores ~0.6 against every
    map in the database with 0.017 between first and second place -- so it must
    refuse rather than guess. Refusing costs nothing (the planner keeps using
    the locally fused map); a wrong lock is unrecoverable.
    """
    gm = list(db)[0]
    tiny = gm.occupancy[:6, :6].astype(np.float32)
    live = cv2.resize(tiny, (18, 18), interpolation=cv2.INTER_NEAREST) > 0.5
    loc = Localizer(db=db)
    for _ in range(200):
        loc.update(live, np.ones_like(live, bool))
    assert loc.pose is None, f"locked on a 6x6 patch: {loc.status()}"
    assert "too little" in loc.status()


def test_pose_round_trips_coordinates(db):
    from perception.localize import Pose
    pose = Pose(game_map=list(db)[0], origin=(7.0, 3.0), scale=3.0, confidence=1.0)
    for cell in ((0, 0), (48, 27), (13.5, 4.25)):
        back = pose.to_live_cell(pose.to_map_cell(cell))
        assert np.allclose(back, cell), (cell, back)
