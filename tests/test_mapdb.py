"""The reference map database extracted from media/showdownmaps."""

import numpy as np
import pytest

from perception import mapdb


@pytest.fixture(scope="module")
def db():
    d = mapdb.database()
    if not len(d):
        pytest.skip("no map database (media/showdownmaps missing?)")
    return d


def test_database_is_populated(db):
    assert len(db) >= 60, f"only {len(db)} maps extracted"


def test_blocked_fractions_are_plausible(db):
    """A playable arena is 30-55% obstacles.

    This is the guard that caught the modal-colour bug: on dense maze maps the
    walls cover more area than the floor, so "floor = most common colour"
    picked the wall and reported the map as ~100% blocked.
    """
    blocked = np.array([m.occupancy.mean() for m in db])
    assert blocked.max() <= mapdb.MAX_BLOCKED_FRACTION, (
        f"{[m.name for m in db if m.occupancy.mean() > mapdb.MAX_BLOCKED_FRACTION]} "
        f"look solid -- extraction failed")
    assert 0.25 <= np.median(blocked) <= 0.60, f"median {np.median(blocked):.2f}"


def test_floor_is_mostly_one_region(db):
    """The extracted floor must not be shattered into islands.

    The bar is deliberately low, and the reason is worth stating: `mapdb`
    classifies BUSHES as blocked (they are walkable in game, but they are not
    the floor colour, and the reference map is used for localisation rather
    than for walkability). On a bush-heavy arena that genuinely does cut the
    extracted floor into pieces, so demanding one connected region would fail
    maps that were extracted correctly.

    Measured over the current 71: median 0.99, and the five bushiest sit at
    0.36-0.50. A genuine mis-extraction -- picking a wall colour as the floor --
    produces a far more shattered result than that, so 0.30 separates the two
    cases with room to spare.
    """
    import cv2
    bad = []
    for m in db:
        free = (~m.occupancy).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=8)
        share = (stats[1:, cv2.CC_STAT_AREA].max() / max(1, free.sum())) if n > 1 else 0.0
        if share < 0.30:
            bad.append((m.name, round(float(share), 2)))
    assert not bad, f"floor shattered into islands, extraction likely failed: {bad}"


def test_out_of_bounds_is_blocked(db):
    for m in db:
        assert m.occupancy[~m.in_bounds].all(), (
            f"{m.name}: ground outside the arena is not marked blocked")
