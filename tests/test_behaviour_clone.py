"""Action recovery for behaviour cloning.

The recording end-to-end cannot be tested here (there is no footage), but the
label reconstruction is pure arithmetic and is exactly where a silent error
would be most damaging: getting the heading convention wrong would train the
policy to walk mirrored, and nothing downstream would complain.
"""

import numpy as np
import pytest

from tools.behaviour_clone import heading_from_velocity, distance_tier_from_run
from rl.actions import N_HEADINGS
from rl.path_planner import WaypointPlanner


@pytest.mark.parametrize("vel,name", [
    ((1, 0), "east"), ((0, 1), "south"), ((-1, 0), "west"), ((0, -1), "north"),
    ((1, 1), "south-east"), ((-1, -1), "north-west"),
])
def test_heading_matches_the_planner_convention(vel, name):
    """Recovered headings must mean the same thing the planner means.

    Heading 0 is RIGHT and indices advance clockwise (y grows downward). If
    this drifts from `WaypointPlanner.target_for`, cloning teaches a mirrored
    or rotated policy and every later measurement is quietly wrong.
    """
    idx = heading_from_velocity(vel[0], vel[1], N_HEADINGS)
    planner = WaypointPlanner(localize=False)
    rect = (0, 0, 1000, 1000)
    tx, ty = planner.target_for(idx, 1, (500, 500), rect)
    got = (tx - 500, ty - 500)
    norm = np.hypot(*got)
    want = np.array(vel, float) / np.linalg.norm(vel)
    assert np.allclose(np.array(got) / norm, want, atol=0.02), (
        f"{name}: heading {idx} points {np.array(got)/norm}, expected {want}")


def test_all_headings_round_trip():
    planner = WaypointPlanner(localize=False)
    for idx in range(N_HEADINGS):
        tx, ty = planner.target_for(idx, 1, (500, 500), (0, 0, 1000, 1000))
        back = heading_from_velocity(tx - 500, ty - 500, N_HEADINGS)
        assert back == idx, f"heading {idx} recovered as {back}"


def test_distance_tier_rises_with_commitment_length():
    assert distance_tier_from_run(1) == 0
    assert distance_tier_from_run(8) == 1
    assert distance_tier_from_run(40) == 2
    tiers = [distance_tier_from_run(n) for n in range(1, 40)]
    assert tiers == sorted(tiers), "tier must be monotonic in run length"
