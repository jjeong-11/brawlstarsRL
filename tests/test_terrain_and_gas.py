"""Regressions for four bugs that all looked the same from the outside.

Every one of them showed up as "the bot walks somewhere stupid", and that is
exactly why they need tests rather than a look at the trace: a phantom wall, a
false safe pocket, and a stale map from the previous match all render as a
perfectly sensible A* route around nothing.

  1. Unrecognised pixels counted as walls   -> getTerrain three-way classify
  2. Bushes read as obstacles               -> bush is walkable, unconditionally
  3. Wall pockets inside gas read as safe   -> seen_gas + gas hole closing
  4. World map survived into the next match -> the runner's episode boundary
"""

import numpy as np
import pytest

from perception.getTerrain import (BUSH_WALKABLE_FRACTION, MAX_UNKNOWN_FRACTION,
                                   WALKABLE_FRACTION, WALL_FRACTION,
                                   find_terrain, select_profile)


# --- 1 + 2: terrain classification ----------------------------------------- #
def test_unclassifiable_ground_is_unknown_not_blocked(showdown_frame):
    """The core fix. A cell the palette cannot explain is evidence of nothing.

    The old rule was `blocked = walk_frac < threshold`, so every pixel outside
    the floor/bush bounds became a wall. Measured on this fixture with the
    best-matching profile, 39% of pixels match no class at all — which is how
    47% of the grid ended up solid on a map that is 30-50% obstacles.
    """
    t = find_terrain(showdown_frame)
    assert t is not None, "the committed fixture must still classify"

    occ, unknown, walk = t["occupancy"], t["unknown"], t["walkable"]
    old_rule_blocked = (walk < WALKABLE_FRACTION)

    print(f"blocked {occ.mean():.3f} (old rule would say "
          f"{old_rule_blocked.mean():.3f}), unknown {unknown.mean():.3f}")
    assert occ.mean() < old_rule_blocked.mean(), "no phantom walls were removed"
    # A Showdown map is 30-50% obstacles; well under half is the sane range.
    assert occ.mean() < 0.35, f"still walling off {occ.mean():.0%} of the map"
    # ...but it must not have gone the other way and dissolved every wall.
    assert occ.sum() > 50, "no walls detected at all — the fix overshot"


def test_bush_is_never_an_obstacle(showdown_frame):
    """Bush is walkable ground in Brawl Stars, whatever else is in the cell.

    This is the fix for gas-tinted bushes: the overlay shifts their hue out of
    the bush bounds, the cell then matches nothing, and under the old rule it
    became a wall. Encoding "bush is walkable" as a fact about the game rather
    than about the palette makes that impossible by construction.
    """
    t = find_terrain(showdown_frame)
    bushy = t["bush"] >= BUSH_WALKABLE_FRACTION
    assert bushy.sum() > 0, "fixture has no bushes; the test proves nothing"
    blocked_bush = int((bushy & t["occupancy"]).sum())

    old_rule = (t["walkable"] < WALKABLE_FRACTION)
    print(f"{int(bushy.sum())} bushy cells; blocked now {blocked_bush}, "
          f"old rule blocked {int((bushy & old_rule).sum())}")
    assert blocked_bush == 0


def test_blocked_is_either_a_wall_or_off_the_map(showdown_frame):
    """Two ways to be blocked, and nothing may sneak in a third.

    A cell is impassable because it LOOKS like a wall, or because it is
    unclassified ground connected to the frame edge — the decoration past the
    arena border, which is blocked by topology rather than by colour. What must
    never happen again is the old third way: blocked because the palette simply
    did not recognise it.
    """
    t = find_terrain(showdown_frame)
    occ, wall, walk, off = (t["occupancy"], t["wall"], t["walkable"],
                            t["off_map"])
    looks_like_wall = (wall >= WALL_FRACTION) | (wall > walk)
    bad = occ & ~looks_like_wall & ~off
    print(f"blocked {int(occ.sum())} = {int((occ & looks_like_wall).sum())} wall "
          f"+ {int((occ & off & ~looks_like_wall).sum())} off-map")
    assert not bad.any(), f"{int(bad.sum())} cells blocked for no reason at all"


def test_off_map_decoration_is_blocked_not_unknown(showdown_frame):
    """The other half of the fix, and the one that carries load elsewhere.

    `PathPlannerConfig.border_cost` infers "close to the map edge" from blocked
    ground nearby, with no map knowledge at all. If off-map decoration were
    merely `unknown`, the world map's optimistic prior would say the agent may
    walk off the arena AND the border penalty would silently stop working —
    which is the failure that used to pull every destination onto the edge ring.
    """
    t = find_terrain(showdown_frame)
    off = t["off_map"]
    assert off.any(), "no off-map region found on a frame that has one"
    assert (t["occupancy"] | ~off).all(), "off-map cells are not blocked"
    assert not (off & t["unknown"]).any(), "off-map cells leaked into unknown"


def test_unclassified_islands_stay_unknown():
    """An unreadable patch surrounded by classified ground is NOT off-map.

    The whole point of the flood is that it is topological, so this is the case
    that proves it discriminates rather than just blocking everything it cannot
    read.
    """
    from perception.getTerrain import _flood_from_border

    mask = np.zeros((27, 48), bool)
    mask[0:3, :] = True           # a band along the top edge  -> off map
    mask[12:15, 20:24] = True     # an island in the middle    -> unknown
    out = _flood_from_border(mask)
    assert out[0:3, :].all(), "the border band was not reached"
    assert not out[12:15, 20:24].any(), "an enclosed island was called off-map"


def test_a_frame_nothing_matches_returns_none():
    """Refusing beats guessing: the caller falls back to direct steering.

    Now that unrecognised ground reports `unknown` rather than `blocked`, a bad
    profile no longer trips MAX_BLOCKED_FRACTION — it yields a grid that is
    almost entirely unknown. MAX_UNKNOWN_FRACTION is the matching guard.
    """
    flat = np.full((1080, 2424, 3), 200, np.uint8)      # featureless grey
    assert find_terrain(flat) is None


def test_hud_cells_are_unknown_not_walls(showdown_frame):
    """HUD overlays sit ON TOP of terrain; the pixels underneath are unreadable.
    They must inherit from neighbours, never become walls."""
    t = find_terrain(showdown_frame)
    assert t["unknown"].any()
    assert not (t["occupancy"] & t["unknown"]).any()


# --- 3: gas pockets --------------------------------------------------------- #
def _world_with_gas(gas_grid, occ_grid=None):
    """A WorldMap fused with one gas reading (and optional terrain)."""
    from rl.world_map import WorldMap
    gh, gw = gas_grid.shape
    w = WorldMap()
    w._reset_arrays(gw, gh)
    w.set_geometry((0, 0, gw * 10, gh * 10), (gw, gh))
    terrain = None
    if occ_grid is not None:
        terrain = {
            "occupancy": occ_grid,
            "bush": np.zeros_like(gas_grid),
            "unknown": np.zeros(gas_grid.shape, bool),
            "rect": (0, 0, gw * 10, gh * 10),
            "cell": (10.0, 10.0),
        }
    for _ in range(6):          # a few frames so the EMA settles
        w.update(terrain, gas_grid)
    return w


def test_walls_inside_the_cloud_are_not_credited_as_gas_free():
    """The pocket bug, at its source.

    Gas is drawn on the FLOOR, so a wall block inside the cloud reports gas=0
    however deep in it sits. Fusing that zero as an *observation* is the
    strongest claim we can make, from the one place we cannot make it.
    """
    gh, gw = 27, 48
    gas = np.ones((gh, gw), np.float32)
    occ = np.zeros((gh, gw), bool)
    occ[10:16, 20:28] = True          # a block of wall, deep inside the cloud
    gas[occ] = 0.0                    # ...which therefore reads as gas-free

    w = _world_with_gas(gas, occ)
    ys, xs = np.nonzero(occ)
    cy, cx = int(np.mean(ys)), int(np.mean(xs))
    oy, ox = int(w.origin[1]), int(w.origin[0])

    assert w.gas_unobserved[oy + cy, ox + cx], \
        "a wall cell was credited as gas-observed"
    # Open ground inside the same cloud must still count as observed, or the
    # fix would just blind the detector everywhere.
    assert not w.gas_unobserved[oy + 2, ox + 2]


def test_gas_field_fills_a_pocket_enclosed_by_cloud():
    """Belt and braces: whatever caused the hole, geometry closes it.

    An island of clear ground surrounded by gas is not a refuge — and it is by
    construction the NEAREST safe ground, so it is maximally attractive to the
    planner. That is why it is worth closing off independently of the cause.
    """
    from rl.path_planner import WaypointPlanner

    gh, gw = 27, 48
    gas = np.ones((gh, gw), np.float32)
    gas[12:15, 22:26] = 0.0                 # a small hole in the middle
    planner = WaypointPlanner()
    planner.world = _world_with_gas(gas)

    field = planner._gas_field()
    oy, ox = int(planner.world.origin[1]), int(planner.world.origin[0])
    hole = field[oy + 12:oy + 15, ox + 22:ox + 26]
    print(f"hole gas after closing: min {hole.min():.2f} max {hole.max():.2f}")
    assert hole.min() > 0.5, "the enclosed pocket still reads as safe"


def test_closing_does_not_inflate_the_real_cloud_boundary():
    """Closing may only fill holes. If it grew the cloud outward, the agent
    would flee ground that is genuinely safe — and in the endgame, when the
    safe zone is small, that is fatal."""
    from rl.path_planner import PathPlannerConfig, WaypointPlanner
    from rl.path_planner import _close_grid

    gh, gw = 40, 60
    gas = np.zeros((gh, gw), np.float32)
    gas[:, :20] = 1.0                        # a clean half-plane of gas
    closed = _close_grid(gas, PathPlannerConfig().gas_close_cells)
    assert np.array_equal(closed, gas), "closing moved the true boundary"


def test_gas_close_cells_can_be_disabled():
    from rl.path_planner import _close_grid
    g = np.random.default_rng(0).random((20, 20)).astype(np.float32)
    assert np.array_equal(_close_grid(g, 0), g)


# --- 4: the runner's episode boundary --------------------------------------- #
def test_runner_resets_the_world_between_matches():
    """The stale-map bug: `planner.reset()` is what clears `planner.world`.

    Asserted on the real objects rather than the loop, because the loop needs a
    phone. What the loop must do is call this — which the next test pins down.
    """
    from rl.path_planner import WaypointPlanner

    p = WaypointPlanner()
    p.world.occ_p[:] = 0.9          # a previous match's walls
    p.world.gas[:] = 1.0
    p.world.seen[:] = 1.0
    assert p.world.occupancy.any()

    p.reset()
    assert not p.world.occupancy.any(), "last match's walls survived the reset"
    assert p.world.gas.max() == 0.0
    assert p.world.unexplored.all()


def test_runner_loop_drives_the_menus_and_rebuilds_state():
    """The web UI runner must do what `rl/env.py:reset()` does.

    It reimplements the per-step work, and the first version omitted the whole
    episode lifecycle — which is why Exit was never pressed and the map never
    cleared. Checking the source is crude, but the alternative is a phone.
    """
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent.parent
           / "webui" / "runner.py").read_text()
    assert "navigate_to_match" in src, "nothing ever presses Exit / Play Again"
    assert "def new_match" in src, "no per-match state rebuild"
    for piece in ("WaypointPlanner(", "CameraTracker(", "ProfileSelector(",
                  "LivePerception("):
        assert src.count(piece) >= 1, f"{piece} is never rebuilt per match"
    assert "state.match_over or not state.is_alive" in src, \
        "no end-of-match trigger"


@pytest.mark.parametrize("screen,expect_button", [
    ("defeated", True), ("match_end", True), ("in_match", False),
    ("loading", False), ("unknown", False),
])
def test_menu_buttons_cover_every_screen_that_has_one(screen, expect_button):
    """`navigate_to_match` taps `BUTTONS[screen]`, so a screen missing from that
    table is a screen the session will sit on forever."""
    from rl.menus import BUTTONS
    assert (screen in BUTTONS) is expect_button


def test_defeated_screen_is_recognised_and_maps_to_exit(repo_root):
    """End to end on the committed fixture: the death screen classifies as
    `defeated`, and `defeated` has an Exit button to press."""
    import cv2
    from perception.getGameState import get_game_state
    from rl.menus import BUTTONS, EXIT_BUTTON_NORM

    img = cv2.imread(str(repo_root / "media" / "fixtures" / "defeated.png"))
    if img is None:
        pytest.skip("media/fixtures/defeated.png not present")
    state = get_game_state(img)["state"]
    assert state == "defeated"
    assert BUTTONS[state] == EXIT_BUTTON_NORM


def test_death_ends_the_episode():
    """`terminated` must fire on the death screen, not only on the results
    screen — otherwise the loop keeps stepping while a menu is up."""
    from rl.state import adapt_live_state

    live = {"game_state": {"state": "defeated", "brawlers_left": None,
                           "rank": None}}
    s = adapt_live_state(live)
    assert not s.is_alive and not s.match_over
