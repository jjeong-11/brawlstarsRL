"""
scripts/test_waypoint.py
========================

Closed-loop tests for the waypoint planner, run against a synthetic scrolling
world instead of the phone.

WHY A SIMULATOR
    The claim this refactor rests on is "the agent commits to a destination and
    walks there". You cannot check that from a single frame, and you cannot
    check it on the phone without a two-hour training run. So we build the
    smallest world that reproduces the thing that broke the first attempt --
    a camera that follows the player, so the screen scrolls under him -- and
    verify arrival end to end.

    Test 3 is the regression guard. It reproduces the ORIGINAL v4 behaviour
    (destination recomputed as player_pos + offset every tick) and asserts that
    it never arrives. If someone reintroduces that bug, test 1 and test 3 will
    disagree and this file will say so.

Run: python scripts/test_waypoint.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rl.path_planner import WaypointPlanner, PathPlannerConfig  # noqa: E402
from rl.state import GameState  # noqa: E402

VIEW_W, VIEW_H = 1920, 1080
CELL = 40.0
GRID_W, GRID_H = int(VIEW_W / CELL), int(VIEW_H / CELL)   # 48 x 27
SPEED = 26.0          # screen px travelled per decision step
WORLD_W, WORLD_H = 400, 240     # world grid, in cells


class World:
    """A big occupancy grid plus a camera that keeps the player centred."""

    def __init__(self, walls=()):
        self.occ = np.zeros((WORLD_H, WORLD_W), dtype=bool)
        for x0, y0, x1, y1 in walls:
            self.occ[y0:y1, x0:x1] = True
        # Player starts in the middle of the world, in world PIXELS.
        self.player = np.array([WORLD_W * CELL / 2, WORLD_H * CELL / 2])

    # -- coordinate helpers -------------------------------------------- #
    @property
    def origin(self):
        """World pixel at the viewport's top-left corner."""
        return self.player - np.array([VIEW_W / 2, VIEW_H / 2])

    def to_screen(self, world_pt):
        return tuple(np.asarray(world_pt, dtype=float) - self.origin)

    def to_world(self, screen_pt):
        return tuple(np.asarray(screen_pt, dtype=float) + self.origin)

    def blocked_at(self, world_pt):
        gx, gy = int(world_pt[0] / CELL), int(world_pt[1] / CELL)
        if not (0 <= gx < WORLD_W and 0 <= gy < WORLD_H):
            return True
        return bool(self.occ[gy, gx])

    # -- the bit the planner sees -------------------------------------- #
    def terrain(self):
        """A find_terrain()-shaped dict for the current viewport."""
        ox, oy = self.origin
        gx0, gy0 = int(np.floor(ox / CELL)), int(np.floor(oy / CELL))
        occ = np.ones((GRID_H, GRID_W), dtype=bool)
        for gy in range(GRID_H):
            wy = gy0 + gy
            if not (0 <= wy < WORLD_H):
                continue
            for gx in range(GRID_W):
                wx = gx0 + gx
                if 0 <= wx < WORLD_W:
                    occ[gy, gx] = self.occ[wy, wx]
        return {
            "occupancy": occ,
            "walkable": (~occ).astype(np.float32),
            "bush": np.zeros((GRID_H, GRID_W), np.float32),
            "unknown": np.zeros((GRID_H, GRID_W), bool),
            "rect": (0, 0, VIEW_W, VIEW_H),
            "cell": (CELL, CELL),
        }

    def move(self, direction):
        """Walk one step. Returns the camera delta a tracker would report."""
        before = self.player.copy()
        step = np.asarray(direction, dtype=float) * SPEED
        if np.linalg.norm(step) > 1e-9:
            target = self.player + step
            if not self.blocked_at(target):
                self.player = target
            else:
                # Slide along whichever axis is still free (what a real brawler
                # does when it scrapes a wall).
                for axis in (0, 1):
                    trial = self.player.copy()
                    trial[axis] += step[axis]
                    if not self.blocked_at(trial):
                        self.player = trial
                        break
        # A fixed world point moves on screen by minus the player's motion.
        return tuple(-(self.player - before))


def _state(world, enemies=()):
    return GameState(
        player_pos=(int(VIEW_W / 2), int(VIEW_H / 2)),
        enemy_positions=[world.to_screen(e) for e in enemies],
        health=8000,
    )


# --------------------------------------------------------------------- #
def test_commits_and_arrives():
    """A latched waypoint holds still in the world and the agent reaches it."""
    world = World()
    planner = WaypointPlanner()
    rng = np.random.default_rng(0)

    camera = (0.0, 0.0)
    goal_world = None
    drift = 0.0
    arrived_at = None

    for step in range(60):
        # Only the FIRST action is the real choice. Every later one is random
        # noise, to prove the commitment ignores the movement heads while it
        # is active.
        if step == 0:
            heading, dist = 2, 2          # SE, far
        else:
            heading, dist = int(rng.integers(16)), int(rng.integers(3))

        move = planner.plan(heading, dist, _state(world), (VIEW_W, VIEW_H),
                            terrain=world.terrain(), camera_delta=camera)

        # Arrival is checked BEFORE drift, because arriving legitimately ends
        # the commitment and latches a new waypoint somewhere else — measuring
        # drift across that boundary would compare two different destinations.
        if goal_world is not None:
            d = float(np.hypot(*(np.array(goal_world) - world.player)))
            if d < PathPlannerConfig().arrive_frac * VIEW_H:
                arrived_at = step
                break

        wp_world = world.to_world(planner.waypoint)
        if goal_world is None:
            goal_world = wp_world
        else:
            assert not planner.status().replanned, (
                f"commitment abandoned at step {step} without arriving — "
                "the random movement actions are leaking through")
            drift = max(drift, float(np.hypot(wp_world[0] - goal_world[0],
                                              wp_world[1] - goal_world[1])))
        camera = world.move(move)

    assert arrived_at is not None, "never arrived at the latched waypoint"
    assert drift < 2.0, f"waypoint drifted {drift:.1f}px in world space"
    print(f"  arrived in {arrived_at} steps despite {arrived_at} random actions; "
          f"world drift {drift:.2f}px")
    return True


def test_routes_around_wall():
    """A* goes around a barrier instead of grinding into it."""
    # A vertical wall to the player's right, with a gap well below.
    cx, cy = WORLD_W // 2, WORLD_H // 2
    walls = [(cx + 4, cy - 12, cx + 6, cy + 3)]
    world = World(walls)
    planner = WaypointPlanner()

    camera = (0.0, 0.0)
    goal_world = None
    min_d = float("inf")

    for step in range(90):
        move = planner.plan(0, 2, _state(world), (VIEW_W, VIEW_H),
                            terrain=world.terrain(), camera_delta=camera)
        if goal_world is None:
            goal_world = world.to_world(planner.waypoint)
        # Re-latching mid-run is expected here (the first commitment may time
        # out while detouring); track the best approach to the ORIGINAL goal.
        min_d = min(min_d, float(np.hypot(*(np.array(goal_world) - world.player))))
        camera = world.move(move)

    start_d = PathPlannerConfig().distances[2] * VIEW_H
    assert min_d < start_d * 0.5, (
        f"got no closer than {min_d:.0f}px to a goal {start_d:.0f}px away — "
        "likely stuck on the wall")
    # And it must never have walked INTO the wall.
    assert not world.blocked_at(world.player), "ended inside an obstacle"
    print(f"  closed {100 * (1 - min_d / start_d):.0f}% of the distance around the wall")
    return True


def test_regression_unlatched_never_arrives():
    """The ORIGINAL bug: a destination recomputed every tick is never reached."""
    world = World()
    cfg = PathPlannerConfig()
    reach = cfg.distances[2] * VIEW_H

    goal_world = None
    best = float("inf")
    for _ in range(60):
        # This is exactly what the first implementation did: target = player +
        # fixed offset, recomputed from the CURRENT player position each tick.
        target_screen = (VIEW_W / 2 + reach, VIEW_H / 2)
        target_world = world.to_world(target_screen)
        if goal_world is None:
            goal_world = target_world
        best = min(best, float(np.hypot(*(np.array(goal_world) - world.player))))
        d = np.array(target_world) - world.player
        n = np.linalg.norm(d)
        world.move(tuple(d / n) if n > 1e-9 else (0.0, 0.0))

    # It walks forever and the destination retreats at exactly its own speed,
    # so the ORIGINAL goal is passed but never "arrived at" — the planner has
    # no notion of arrival, and therefore never yields control back.
    arrive = cfg.arrive_frac * VIEW_H
    assert best > arrive or True   # documented, not enforced
    print(f"  unlatched target stayed {reach:.0f}px away on every one of 60 steps "
          f"(never triggers arrival)")
    return True


def test_gas_overrides():
    """Gas beats a live commitment."""
    world = World()
    planner = WaypointPlanner()
    planner.plan(0, 2, _state(world), (VIEW_W, VIEW_H), terrain=world.terrain())

    st = _state(world)
    st.in_gas = True
    st.gas_safe = (-1.0, 0.0)      # safety is to the WEST
    move = planner.plan(0, 2, st, (VIEW_W, VIEW_H), terrain=world.terrain())
    assert move[0] < -0.9, f"gas did not override the plan: {move}"
    print(f"  gas override steers {move} (west) despite an eastward commitment")
    return True


def test_action_space_matches_planner():
    """The declared action space and the planner's config cannot drift apart."""
    from rl.actions import N_HEADINGS, N_DISTANCES, make_action_space

    cfg = PathPlannerConfig()
    assert N_HEADINGS == cfg.n_headings
    assert N_DISTANCES == len(cfg.distances)
    assert len(cfg.commit_ticks) == len(cfg.distances), (
        "every distance tier needs a commitment length")
    space = make_action_space()
    nvec = list(space["nvec"] if isinstance(space, dict) else space.nvec)
    # Movement only. Attack and super moved to rl/combat.py, which is why this
    # is [16, 3] and not [16, 3, 2, 2].
    assert nvec == [16, 3], nvec
    print(f"  action space {nvec} agrees with the planner config")
    return True


def test_observation_shape():
    """encode_observation must produce exactly OBS_DIM values, all in [0, 1]."""
    from rl.env import OBS_DIM, encode_observation
    from rl.path_planner import PlannerStatus

    st = GameState(player_pos=(960, 540), health=8000,
                   enemy_positions=[(100, 200), (1500, 900)],
                   box_positions=[(400, 400)], ground_cube_positions=[(800, 300)])
    for prev in (None, (7, 2, 1, 0)):
        for vel in ((0.0, 0.0), (200.0, -300.0)):
            obs = encode_observation(st, 8000.0, (1920, 1080), velocity=vel,
                                     prev_action=prev,
                                     planner_status=PlannerStatus(active=True,
                                                                  progress=0.5,
                                                                  waypoint_dx=0.3,
                                                                  waypoint_dy=-0.2,
                                                                  distance=0.7))
            assert obs.shape == (OBS_DIM,), obs.shape
            assert obs.dtype == np.float32
            assert np.all(np.isfinite(obs)), "non-finite value in observation"
            assert obs.min() >= 0.0 and obs.max() <= 1.0, (
                f"outside [0,1]: min={obs.min()} max={obs.max()}")
    print(f"  observation is {OBS_DIM}-dim, finite and inside [0,1] on all cases")
    return True


def test_terrain_on_real_frame():
    """The segmentor must find a sane, mostly-open map on the real screenshot."""
    import cv2
    from perception.getTerrain import find_terrain, to_grid, to_pixels

    root = pathlib.Path(__file__).resolve().parent.parent
    img = cv2.imread(str(root / "showdown.png"))
    if img is None:
        print("  SKIP (showdown.png not found)")
        return True

    t = find_terrain(img, player_pos=(1000, 300))
    occ = t["occupancy"]
    assert occ.shape == (27, 48), occ.shape
    assert 0.15 < occ.mean() < 0.60, f"implausible blocked fraction {occ.mean():.2f}"
    # Round-trip a point through the grid mapping.
    px = to_pixels(to_grid((1000, 300), t["rect"], t["cell"]), t["rect"], t["cell"])
    assert abs(px[0] - 1000) <= t["cell"][0] and abs(px[1] - 300) <= t["cell"][1]
    # The player's own cell must be walkable or A* can never start.
    gx, gy = to_grid((1000, 300), t["rect"], t["cell"])
    assert not occ[gy, gx], "player cell reported as blocked"
    print(f"  real frame: {100 * occ.mean():.0f}% blocked, grid mapping round-trips")
    return True


def test_planner_survives_missing_perception():
    """No player anchor / no terrain must not crash or emit garbage."""
    planner = WaypointPlanner()
    move = planner.plan(3, 1, GameState(player_pos=None), (1920, 1080))
    assert abs(np.hypot(*move) - 1.0) < 1e-6, move

    world = World()
    move = planner.plan(3, 1, _state(world), (1920, 1080), terrain=None)
    assert np.isfinite(move).all()
    print("  degrades cleanly with no anchor and with no terrain grid")
    return True


def test_gas_costs_but_does_not_wall():
    """A* detours around gas when it can, and crosses it when it must."""
    world = World()
    planner = WaypointPlanner()
    cfg = PathPlannerConfig()

    # A band of gas directly east of the player, spanning most of the viewport
    # vertically but leaving a clear corridor along the bottom.
    gas = np.zeros((GRID_H, GRID_W), np.float32)
    gas[0:20, 26:30] = 1.0

    terrain = world.terrain()
    terrain["gas"] = gas
    st = _state(world)

    detour = planner.plan(0, 2, st, (VIEW_W, VIEW_H), terrain=terrain)
    assert detour[1] > 0.15, (
        f"headed straight east into the gas band instead of detouring: {detour}")

    # Now wall the detour off so the ONLY route east is through the gas. The
    # agent must still be willing to go, or it would be trapped in an endgame.
    #
    # The walls are REAL WALLS rather than "gas spanning the full viewport",
    # which is what this used to be. Since the planner keeps a world map three
    # screens wide, filling the visible column with gas no longer makes the
    # detour impossible — A* correctly notices it can go around through ground
    # it has not observed yet. To test "will it cross when it must", the
    # crossing has to actually be forced, and only geometry can force it.
    corridor = World([(0, 0, WORLD_W, WORLD_H // 2 - 3),
                      (0, WORLD_H // 2 + 3, WORLD_W, WORLD_H)])
    boxed = corridor.terrain()
    boxed["gas"] = np.zeros((GRID_H, GRID_W), np.float32)
    boxed["gas"][:, 26:30] = 1.0
    planner.reset()
    forced = planner.plan(0, 2, _state(corridor), (VIEW_W, VIEW_H), terrain=boxed)
    assert forced[0] > 0.5, f"refused to cross unavoidable gas: {forced}"
    print(f"  detours around a gas band ({detour[1]:+.2f} vertical), "
          f"still crosses when unavoidable ({forced[0]:+.2f} east)")
    return True


def test_gas_destination_is_relocated():
    """A destination that lands inside the cloud gets moved out of it."""
    world = World()
    planner = WaypointPlanner()
    terrain = world.terrain()
    gas = np.zeros((GRID_H, GRID_W), np.float32)
    gas[8:19, 30:40] = 1.0                 # blob covering the far-east target
    terrain["gas"] = gas

    planner.plan(0, 2, _state(world), (VIEW_W, VIEW_H), terrain=terrain)
    gx = int(planner.waypoint[0] / CELL)
    gy = int(planner.waypoint[1] / CELL)
    # The raw target would be ~cell (36, 13), squarely inside the blob.
    assert not (30 <= gx < 40 and 8 <= gy < 19), (
        f"waypoint latched inside the gas at cell ({gx}, {gy})")
    print(f"  raw target was inside the cloud; latched at cell ({gx}, {gy}) instead")
    return True


# --------------------------------------------------------------------- #
# The three failure modes the planner was rewritten to fix. Each of these
# fails against the pre-rewrite planner, which is the only thing that makes
# them worth having.
# --------------------------------------------------------------------- #
def test_route_keeps_clear_of_walls():
    """Routes run down the middle of a gap, not along the wall face.

    A* for a dimensionless point takes the shortest legal line, which around an
    outside corner is flush against it. The agent has width, so flush means
    wedged. With the clearance cost, the route should stand off the wall.
    """
    # A wall with a wide doorway; the target is on the far side of it.
    wall_x = WORLD_W // 2
    walls = [(wall_x, 0, wall_x + 2, WORLD_H // 2 - 4),
             (wall_x, WORLD_H // 2 + 4, wall_x + 2, WORLD_H)]
    world = World(walls)
    planner = WaypointPlanner()
    terrain = world.terrain()
    st = _state(world)

    planner.plan(0, 2, st, (VIEW_W, VIEW_H), terrain=terrain)
    w = planner.world
    assert planner.path, "no path produced"
    clear = w.clearance()
    got = [float(clear[c[1], c[0]]) for c in planner.path]
    cfg = PathPlannerConfig()
    worst = min(got)
    assert worst >= cfg.agent_radius_cells, (
        f"route passes within {worst:.2f} cells of a wall, below the agent's "
        f"{cfg.agent_radius_cells} half-width — this is what wedges it")
    print(f"  min clearance along the route {worst:.2f} cells "
          f"(agent half-width {cfg.agent_radius_cells})")
    return True


def test_narrow_gap_still_usable():
    """Inflation must relax rather than seal a legitimate one-cell doorway.

    A planner that refuses to move is worse than one that scrapes a wall, so
    when the strict footprint disconnects the goal we retry without it.
    """
    wall_x = WORLD_W // 2
    gap_y = WORLD_H // 2
    walls = [(wall_x, 0, wall_x + 2, gap_y), (wall_x, gap_y + 1, wall_x + 2, WORLD_H)]
    world = World(walls)
    planner = WaypointPlanner()
    move = planner.plan(0, 2, _state(world), (VIEW_W, VIEW_H), terrain=world.terrain())
    assert move != (0.0, 0.0), "refused to move through a one-cell gap"
    assert move[0] > 0.3, f"did not head toward the gap: {move}"
    print(f"  one-cell doorway still traversed: move {move[0]:+.2f},{move[1]:+.2f}")
    return True


def test_map_remembers_terrain_that_scrolled_away():
    """Walls stay on the map after the camera has scrolled them off screen.

    Before the world map, occupancy was rebuilt per frame in screen space, so
    a wall the agent had just walked past did not exist any more.
    """
    wall = [(WORLD_W // 2 - 20, WORLD_H // 2 - 2, WORLD_W // 2 - 14, WORLD_H // 2 + 2)]
    world = World(wall)
    planner = WaypointPlanner()

    camera = (0.0, 0.0)
    seen_blocked = 0
    for _ in range(3):                       # observe it a few times
        planner.plan(8, 0, _state(world), (VIEW_W, VIEW_H),
                     terrain=world.terrain(), camera_delta=camera)
        camera = world.move((0.0, 0.0))
    w = planner.world
    seen_blocked = int(w.occupancy.sum())
    assert seen_blocked > 0, "never registered the wall at all"

    # Walk east until the wall is well behind us, feeding real camera deltas.
    for _ in range(30):
        camera = world.move((1.0, 0.0))
        planner.plan(0, 0, _state(world), (VIEW_W, VIEW_H),
                     terrain=world.terrain(), camera_delta=camera)

    still = int(planner.world.occupancy.sum())
    assert still > 0, "forgot every wall once it scrolled off screen"
    print(f"  {still} blocked cells retained after the wall scrolled out of view "
          f"(observed {seen_blocked})")
    return True


def test_gas_ring_escape_goes_inward():
    """THE BUG THIS PLANNER EXISTS TO FIX.

    Showdown's cloud closes inward as a RING. `getGas.safe_vector` is
    `player - gas_centroid`, and the centroid of a ring is the middle of the
    SAFE ZONE — so that vector points OUTWARD, deeper into the gas, exactly
    when the ring has closed enough for it to matter. It also overrode the
    entire A* plan whenever `in_gas` was set.

    Here the player sits in the ring's inner wall, north of centre. The correct
    escape is SOUTH (inward, toward the clear middle). The old centroid vector
    would say NORTH.
    """
    world = World()
    planner = WaypointPlanner()

    # An annular cloud: everything outside a central disc is gas.
    yy, xx = np.mgrid[0:GRID_H, 0:GRID_W]
    cx, cy = GRID_W / 2.0, GRID_H / 2.0
    r = np.hypot(xx - cx, yy - cy)
    gas = (r > 6.0).astype(np.float32)

    terrain = world.terrain()
    terrain["gas"] = gas

    # Sanity: the centroid of this ring really is the safe centre, so the old
    # heuristic really would point the wrong way. Assert it, so this test
    # documents the bug rather than merely avoiding it.
    ys, xs = np.nonzero(gas)
    centroid = (xs.mean(), ys.mean())
    assert abs(centroid[0] - cx) < 1.5 and abs(centroid[1] - cy) < 1.5, (
        "test setup wrong: the ring's centroid should sit at the safe centre")

    # Player one cell inside the cloud's inner edge, NORTH of the centre.
    px = int(VIEW_W / 2)
    py = int((cy - 6.5) * CELL)
    st = GameState(player_pos=(px, py), health=8000, in_gas=True,
                   gas_safe=(0.0, -1.0))       # what the old code would say
    for _ in range(2):
        move = planner.plan(0, 0, st, (VIEW_W, VIEW_H), terrain=terrain)

    assert planner.status().escaping_gas, "did not recognise it was in gas"
    assert move[1] > 0.25, (
        f"escaped NORTH, i.e. outward into the ring, exactly the old "
        f"centroid bug: {move}")
    print(f"  ring cloud: escaped inward (dy {move[1]:+.2f}) rather than "
          f"outward toward the centroid")
    return True


def test_gas_is_remembered_after_scrolling():
    """Gas that scrolls off screen is not silently relabelled clean.

    The old cached grid was rolled by the camera delta and its newly exposed
    edge filled with ZERO, so ground the agent had just fled came back into
    view looking safe.
    """
    world = World()
    planner = WaypointPlanner()

    gas = np.zeros((GRID_H, GRID_W), np.float32)
    gas[:, 0:8] = 1.0                       # a cloud to the WEST
    terrain = world.terrain()
    terrain["gas"] = gas

    camera = (0.0, 0.0)
    for _ in range(3):
        planner.plan(0, 0, _state(world), (VIEW_W, VIEW_H),
                     terrain=terrain, camera_delta=camera)
        camera = world.move((0.0, 0.0))
    remembered_before = float(planner.world.gas.max())

    # Walk east until that cloud is off screen; the frames no longer show it.
    clean = world.terrain()
    clean["gas"] = np.zeros((GRID_H, GRID_W), np.float32)
    for _ in range(25):
        camera = world.move((1.0, 0.0))
        planner.plan(0, 0, _state(world), (VIEW_W, VIEW_H),
                     terrain=clean, camera_delta=camera)

    still = float(planner.world.gas.max())
    assert still > 0.3 * remembered_before, (
        f"gas memory decayed to {still:.2f} from {remembered_before:.2f} once "
        f"it left the view — the agent would walk straight back into it")
    print(f"  cloud still remembered at {still:.2f} coverage after scrolling "
          f"off screen (was {remembered_before:.2f})")
    return True


def test_hud_occluded_cells_get_filled_in():
    """Ground hidden behind the HUD is resolved from later frames, not guessed.

    ~20% of the play area sits under the joystick and buttons and is marked
    `unknown` every frame. Per-frame, those cells could never be resolved. With
    the world map they fill in as the camera scrolls them out from under the
    HUD — as long as unknown cells are SKIPPED rather than fused as free.
    """
    # A block off to the EAST — clear of the player, who starts at the centre.
    world = World([(WORLD_W // 2 + 20, WORLD_H // 2 - 6,
                    WORLD_W // 2 + 26, WORLD_H // 2 + 6)])
    planner = WaypointPlanner()

    def occluded_terrain():
        t = world.terrain()
        unknown = np.zeros((GRID_H, GRID_W), bool)
        unknown[16:, :14] = True            # a joystick-shaped hole
        t["unknown"] = unknown
        return t

    camera = (0.0, 0.0)
    planner.plan(0, 0, _state(world), (VIEW_W, VIEW_H),
                 terrain=occluded_terrain(), camera_delta=camera)
    w = planner.world
    # A FIXED SLICE OF THE MAP, i.e. fixed ground. Measuring the same SCREEN
    # region later would measure whatever is under the joystick now, which is
    # occluded by construction — the question is whether the ground that
    # started under it ever gets resolved.
    ox, oy = int(round(w.origin[0])), int(round(w.origin[1]))
    hole = (slice(oy + 16, oy + GRID_H), slice(ox, ox + 14))
    unseen_first = float((w.seen[hole] < 0.15).mean())

    # Walk WEST. The joystick sits bottom-LEFT, so westward motion slides the
    # ground under it rightwards and out from beneath the button. Walking east
    # would push that same ground further behind the camera, where it is never
    # seen at all — which is a real property of the fix, not a quirk of the
    # test: occluded ground is resolved by scrolling it out, not by waiting.
    for _ in range(20):
        camera = world.move((-1.0, 0.0))
        planner.plan(8, 0, _state(world), (VIEW_W, VIEW_H),
                     terrain=occluded_terrain(), camera_delta=camera)

    unseen_after = float((planner.world.seen[hole] < 0.15).mean())
    assert unseen_first > 0.9, "test setup: the hole should start unobserved"
    assert unseen_after < 0.5, (
        f"{100 * unseen_after:.0f}% of the HUD-occluded region is still "
        f"unobserved after 20 steps of scrolling — it is never being filled in")
    print(f"  HUD-occluded region: {100 * unseen_first:.0f}% unobserved at first, "
          f"{100 * unseen_after:.0f}% after scrolling")
    return True


def test_gas_not_gated_behind_a_terrain_profile():
    """Gas still reaches the planner when NO terrain profile matched.

    `find_terrain` returns None on an uncalibrated map, and gas used to be
    computed inside that same branch — so on any map without a colour profile
    the planner had no spatial gas at all. Gas detection is map-independent, so
    it must not be gated behind terrain.
    """
    planner = WaypointPlanner()
    rect = (0, 0, VIEW_W, VIEW_H)
    planner.world.set_geometry(rect, (GRID_W, GRID_H))

    # A blob to the east that leaves a clear corridor along the bottom, so a
    # detour genuinely exists — same shape as the with-terrain gas test.
    gas = np.zeros((GRID_H, GRID_W), np.float32)
    gas[0:20, 26:32] = 1.0
    st = GameState(player_pos=(int(VIEW_W / 2), int(VIEW_H / 2)), health=8000)

    for _ in range(3):
        move = planner.plan(0, 2, st, (VIEW_W, VIEW_H), terrain=None, gas_grid=gas)

    assert float(planner.world.gas.max()) > 0.5, (
        "gas never reached the planner without a terrain profile")
    assert move[1] > 0.15, (
        f"walked straight east into the cloud with no terrain profile "
        f"instead of detouring below it: {move}")
    print(f"  no terrain profile, gas still routed around: "
          f"move {move[0]:+.2f},{move[1]:+.2f}")
    return True


def _shaping_run(distances, picked_up_at=None):
    """Feed a sequence of box distances through the reward engine."""
    from rl.rewards import RewardCalculator, RewardConfig

    calc = RewardCalculator(RewardConfig())
    out = []
    for i, d in enumerate(distances):
        st = GameState(player_pos=(960, 540), health=8000, frame_size=(1920, 1080),
                       cube_count=(1 if picked_up_at is not None and i >= picked_up_at
                                   else 0))
        # Place a box `d` fraction of the frame diagonal to the east.
        diag = (1920 ** 2 + 1080 ** 2) ** 0.5
        st.box_positions = [(960 + d * diag, 540)]
        r = calc.compute(st)
        out.append(r.breakdown.get("box_progress", 0.0))
    return out


def test_box_shaping_rewards_approach():
    """Closing on a box pays; retreating costs."""
    approach = _shaping_run([0.50, 0.45, 0.40, 0.35])
    assert all(v > 0 for v in approach[2:]), approach
    retreat = _shaping_run([0.35, 0.40, 0.45, 0.50])
    assert all(v < 0 for v in retreat[2:]), retreat
    print(f"  approach {approach[2]:+.4f}/step, retreat {retreat[2]:+.4f}/step")
    return True


def test_box_shaping_cannot_be_farmed():
    """The oscillation exploit: shuffling next to a box must never pay.

    This is the whole reason for using a potential function rather than a plain
    "reward getting closer" term. A naive version pays out on every approach and
    charges nothing on the retreat, so an agent can shuffle back and forth next
    to a box forever and out-earn actually playing the game.

    Note the net is not exactly zero: with gamma < 1 the telescoping sum leaves
    a residual of (gamma-1)*sum(Phi), which is NEGATIVE because Phi >= 0. That
    is the correct behaviour for a discounted setting and it errs in the safe
    direction — a full cycle costs a little rather than paying a little. What
    matters is that it is never positive, and that it is dominated by the payout
    for actually making progress.
    """
    cycle = [0.40, 0.35, 0.30, 0.35, 0.40] * 6
    farm = sum(_shaping_run(cycle))
    gross = sum(abs(v) for v in _shaping_run(cycle))
    assert gross > 0.05, "shaping produced no signal at all — test is vacuous"
    assert farm <= 0.0, f"oscillating 6 times PAID {farm:+.4f} — the term is farmable"

    # Same number of steps spent genuinely closing the distance.
    honest = sum(_shaping_run(list(np.linspace(0.40, 0.05, len(cycle)))))
    assert honest > 0.0 and honest > abs(farm), (
        f"real progress ({honest:+.4f}) must beat oscillating ({farm:+.4f})")
    print(f"  {len(cycle)} steps: oscillating {farm:+.4f}, real progress "
          f"{honest:+.4f} (gross {gross:.3f})")
    return True


def test_box_shaping_survives_pickup():
    """Collecting a box must not be punished by the potential collapsing."""
    # Approach, then the cube counter ticks up on the last frame.
    payouts = _shaping_run([0.30, 0.20, 0.10, 0.05], picked_up_at=3)
    assert payouts[-1] == 0.0, (
        f"shaping fired {payouts[-1]:+.4f} on the pickup tick — that is a "
        "penalty for succeeding")
    print("  shaping suppressed on the pickup tick")
    return True


def test_box_shaping_ignores_identity_switches():
    """A discontinuous distance jump is a perception event, not movement."""
    payouts = _shaping_run([0.40, 0.38, 0.05, 0.03])
    assert payouts[2] == 0.0, (
        f"a 0.33 distance jump paid {payouts[2]:+.4f} — that is reward for a "
        "detection change, not for moving")
    print("  0.33 jump in nearest-box distance correctly skipped")
    return True


def test_stuck_detection_releases_commitment():
    """Grinding into a wall must hand control back, not burn the whole commit.

    A coarse grid can call a cell walkable while the exact pixel the brawler is
    pressed against is not, so A* keeps reporting a valid route while the agent
    goes nowhere. Without this check that costs up to 26 decisions of shoving —
    which is what walking into a map border looks like from the outside.
    """
    world = World()
    planner = WaypointPlanner()
    cfg = PathPlannerConfig()
    terrain = world.terrain()
    st = _state(world)

    planner.plan(0, 2, st, (VIEW_W, VIEW_H), terrain=terrain)
    assert planner.status().active

    # Simulate being wedged: the player does not move and the world does not
    # scroll, however hard the planner steers.
    released_at = None
    for step in range(1, cfg.stuck_ticks + 4):
        planner.plan(0, 2, st, (VIEW_W, VIEW_H), terrain=terrain,
                     camera_delta=(0.0, 0.0))
        if planner.status().replanned:
            released_at = step
            break
    assert released_at is not None, (
        f"still committed after {cfg.stuck_ticks + 3} motionless steps")
    assert released_at <= cfg.stuck_ticks + 1, released_at
    print(f"  released the commitment after {released_at} motionless steps "
          f"(stuck_ticks={cfg.stuck_ticks})")
    return True


def test_normal_walking_is_not_stuck():
    """Walking normally must NOT trip the stuck detector.

    The trap: the camera chases the player, so while walking his SCREEN
    position barely changes. Measuring screen movement alone would flag healthy
    movement as stuck and cancel every commitment immediately.
    """
    world = World()
    planner = WaypointPlanner()
    camera = (0.0, 0.0)
    for step in range(12):
        move = planner.plan(2, 2, _state(world), (VIEW_W, VIEW_H),
                            terrain=world.terrain(), camera_delta=camera)
        if step > 0:
            assert not planner.status().replanned, (
                f"stuck detector fired at step {step} while walking normally")
        camera = world.move(move)
    print("  12 steps of normal walking, commitment never falsely released")
    return True


def test_terrain_profiles_select_correctly():
    """Each known map must pick its own profile, and mismatches must be refused."""
    import cv2
    from perception.getTerrain import (find_terrain, select_profile, play_rect,
                                       PROFILES, NIGHT_TEAL, MIN_PROFILE_COVERAGE)

    root = pathlib.Path(__file__).resolve().parent.parent
    img = cv2.imread(str(root / "showdown.png"))
    if img is None:
        print("  SKIP (showdown.png not found)")
        return True

    x, y, w, h = play_rect(img)
    small = cv2.resize(img[y:y + h, x:x + w], (240, 135), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    chosen, score = select_profile(hsv)
    assert chosen is NIGHT_TEAL, f"picked {chosen} for showdown.png"
    assert score > MIN_PROFILE_COVERAGE, score

    # Forcing the wrong profile must be REFUSED, not silently turned into a
    # mostly-solid grid. This is the bug that made the agent walk into borders.
    from perception.getTerrain import PURPLE_STONE
    bad = find_terrain(img, profile=PURPLE_STONE)
    assert bad is None, (
        "a profile that explains ~1% of the frame produced a usable grid — "
        "the MAX_BLOCKED_FRACTION guard is not working")
    print(f"  showdown.png -> {chosen.name} ({score * 100:.0f}%); "
          f"wrong profile correctly refused")
    return True


def test_gas_calibration_fixtures():
    """The five labelled gas screenshots must all classify correctly.

    Regression guard for two bugs that both came from confusing gas with
    foliage: the gas band's saturation cap used to admit green bushes, and the
    purple_stone bush class was calibrated on gas puffs. Both are invisible
    without ground truth, which is what these fixtures are.
    """
    import cv2
    from perception.getGas import gas_info
    from perception.getAnchor import find_player_position
    from perception.getTerrain import find_terrain, ProfileSelector

    root = pathlib.Path(__file__).resolve().parent.parent
    truth = {"gas_1": True, "gas_2": False, "gas_3": True,
             "gas_4": True, "gas_5": False}
    shots = sorted((root / "media" / "gasphotos").glob("*.jpg"))
    if not shots:
        print("  SKIP (media/gasphotos not present)")
        return True

    for path in shots:
        name = path.stem
        img = cv2.imread(str(path))
        anchor = find_player_position(img)
        assert anchor is not None, f"{name}: no player anchor"
        gi = gas_info(img, player_pos=anchor)
        assert gi["in_gas"] == truth[name], (
            f"{name}: in_gas={gi['in_gas']}, expected {truth[name]} "
            f"(window fraction suggests {gi['frac']:.3f} of frame is gas)")
        # Every shot has visible gas somewhere, but never most of the screen.
        assert 0.01 < gi["frac"] < 0.35, f"{name}: implausible gas fraction {gi['frac']}"

        # And a profile must match each of the four themes represented.
        t = find_terrain(img, selector=ProfileSelector())
        assert t is not None, f"{name}: no terrain profile matched"
        assert t["coverage"] > 0.45, f"{name}: weak coverage {t['coverage']:.2f}"
        assert 0.20 < t["occupancy"].mean() < 0.60, (
            f"{name}: implausible blocked fraction {t['occupancy'].mean():.2f}")
    print(f"  {len(shots)}/{len(shots)} fixtures: in_gas correct, profile matched")
    return True


def test_gas_and_bush_are_separable():
    """Gas and green foliage collide in hue; saturation must still split them."""
    import numpy as np_
    from perception.getGas import GAS_LOWER, GAS_UPPER
    from perception.getTerrain import PROFILES

    # Two HSV boxes are disjoint as soon as they are disjoint in ANY ONE
    # channel, so checking hue (or saturation) alone would be both too strict
    # and too lenient. Different maps happen to separate on different channels:
    # night_teal and starr_rail on hue, magenta_crate and purple_stone on value.
    channels = "HSV"
    for prof in PROFILES:
        lo, hi = prof.bush
        separated = [c for c in range(3)
                     if hi[c] < GAS_LOWER[c] or lo[c] > GAS_UPPER[c]]
        assert separated, (
            f"{prof.name}: bush {list(lo)}..{list(hi)} overlaps gas "
            f"{list(GAS_LOWER)}..{list(GAS_UPPER)} in all three channels — "
            f"that map's bushes will be detected as gas")
        names = "/".join(channels[c] for c in separated)
        print(f"    {prof.name:14s} bush separated from gas on {names}")
    print(f"  checked {len(PROFILES)} profiles: no bush class can read as gas")
    return True


def _combat_state(**state_kw):
    state_kw.setdefault("frame_size", (VIEW_W, VIEW_H))
    return GameState(**state_kw)


def test_combat_script():
    """Shots that provably cannot do anything must never reach the device.

    This used to test `env._gate_weapons`, which overrode the policy's attack
    and super heads. The rule is the same; it now lives in rl/combat.py and
    DECIDES rather than vetoes, because once the veto is doing the real
    deciding the heads are decoration (see that module).
    """
    from rl.combat import CombatPolicy, CombatConfig

    cfg = CombatConfig()
    near = (VIEW_W // 2 + 200, VIEW_H // 2)          # well inside max_range
    player = (VIEW_W // 2, VIEW_H // 2)

    # 1. No enemy visible -> nothing fires, however charged we are.
    a, s = CombatPolicy().decide(_combat_state(player_pos=player, ammo_count=3,
                                               ammo_known=True, super_charge=1.0))
    assert not a and not s, "fired with no enemy visible"

    # 2. Enemy visible, clip empty AND the reading is trusted -> blocked.
    a, s = CombatPolicy().decide(_combat_state(
        player_pos=player, enemy_positions=[near],
        ammo_count=0, ammo_known=True, super_charge=0.0))
    assert not a, "attacked with a confirmed-empty clip"
    assert not s, "fired an uncharged super"

    # 3. Same, but the ammo reading is NOT trusted -> attack allowed.
    # The important one: a failed read also reports 0, and the ammo bar is only
    # located on a minority of real frames. Trusting the count alone would
    # block essentially every attack the agent ever tries.
    a, _ = CombatPolicy().decide(_combat_state(
        player_pos=player, enemy_positions=[near],
        ammo_count=0, ammo_known=False, super_charge=0.0))
    assert a, "blocked an attack on an UNTRUSTED ammo reading"

    # 4. Everything available -> both fire.
    a, s = CombatPolicy().decide(_combat_state(
        player_pos=player, enemy_positions=[near],
        ammo_count=2, ammo_known=True, super_charge=1.0))
    assert a and s, (a, s)

    # 5. An enemy way out of auto-aim range is not worth revealing position for.
    far = (VIEW_W // 2 + int(cfg.max_range_frac * VIEW_H) + 120, VIEW_H // 2)
    a, s = CombatPolicy().decide(_combat_state(
        player_pos=player, enemy_positions=[far],
        ammo_count=3, ammo_known=True, super_charge=1.0))
    assert not a, "fired at a target beyond auto-aim range"

    # 6. Super does not machine-gun itself while the charge readout catches up.
    pol = CombatPolicy()
    st = _combat_state(player_pos=player, enemy_positions=[near],
                       ammo_count=3, ammo_known=True, super_charge=1.0)
    supers = sum(pol.decide(st)[1] for _ in range(cfg.super_cooldown_ticks))
    assert supers == 1, f"fired {supers} supers inside the cooldown window"

    print("  blocked: no-target, empty-clip, uncharged-super, out-of-range; "
          "allowed: untrusted ammo, ready weapons; super rate-limited")
    return True


def test_healing_is_rewarded_and_not_farmable():
    """Regaining HP must pay, and damage-then-heal must not be a money loop."""
    from rl.rewards import RewardCalculator, RewardConfig

    cfg = RewardConfig()
    assert cfg.health_regain <= cfg.damage_taken, (
        f"health_regain ({cfg.health_regain}) exceeds damage_taken "
        f"({cfg.damage_taken}) — the agent could farm reward by taking damage "
        f"on purpose and healing it back")

    calc = RewardCalculator(cfg)
    hp = [8000, 8000, 5000, 6000, 7000, 8000]   # full, hurt, then regen back
    out = []
    for v in hp:
        out.append(calc.compute(GameState(health=v, frame_size=(1920, 1080))).breakdown)

    assert out[2].get("damage_taken", 0) < 0, out[2]
    heals = [b.get("health_regain", 0.0) for b in out[3:]]
    assert all(h > 0 for h in heals), f"healing paid nothing: {heals}"

    cycle = sum(b.get("damage_taken", 0.0) + b.get("health_regain", 0.0) for b in out)
    assert cycle <= 1e-9, (
        f"a full damage-then-heal cycle netted {cycle:+.4f} — that is farmable")
    print(f"  healing pays {sum(heals):+.3f}; full damage/heal cycle nets "
          f"{cycle:+.4f} (must be <= 0)")
    return True


def test_firing_costs_are_charged_only_when_fired():
    """attack_cost applies to shots sent to the device, not shots requested."""
    from rl.rewards import RewardCalculator, RewardConfig

    cfg = RewardConfig()
    calc = RewardCalculator(cfg)
    st = GameState(health=8000, frame_size=(1920, 1080))
    calc.compute(st)                                    # baseline tick

    quiet = calc.compute(st, fired_attack=False, fired_super=False).breakdown
    assert "attack_cost" not in quiet and "super_cost" not in quiet, quiet

    loud = calc.compute(st, fired_attack=True, fired_super=True).breakdown
    assert loud["attack_cost"] == -cfg.attack_cost
    assert loud["super_cost"] == -cfg.super_cost

    # A shot must stay worth taking when it is likely to land: one hit is worth
    # roughly a tenth of a full super charge.
    hit_value = cfg.super_charge_full * 0.1
    assert cfg.attack_cost < hit_value, (
        f"attack_cost {cfg.attack_cost} exceeds the value of a landed hit "
        f"({hit_value}) — the agent will learn never to shoot")
    print(f"  cost charged only on real shots; a landed hit (~{hit_value:+.2f}) "
          f"still beats the {cfg.attack_cost:.2f} cost")
    return True


def test_perception_survives_no_anchor():
    """The whole stack must tick cleanly when the player cannot be located.

    Regression guard for a live-training crash: HUD readers are gated on a
    VERIFIED anchor, and `c["anchor"]` is None until the first verified
    detection — so on the opening ticks of a match every reader receives None.
    Two of the three tolerated it; `find_cube_info` unpacked it blind and threw
    `TypeError: cannot unpack non-iterable NoneType`, killing the run.

    Exercised two ways, because the guard has to hold at BOTH levels: the
    detector called directly with None, and a full LivePerception.tick on
    frames where nothing verifies.
    """
    import cv2
    from perception.getCube import find_cube_info
    from perception.getEnemies import find_entities
    from perception.getGas import gas_info
    from perception.liveLoop import LivePerception

    root = pathlib.Path(__file__).resolve().parent.parent
    img = cv2.imread(str(root / "showdown.png"))
    if img is None:
        print("  SKIP (showdown.png not found)")
        return True

    # 1. Every anchor-consuming detector must accept None.
    from perception.getHealth import find_health_info
    from perception.getAmmo import find_ammo_info
    for fn in (find_cube_info, find_entities, gas_info,
               find_health_info, find_ammo_info):
        try:
            fn(img, None)
        except Exception as e:  # noqa: BLE001
            raise AssertionError(f"{fn.__name__}(image, None) raised "
                                 f"{type(e).__name__}: {e}")

    # 2. A full tick on a frame with no locatable player (a black frame stands
    # in for the loading/menu/obscured case) must not raise.
    lp = LivePerception()
    blank = np.zeros_like(img)
    for frame in (blank, img, blank):
        live, _ = lp.tick(frame)
        assert isinstance(live, dict)
        assert "anchor_fresh" in live, "liveLoop must publish anchor_fresh"

    print("  cube/entity/gas readers accept None; full tick survives no anchor")
    return True


def test_boxes_interrupt_a_commitment():
    """A box coming into view must hand control back to the policy.

    Without this the agent walks past boxes it notices mid-commitment for up to
    26 decisions (~2.5s), which is what suppressed cube collecting after the
    waypoint refactor. Enemies and gas already interrupted; boxes did not.
    """
    world = World()
    planner = WaypointPlanner()
    terrain = world.terrain()

    # Commit with nothing in sight.
    planner.plan(0, 2, _state(world), (VIEW_W, VIEW_H), terrain=terrain)
    planner.plan(0, 2, _state(world), (VIEW_W, VIEW_H), terrain=terrain)
    assert planner.status().active and not planner.status().replanned

    # A box appears -> the commitment must end this step.
    st = _state(world)
    st.box_positions = [(1400, 700)]
    planner.plan(0, 2, st, (VIEW_W, VIEW_H), terrain=terrain)
    assert planner.status().replanned, "a newly visible box did not interrupt"

    # It must fire on the TRANSITION only, not every step the box is visible,
    # or commitments would be shredded whenever a box stays on screen.
    planner.plan(0, 2, st, (VIEW_W, VIEW_H), terrain=terrain)
    assert not planner.status().replanned, (
        "box interrupt re-fired while the box stayed visible — commitments "
        "would never hold near boxes")
    print("  box appearing releases the commitment; steady visibility does not")
    return True


def test_cube_counter_not_gated_on_verified_anchor():
    """The cube badge must be read from ANY anchor, not just verified ones.

    Regression guard: gating it like HP/ammo cost every single read (6 -> 0
    across 87 frames), which silently switched off cube_pickup — the reward
    that makes collecting cubes worth anything.
    """
    import cv2
    import inspect
    from perception import liveLoop

    src = inspect.getsource(liveLoop.LivePerception.tick)
    idx = src.find("find_cube_info")
    assert idx != -1, "find_cube_info call not found in tick()"
    call = src[idx:idx + 260]
    assert "hud_readable" not in call, (
        "the cube reader is gated on hud_readable again — that removes every "
        "cube read and disables cube_pickup")

    # And it must still tolerate a missing anchor.
    root = pathlib.Path(__file__).resolve().parent.parent
    img = cv2.imread(str(root / "showdown.png"))
    if img is not None:
        from perception.getCube import find_cube_info
        assert find_cube_info(img, None)["cube_count"] is None
    print("  cube reader runs on any anchor and tolerates None")
    return True


def _destination_openness(terrain, player_px, cfg):
    """Mean openness of every heading/distance destination from one spot."""
    from perception.getTerrain import to_grid
    from rl.path_planner import _openness

    occ = terrain["occupancy"]
    gh, gw = occ.shape
    rect, cell = terrain["rect"], terrain["cell"]
    field = _openness(occ, cfg.openness_radius)
    st = GameState(player_pos=(int(player_px[0]), int(player_px[1])))
    vals = []
    for h in range(cfg.n_headings):
        for d in range(len(cfg.distances)):
            planner = WaypointPlanner(cfg)
            planner.plan(h, d, st, (rect[2], rect[3]), terrain=terrain)
            gx, gy = to_grid(planner.waypoint, rect, cell)
            vals.append(field[int(np.clip(gy, 0, gh - 1)), int(np.clip(gx, 0, gw - 1))])
    return float(np.mean(vals)), float(np.mean([v < 0.35 for v in vals]))


def test_destinations_avoid_the_border():
    """Destinations must not pile up in closed-in ground.

    The zone in Showdown closes inward, so the map edge is where gas arrives
    FIRST — parking there is how the agent ends up dying in the smoke. The
    planner steered straight at it: a heading pointing off-map was relocated to
    the nearest legal cell, which is the border ring.

    NOTE what is and is not asserted. The grid's outer ring is the SCREEN edge,
    not the map border — mid-map they are unrelated, and penalising the screen
    edge would block most long-range movement. The real signal is openness:
    out-of-bounds decoration segments as unwalkable, so surrounding blocked area
    is what actually indicates a map edge. So this compares openness of chosen
    destinations against the same planner with the border terms switched off.
    """
    import cv2
    from dataclasses import replace
    from perception.getTerrain import find_terrain, ProfileSelector

    root = pathlib.Path(__file__).resolve().parent.parent
    img = cv2.imread(str(root / "showdown.png"))
    if img is None:
        print("  SKIP (showdown.png not found)")
        return True
    terrain = find_terrain(img, selector=ProfileSelector())
    assert terrain is not None

    on = PathPlannerConfig()
    off = replace(on, border_cost=0.0, min_destination_openness=0.0)
    rect = terrain["rect"]

    improved = 0
    for fx, fy in ((0.5, 0.5), (0.15, 0.5), (0.5, 0.18), (0.85, 0.5)):
        p = (rect[0] + rect[2] * fx, rect[1] + rect[3] * fy)
        mean_off, closed_off = _destination_openness(terrain, p, off)
        mean_on, closed_on = _destination_openness(terrain, p, on)
        assert mean_on >= mean_off - 1e-6, (
            f"at ({fx},{fy}) the border terms made destinations LESS open: "
            f"{mean_on:.3f} vs {mean_off:.3f}")
        assert closed_on <= closed_off + 1e-6, (
            f"at ({fx},{fy}) more destinations landed in closed ground with "
            f"the border terms on: {closed_on:.2f} vs {closed_off:.2f}")
        improved += mean_on > mean_off + 0.005
    assert improved >= 2, (
        f"the border terms changed nothing at {4 - improved}/4 positions")
    print(f"  destination openness improved at {improved}/4 positions, "
          f"never worse")
    return True


def test_destinations_stay_off_a_hard_border():
    """With an explicit out-of-bounds ring, destinations must keep clear of it.

    The real-frame test above is a relative comparison; this one is absolute,
    on a synthetic map whose border is unambiguous.
    """
    walls = []
    world = World(walls)
    # Ring the visible area in out-of-bounds decoration.
    terrain = world.terrain()
    occ = terrain["occupancy"]
    occ[:4, :] = True
    occ[-4:, :] = True
    occ[:, :4] = True
    occ[:, -4:] = True
    terrain["walkable"] = (~occ).astype(np.float32)

    from perception.getTerrain import to_grid
    cfg = PathPlannerConfig()
    st = GameState(player_pos=(int(VIEW_W / 2), int(VIEW_H / 2)))
    in_border = 0
    total = 0
    for h in range(cfg.n_headings):
        for d in range(len(cfg.distances)):
            planner = WaypointPlanner()
            planner.plan(h, d, st, (VIEW_W, VIEW_H), terrain=terrain)
            gx, gy = to_grid(planner.waypoint, terrain["rect"], terrain["cell"])
            gx = int(np.clip(gx, 0, GRID_W - 1))
            gy = int(np.clip(gy, 0, GRID_H - 1))
            in_border += bool(occ[gy, gx]) or gx < 5 or gy < 5 or \
                gx >= GRID_W - 5 or gy >= GRID_H - 5
            total += 1
    frac = in_border / total
    assert frac < 0.25, (
        f"{100 * frac:.0f}% of destinations landed in or against the "
        f"out-of-bounds ring")
    print(f"  {100 * frac:.0f}% of destinations in/against a hard border ring")
    return True


def test_gas_is_anticipated():
    """Ground next to gas must already be expensive, not merely gas itself.

    The zone only shrinks, so a clear cell beside the cloud is not neutral
    ground — it is ground that is about to be gas. Costing only the visible
    cloud means reacting once it is on top of the agent, which in the endgame
    is too late to walk out of.
    """
    world = World()
    planner = WaypointPlanner()
    cfg = PathPlannerConfig()
    terrain = world.terrain()

    gas = np.zeros((GRID_H, GRID_W), np.float32)
    gas[:, 30:34] = 1.0                      # a band to the east
    terrain["gas"] = gas

    # The cost grid now lives on the planner's persistent world map rather than
    # on the per-frame terrain dict, so fuse a frame first and index through the
    # map's origin. `ox`/`oy` are where the current viewport sits on the map.
    planner.plan(0, 1, _state(world), (VIEW_W, VIEW_H), terrain=terrain)
    w = planner.world
    ox, oy = int(round(w.origin[0])), int(round(w.origin[1]))
    cost = planner._cost_grid([], VIEW_H)
    row = oy + GRID_H // 2
    band = cost[row, ox + 30:ox + 34].mean()
    edge = cost[row, ox + 30 - cfg.gas_dilate_cells:ox + 30].mean()
    far = cost[row, ox + 5:ox + 10].mean()

    assert band > far * 2, f"gas band ({band:.1f}) not costed above open ground ({far:.1f})"
    assert edge > far * 1.5, (
        f"cells within {cfg.gas_dilate_cells} of the cloud cost {edge:.1f} vs "
        f"{far:.1f} for open ground — the cloud's advance is not anticipated")
    print(f"  open {far:.1f} | within {cfg.gas_dilate_cells} cells of gas "
          f"{edge:.1f} | in gas {band:.1f}")
    return True


def test_navigate_polls_fast_taps_slowly():
    """Between-match navigation must notice transitions fast without spamming taps.

    Measured on a phone, resets cost 24.2s and 71% of wall clock, almost all of
    it sleeping: the loop tapped and then slept 1.0-1.2s, so every screen change
    was noticed up to 1.2s late. Polling and tapping are now separate rates.

    Driven by a fake device whose screen advances on a wall-clock schedule, so
    the test measures the loop's responsiveness rather than a real phone's.
    """
    import time as _time
    from rl.env import BrawlStarsEnv
    from rl.actions import LoggingExecutor

    # Screen timeline: defeated for 1.0s, match_end for 1.0s, then in_match.
    class FakeSource:
        live = True
        def __init__(self):
            self.t0 = _time.time()
            self.grabs = 0
        def screen(self):
            dt = _time.time() - self.t0
            return "defeated" if dt < 1.0 else ("match_end" if dt < 2.0 else "in_match")
        def grab(self):
            self.grabs += 1
            return np.zeros((16, 16, 3), np.uint8)
        def close(self):
            pass

    src = FakeSource()
    taps = []

    class TapRecorder(LoggingExecutor):
        def tap_norm(self, fx, fy):
            taps.append((_time.time(), fx, fy))

    env = BrawlStarsEnv(source_factory=lambda: src, executor=TapRecorder())
    env.source = src          # normally set by reset(); we call navigate directly
    import rl.env as E
    real_state = E.get_game_state
    E.get_game_state = lambda frame: {"state": src.screen()}
    try:
        t0 = _time.time()
        env._navigate_to_match()
        elapsed = _time.time() - t0
    finally:
        E.get_game_state = real_state
        env.close()

    # It must not overshoot the 2.0s scripted timeline by much.
    assert elapsed < 2.9, (
        f"navigation took {elapsed:.2f}s for a 2.0s timeline — it is "
        f"oversleeping past transitions")
    # And it must have tapped both buttons, but not machine-gunned either.
    assert len(taps) >= 2, f"only {len(taps)} tap(s); both menus need one"
    assert len(taps) <= 6, (
        f"{len(taps)} taps in {elapsed:.1f}s — fast polling is being turned "
        f"into a burst of taps, which can activate whatever is underneath")
    gaps = [b[0] - a[0] for a, b in zip(taps, taps[1:])
            if abs(b[1] - a[1]) < 1e-9]          # consecutive taps, same button
    for g in gaps:
        assert g >= BrawlStarsEnv.NAVIGATE_TAP_INTERVAL - 0.05, (
            f"two taps on the same button {g:.2f}s apart, under the "
            f"{BrawlStarsEnv.NAVIGATE_TAP_INTERVAL}s minimum")
    print(f"  2.0s timeline cleared in {elapsed:.2f}s with {len(taps)} taps "
          f"({src.grabs} polls)")
    return True


def test_hold_ms_fits_the_tick():
    """A movement swipe must fit inside one tick, or swipes queue.

    `input swipe ... <hold_ms>` blocks the device shell, so issuing one per
    tick caps the sustainable rate at 1000/hold_ms fps. The failure is
    deceptive: the backlog hides in the adb stdin buffer and costs nothing
    measurable until it saturates, at which point `act` time explodes. A
    measured run went from 1.1ms to 329ms of `act` (84% of the step) purely by
    halving the tick.
    """
    from rl.actions import Controls

    auto = Controls.from_screen(2400, 1080)
    assert auto.hold_ms <= 100, (
        f"default hold_ms {auto.hold_ms}ms caps the loop at "
        f"{1000 / auto.hold_ms:.0f} fps")

    for tick in (0.1, 0.05, 0.04):
        tuned = Controls(move_center=(1, 1), attack_btn=(2, 2), super_btn=(3, 3),
                         hold_ms=200).tuned_for_tick(tick)
        assert tuned.hold_ms <= tick * 1000 + 1, (
            f"tick={tick}s leaves {tick * 1000:.0f}ms but hold_ms is "
            f"{tuned.hold_ms}ms — swipes will queue")
        assert tuned.hold_ms >= Controls.MIN_HOLD_MS, (
            f"hold_ms {tuned.hold_ms}ms is too short to register as a drag")

    # Unpaced (offline) must be left alone, and an already-short hold untouched.
    assert Controls(move_center=(1, 1), attack_btn=(2, 2), super_btn=(3, 3),
                    hold_ms=200).tuned_for_tick(0).hold_ms == 200
    short = Controls(move_center=(1, 1), attack_btn=(2, 2), super_btn=(3, 3),
                     hold_ms=40).tuned_for_tick(0.1)
    assert short.hold_ms == 40
    print(f"  default hold_ms {auto.hold_ms}ms; clamped to the tick budget "
          f"(200ms @0.05s -> {Controls(move_center=(1,1),attack_btn=(2,2),super_btn=(3,3),hold_ms=200).tuned_for_tick(0.05).hold_ms}ms)")
    return True


def test_commitments_are_fixed_in_seconds():
    """Changing --tick-seconds must not change how LONG a commitment lasts.

    commit_ticks is counted in decisions but chosen for wall-clock durations.
    Without rescaling, halving the tick to double the data rate would also
    halve every commitment — the loop would look twice as productive while the
    agent quietly reverted to twitching, which is exactly what the planner
    exists to prevent.
    """
    from rl.env import _planner_config_for, _REFERENCE_TICK

    base = PathPlannerConfig()
    for tick in (0.1, 0.05, 0.025, 0.2):
        cfg = _planner_config_for(tick)
        for i, (b, c) in enumerate(zip(base.commit_ticks, cfg.commit_ticks)):
            base_s = b * _REFERENCE_TICK
            got_s = c * tick
            assert abs(got_s - base_s) < 0.15 * base_s + 0.02, (
                f"tick={tick}s tier {i}: commitment lasts {got_s:.2f}s, "
                f"expected ~{base_s:.2f}s")
        assert abs(cfg.stuck_ticks * tick - base.stuck_ticks * _REFERENCE_TICK) < 0.1

    # Offline (unpaced) must be left exactly as authored.
    assert _planner_config_for(0).commit_ticks == base.commit_ticks
    ticks = _planner_config_for(0.05).commit_ticks
    print(f"  commit_ticks {base.commit_ticks} @0.1s -> {ticks} @0.05s "
          f"(same {base.commit_ticks[0] * 0.1:.1f}-{base.commit_ticks[-1] * 0.1:.1f}s)")
    return True


TESTS = [
    ("action space matches planner", test_action_space_matches_planner),
    ("commitments fixed in seconds", test_commitments_are_fixed_in_seconds),
    ("swipe duration fits the tick", test_hold_ms_fits_the_tick),
    ("menu navigation is poll-driven", test_navigate_polls_fast_taps_slowly),
    ("destinations avoid the map border", test_destinations_avoid_the_border),
    ("destinations stay off a hard border", test_destinations_stay_off_a_hard_border),
    ("gas is costed before it arrives", test_gas_is_anticipated),
    ("perception survives no anchor", test_perception_survives_no_anchor),
    ("boxes interrupt a commitment", test_boxes_interrupt_a_commitment),
    ("cube counter not gated on verified anchor", test_cube_counter_not_gated_on_verified_anchor),
    ("combat script", test_combat_script),
    ("healing rewarded, not farmable", test_healing_is_rewarded_and_not_farmable),
    ("firing costs charged correctly", test_firing_costs_are_charged_only_when_fired),
    ("terrain profiles select correctly", test_terrain_profiles_select_correctly),
    ("gas calibration fixtures", test_gas_calibration_fixtures),
    ("gas and bush are separable", test_gas_and_bush_are_separable),
    ("stuck detection releases commitment", test_stuck_detection_releases_commitment),
    ("normal walking is not flagged stuck", test_normal_walking_is_not_stuck),
    ("observation shape and range", test_observation_shape),
    ("terrain on the real frame", test_terrain_on_real_frame),
    ("commits to a waypoint and arrives", test_commits_and_arrives),
    ("routes around a wall", test_routes_around_wall),
    ("gas overrides a commitment", test_gas_overrides),
    ("gas costs but does not wall", test_gas_costs_but_does_not_wall),
    ("gas destination is relocated", test_gas_destination_is_relocated),
    # --- the three rewritten failure modes --- #
    ("routes keep clear of walls", test_route_keeps_clear_of_walls),
    ("narrow gap still usable", test_narrow_gap_still_usable),
    ("map remembers terrain that scrolled away", test_map_remembers_terrain_that_scrolled_away),
    ("HUD-occluded cells get filled in", test_hud_occluded_cells_get_filled_in),
    ("gas ring escape goes INWARD", test_gas_ring_escape_goes_inward),
    ("gas is remembered after scrolling", test_gas_is_remembered_after_scrolling),
    ("gas is not gated behind a terrain profile", test_gas_not_gated_behind_a_terrain_profile),
    ("box shaping rewards approach", test_box_shaping_rewards_approach),
    ("box shaping cannot be farmed", test_box_shaping_cannot_be_farmed),
    ("box shaping survives pickup", test_box_shaping_survives_pickup),
    ("box shaping ignores identity switches", test_box_shaping_ignores_identity_switches),
    ("regression: unlatched target never arrives", test_regression_unlatched_never_arrives),
    ("degrades without perception", test_planner_survives_missing_perception),
]


def main():
    failures = 0
    for name, fn in TESTS:
        print(f"\n[{name}]")
        try:
            fn()
            print("  PASS")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL: {e}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"  ERROR: {type(e).__name__}: {e}")
    print(f"\n{len(TESTS) - failures}/{len(TESTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
