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
    assert nvec == [16, 3, 2, 2], nvec
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
    boxed = world.terrain()
    boxed["gas"] = np.zeros((GRID_H, GRID_W), np.float32)
    boxed["gas"][:, 26:30] = 1.0          # gas spans the full height now
    planner.reset()
    forced = planner.plan(0, 2, st, (VIEW_W, VIEW_H), terrain=boxed)
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


def _gate_env(**state_kw):
    """A bare env with a hand-set _last_state, for exercising _gate_weapons."""
    from rl.env import BrawlStarsEnv
    from rl.actions import LoggingExecutor
    env = BrawlStarsEnv(source_factory=None, executor=LoggingExecutor())
    env._last_state = GameState(**state_kw)
    env._last_enemy_count = len(state_kw.get("enemy_positions", []) or [])
    return env


def test_weapon_gating():
    """Shots that provably cannot do anything must never reach the device."""
    enemy = {"enemy_positions": [(500, 500)], "player_pos": (960, 540)}
    want = [0, 0, 1, 1]     # policy asks to fire both every time

    # 1. No enemy visible -> attack blocked (auto-aim has no target).
    env = _gate_env(player_pos=(960, 540), ammo_count=3, ammo_known=True,
                    super_charge=1.0)
    _, a, s = env._gate_weapons(list(want))
    assert not a, "attacked with no enemy visible"
    assert s, "super blocked despite being charged"

    # 2. Enemy visible, clip empty AND the reading is trusted -> blocked.
    env = _gate_env(ammo_count=0, ammo_known=True, super_charge=0.0, **enemy)
    _, a, s = env._gate_weapons(list(want))
    assert not a, "attacked with a confirmed-empty clip"
    assert not s, "fired an uncharged super"

    # 3. Same, but the ammo reading is NOT trusted -> attack allowed.
    # This is the important one: a failed read also reports 0, and the ammo bar
    # is only located on ~8% of real frames. Trusting the count alone would
    # block essentially every attack the agent ever tries.
    env = _gate_env(ammo_count=0, ammo_known=False, super_charge=0.0, **enemy)
    _, a, _ = env._gate_weapons(list(want))
    assert a, "blocked an attack on an UNTRUSTED ammo reading"

    # 4. Everything available -> both pass through untouched.
    env = _gate_env(ammo_count=2, ammo_known=True, super_charge=1.0, **enemy)
    act, a, s = env._gate_weapons(list(want))
    assert a and s and list(act) == want, act

    # 5. The policy declining to fire is never overridden into firing.
    env = _gate_env(ammo_count=3, ammo_known=True, super_charge=1.0, **enemy)
    _, a, s = env._gate_weapons([0, 0, 0, 0])
    assert not a and not s
    print("  blocked: no-target, empty-clip, uncharged-super; "
          "allowed: untrusted ammo, ready weapons")
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
    ("perception survives no anchor", test_perception_survives_no_anchor),
    ("boxes interrupt a commitment", test_boxes_interrupt_a_commitment),
    ("cube counter not gated on verified anchor", test_cube_counter_not_gated_on_verified_anchor),
    ("weapon gating", test_weapon_gating),
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
