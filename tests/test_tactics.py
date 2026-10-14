"""The decision layer added on top of the planner: tracker, combat upgrades,
tactical brain, goal-driven planning, and the bugs each of them was written to
fix. Every test names the failure it guards against, because these are exactly
the regressions that look like "the bot is just bad" from the outside.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rl.state import EntityObs, GameState, adapt_live_state

from test_planner import CELL, GRID_H, GRID_W, VIEW_H, VIEW_W, World

CENTRE = (VIEW_W // 2, VIEW_H // 2)


# --------------------------------------------------------------------------- #
# GameState: the numbers perception reads are no longer dropped at the seam
# --------------------------------------------------------------------------- #
def test_adapter_keeps_enemy_health_and_cubes():
    live = {"enemies": [{"center": (100, 200), "health": {"current_health": 900},
                         "cubes": {"cube_count": 4}},
                        {"center": None}],
            "boxes": [{"center": (5, 6), "health": {"current_health": 4000}}]}
    s = adapt_live_state(live)
    assert s.enemy_positions == [(100, 200)]
    assert s.enemies == [EntityObs((100, 200), 900, 4)], "HP/cubes were dropped"
    assert s.boxes[0].hp == 4000


def test_enemy_list_falls_back_to_positions():
    s = GameState(enemy_positions=[(1, 2), (3, 4)])
    assert [e.pos for e in s.enemy_list()] == [(1, 2), (3, 4)]


# --------------------------------------------------------------------------- #
# EnemyTracker
# --------------------------------------------------------------------------- #
def _tick(tracker, enemies=(), boxes=(), cam=(0.0, 0.0), fresh=True):
    st = GameState(player_pos=CENTRE, frame_size=(VIEW_W, VIEW_H),
                   enemy_positions=[e[0] for e in enemies],
                   enemies=[EntityObs(*e) for e in enemies],
                   box_positions=list(boxes), boxes=[EntityObs(b) for b in boxes],
                   entities_fresh=fresh)
    tracker.update(st, cam)
    return st


def test_tracker_stationary_enemy_has_no_velocity_while_camera_scrolls():
    """Screen position changes every tick the PLAYER moves; world position
    must not, or every lead is the player's own motion negated."""
    from rl.enemy_tracker import EnemyTracker
    t = EnemyTracker()
    for k in range(12):
        _tick(t, enemies=[((1400 - 20 * k, 400), 3000, 0)], cam=(-20.0 if k else 0.0, 0.0))
    (tr,) = t.enemies()
    assert math.hypot(*tr.vel) < 1e-6, tr.vel


def test_tracker_measures_real_motion():
    from rl.enemy_tracker import EnemyTracker
    t = EnemyTracker()
    for k in range(10):
        _tick(t, enemies=[((1000 + 6 * k, 400), 3000, 0)])
    assert abs(t.enemies()[0].vel[0] - 6.0) < 0.5


def test_stale_entity_lists_are_not_new_observations():
    """liveLoop re-serves the last entity list between refreshes. Converted with
    a newer camera offset it looks like the enemy sliding backwards."""
    from rl.enemy_tracker import EnemyTracker
    t = EnemyTracker()
    _tick(t, enemies=[((1000, 400), 3000, 0)])
    for _ in range(5):          # player walks, the cached list is unchanged
        _tick(t, enemies=[((1000, 400), 3000, 0)], cam=(-20.0, 0.0), fresh=False)
    tr = t.enemies()[0]
    assert tr.hits == 1 and math.hypot(*tr.vel) < 1e-6


def test_enemy_in_a_bush_is_remembered_and_then_forgotten():
    from rl.enemy_tracker import EnemyTracker, TrackerConfig
    t = EnemyTracker(TrackerConfig(memory_ticks=10))
    _tick(t, enemies=[((1100, 500), 3000, 0)])
    for _ in range(5):
        _tick(t)
    assert len(t.hidden_enemies()) == 1, "vanished enemy forgotten immediately"
    for _ in range(10):
        _tick(t)
    assert not t.enemies(), "a long-gone enemy is still shaping decisions"


def test_destroyed_box_is_dropped_when_its_spot_is_seen_empty():
    from rl.enemy_tracker import EnemyTracker
    t = EnemyTracker()
    _tick(t, boxes=[(900, 500)])
    for _ in range(4):
        _tick(t)                 # on screen, fresh, not detected
    assert not t.boxes()


# --------------------------------------------------------------------------- #
# Combat
# --------------------------------------------------------------------------- #
def _cstate(enemies=(), boxes=(), **kw):
    kw.setdefault("player_pos", CENTRE)
    kw.setdefault("frame_size", (VIEW_W, VIEW_H))
    return GameState(enemy_positions=[e if isinstance(e, tuple) and len(e) == 2 and
                                      not isinstance(e[0], tuple) else e[0] for e in enemies],
                     enemies=[EntityObs(*e) if isinstance(e[0], tuple) else EntityObs(e)
                              for e in enemies],
                     box_positions=list(boxes), boxes=[EntityObs(b) for b in boxes], **kw)


def test_boxes_are_shot_when_no_enemy_is_in_range():
    """The bot used to break ZERO boxes: no enemy meant 'no target'."""
    from rl.combat import CombatPolicy
    d = CombatPolicy().decide(_cstate(boxes=[(CENTRE[0] + 150, CENTRE[1])]))
    assert d.attack and d.target_kind == "box"


def test_an_enemy_in_range_outranks_a_box():
    from rl.combat import CombatPolicy
    d = CombatPolicy().decide(_cstate(enemies=[(CENTRE[0] - 200, CENTRE[1])],
                                      boxes=[(CENTRE[0] + 100, CENTRE[1])]))
    assert d.attack and d.target_kind == "enemy"


def test_box_shooting_can_be_held_by_the_brain():
    from rl.combat import CombatHint, CombatPolicy
    d = CombatPolicy().decide(_cstate(boxes=[(CENTRE[0] + 150, CENTRE[1])]),
                              hint=CombatHint(allow_boxes=False))
    assert not d.attack


def test_low_health_target_beats_a_slightly_nearer_full_one():
    """Auto-aim picks the nearest; a 300 HP straggler a little further is a kill."""
    from rl.combat import CombatPolicy
    from rl.enemy_tracker import EnemyTracker
    t = EnemyTracker()
    near, far = (CENTRE[0] + 160, CENTRE[1]), (CENTRE[0] - 200, CENTRE[1])
    for hp_far in (4000, 4000, 300):          # far one takes a beating
        st = _cstate(enemies=[(near, 4000, 0), (far, hp_far, 0)])
        t.update(st, (0, 0))
    d = CombatPolicy().decide(st, tracker=t)
    assert d.attack and d.target == far, d
    assert d.aim is not None and d.mode == "manual", "retarget must be an aimed shot"


def _walled_world():
    from rl.world_map import WorldMap
    wm = WorldMap()
    occ = np.zeros((GRID_H, GRID_W), bool)
    occ[:, GRID_W // 2 + 2:GRID_W // 2 + 4] = True          # a 2-cell wall
    wm.update({"occupancy": occ, "bush": np.zeros(occ.shape, np.float32),
               "unknown": np.zeros(occ.shape, bool), "rect": (0, 0, VIEW_W, VIEW_H),
               "cell": (CELL, CELL)}, None)
    return wm


def test_walls_hold_fire_for_shooters_but_not_throwers():
    from brawlers.abilities import default_abilities
    from rl.combat import CombatPolicy
    book = default_abilities()
    wm = _walled_world()
    st = _cstate(enemies=[(CENTRE[0] + 6 * int(CELL), CENTRE[1])])
    colt = CombatPolicy(ability=book.get("colt")).decide(st, world=wm)
    barley = CombatPolicy(ability=book.get("barley")).decide(st, world=wm)
    assert not colt.attack and "walled" in colt.reason
    assert barley.attack, "a thrower's whole point is the wall in between"


def test_unknown_brawler_ignores_walls_exactly_as_before():
    from rl.combat import CombatPolicy
    st = _cstate(enemies=[(CENTRE[0] + 6 * int(CELL), CENTRE[1])])
    assert CombatPolicy().decide(st, world=_walled_world()).attack


def test_fire_rate_is_limited():
    from rl.combat import CombatConfig, CombatPolicy
    cp = CombatPolicy(CombatConfig(attack_interval_s=0.3, tick_seconds=0.1))
    st = _cstate(enemies=[(CENTRE[0] + 150, CENTRE[1])])
    shots = [cp.decide(st).attack for _ in range(9)]
    assert shots == [True, False, False] * 3, shots


def test_missing_anchor_does_not_silence_the_gun():
    """The camera keeps the player mid-frame; no ring is not no player."""
    from rl.combat import CombatPolicy
    st = _cstate(enemies=[(CENTRE[0] + 150, CENTRE[1])], player_pos=None)
    assert CombatPolicy().decide(st).attack


def test_an_unfed_tracker_falls_back_to_raw_detections():
    from rl.combat import CombatPolicy
    from rl.enemy_tracker import EnemyTracker
    st = _cstate(enemies=[(CENTRE[0] + 150, CENTRE[1])])
    assert CombatPolicy().decide(st, tracker=EnemyTracker()).attack


def test_aimed_shot_leads_a_moving_target():
    from brawlers.abilities import default_abilities
    from rl.combat import CombatConfig, CombatPolicy
    from rl.enemy_tracker import EnemyTracker
    t = EnemyTracker()
    for k in range(8):                       # enemy walking straight down
        st = _cstate(enemies=[((CENTRE[0] + 400, CENTRE[1] - 100 + 8 * k), 3000, 0)])
        t.update(st, (0, 0))
    piper = default_abilities().get("piper")
    led = CombatPolicy(CombatConfig(lead_gain=1.0), ability=piper).decide(st, tracker=t)
    flat = CombatPolicy(CombatConfig(lead_shots=False), ability=piper).decide(st, tracker=t)
    assert led.attack and led.aim is not None and flat.aim is not None
    assert led.aim[1] > flat.aim[1] + 0.03, (
        f"no lead toward the target's (downward) motion: {led.aim} vs {flat.aim}")


# --------------------------------------------------------------------------- #
# Planner: goal-driven planning and the map bugs found by the simulator
# --------------------------------------------------------------------------- #
def test_plan_goal_reaches_a_fixed_world_point_round_a_wall():
    from rl.path_planner import WaypointPlanner
    world = World(walls=[(208, 110, 210, 130)])        # wall east of the player
    planner = WaypointPlanner(localize=False)
    goal_world = world.player + np.array([12 * CELL, 0.0])
    cam = (0.0, 0.0)
    # "Arrived" is the planner's own radius: arrive_frac of the short side.
    arrive = planner.config.arrive_frac * VIEW_H + 0.5 * CELL
    for _ in range(140):
        st = GameState(player_pos=CENTRE, health=8000, frame_size=(VIEW_W, VIEW_H))
        move = planner.plan_goal(world.to_screen(goal_world), st, (VIEW_W, VIEW_H),
                                 terrain=world.terrain(), camera_delta=cam)
        cam = world.move(move)
    # Before the lookahead fix this wedged on the wall's corner 10 cells short.
    assert np.linalg.norm(world.player - goal_world) < arrive, (
        f"stopped {np.linalg.norm(world.player - goal_world) / CELL:.1f} cells short")


def test_waypoint_survives_a_world_map_recentre():
    """Recentring rolls the map ~20 cells; the latched waypoint used to stay
    behind in the old cell coordinates and jump half a screen."""
    from rl.path_planner import WaypointPlanner
    world = World()
    planner = WaypointPlanner(localize=False)
    st = GameState(player_pos=CENTRE, health=8000, frame_size=(VIEW_W, VIEW_H))
    planner.plan(0, 2, st, (VIEW_W, VIEW_H), terrain=world.terrain())
    wp_world = np.array(world.to_world(planner.waypoint_pixels()))
    shift0 = list(planner.world.shift)
    cam = (0.0, 0.0)
    for _ in range(60):                       # walk west: forces a recentre
        planner.plan(0, 2, st, (VIEW_W, VIEW_H), terrain=world.terrain(), camera_delta=cam)
        if planner.world.shift != shift0:
            break
        cam = world.move((-1.0, 0.0))
        planner.ticks_left = 99               # keep the same commitment
    assert planner.world.shift != shift0, "test did not trigger a recentre"
    now = np.array(world.to_world(planner.waypoint_pixels()))
    assert np.linalg.norm(now - wp_world) < CELL, (
        f"waypoint jumped {np.linalg.norm(now - wp_world) / CELL:.1f} cells on recentre")


def test_bump_sensor_marks_a_wall_perception_missed():
    from rl.path_planner import WaypointPlanner
    planner = WaypointPlanner(localize=False)
    world = World()
    terrain = world.terrain()                 # perception says: all clear
    st = GameState(player_pos=CENTRE, health=8000, frame_size=(VIEW_W, VIEW_H))
    goal = (CENTRE[0] + 8 * CELL, CENTRE[1])
    for _ in range(12):                       # ...but the player never moves
        planner.plan_goal(goal, st, (VIEW_W, VIEW_H), terrain=terrain)
    assert planner.bumps >= 1
    cx, cy = planner.world.to_cell(CENTRE)
    ahead = planner.world.occupancy[int(cy), int(cx) + 1:int(cx) + 4]
    assert ahead.any(), "the cell it kept pushing into is still free"


def test_escape_does_not_stop_on_a_sliver_against_a_wall():
    """The first gas-free cell beside the player was a sliver inside a wall
    (the player's own sprite had hidden it); the escape pushed into it until
    the cloud killed him. It must head for deep, standable safety instead."""
    from rl.path_planner import WaypointPlanner
    planner = WaypointPlanner(localize=False)
    occ = np.zeros((GRID_H, GRID_W), bool)
    px, py = GRID_W // 2, GRID_H // 2
    occ[:, px + 2:px + 6] = True                       # wall just east
    occ[py - 1:py + 2, px + 1] = False                 # ...with a sliver
    gas = np.zeros((GRID_H, GRID_W), np.float32)
    gas[:, :px + 1] = 1.0                              # cloud covers the player
    gas[:, px + 1] = 0.0                               # the sliver reads clean
    gas[:2, :] = 0.0                                   # real safety: north strip
    terrain = {"occupancy": occ, "bush": np.zeros(occ.shape, np.float32),
               "unknown": np.zeros(occ.shape, bool), "rect": (0, 0, VIEW_W, VIEW_H),
               "cell": (CELL, CELL), "gas": gas}
    st = GameState(player_pos=CENTRE, health=8000, in_gas=True,
                   frame_size=(VIEW_W, VIEW_H))
    for _ in range(3):
        move = planner.plan_goal(None, st, (VIEW_W, VIEW_H), terrain=terrain)
    assert planner.status().escaping_gas
    assert move[1] < -0.5, f"escape headed for the sliver, not the safe strip: {move}"


def test_terrain_no_longer_frees_walls_under_the_player_sprite(showdown_frame):
    from perception.getAnchor import find_player_position
    from perception.getTerrain import find_terrain
    x, y, _r = find_player_position(showdown_frame)
    t = find_terrain(showdown_frame, player_pos=(x, y))
    if t is None:
        pytest.skip("no terrain profile for the fixture")
    gx = int((x - t["rect"][0]) / t["cell"][0])
    gy = int((y - t["rect"][1]) / t["cell"][1])
    assert not t["occupancy"][gy, gx] and not t["unknown"][gy, gx]
    nb = [(gx + dx, gy + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
    nb = [(a, b) for a, b in nb if 0 <= a < t["unknown"].shape[1]
          and 0 <= b < t["unknown"].shape[0] and not t["off_map"][b, a]]
    assert all(t["unknown"][b, a] for a, b in nb), "sprite-covered cells reported as seen"


# --------------------------------------------------------------------------- #
# Tactical brain
# --------------------------------------------------------------------------- #
def _brain(brawler="shelly"):
    from brawlers.abilities import default_abilities
    from rl.combat import CombatPolicy
    from rl.enemy_tracker import EnemyTracker
    from rl.path_planner import WaypointPlanner
    from rl.tactics import TacticalBrain
    ab = default_abilities().get(brawler)
    cp = CombatPolicy(ability=ab)
    planner = WaypointPlanner(localize=False)
    world = World()
    planner.world.update(world.terrain(), None)
    return TacticalBrain(ability=ab, combat=cp), planner, EnemyTracker()


def _feed(brain, planner, tracker, **kw):
    kw.setdefault("player_pos", CENTRE)
    kw.setdefault("frame_size", (VIEW_W, VIEW_H))
    kw.setdefault("health", 4000)
    st = _cstate(**kw)
    tracker.update(st, (0, 0))
    return brain.decide(st, planner, tracker)


def test_brain_flees_the_gas_first():
    brain, planner, tracker = _brain()
    g = _feed(brain, planner, tracker, in_gas=True,
              enemies=[(CENTRE[0] + 100, CENTRE[1])])
    assert g.mode == "gas"


def test_brain_retreats_when_hurt_with_a_threat_near():
    brain, planner, tracker = _brain()
    _feed(brain, planner, tracker, health=4000)            # learn max HP
    enemy = (CENTRE[0] + 250, CENTRE[1])
    g = _feed(brain, planner, tracker, health=900, enemies=[(enemy, 4000, 0)])
    assert g.mode == "retreat"
    assert g.point[0] < CENTRE[0], "retreated TOWARD the threat"


def test_brain_breaks_boxes_from_inside_its_range():
    brain, planner, tracker = _brain("colt")
    box = (CENTRE[0] + 700, CENTRE[1])
    g = _feed(brain, planner, tracker, boxes=[box])
    assert g.mode == "box"
    reach = brain.combat.range_px((VIEW_W, VIEW_H))
    assert math.hypot(g.point[0] - box[0], g.point[1] - box[1]) <= reach


def test_brain_walks_onto_ground_cubes_precisely():
    brain, planner, tracker = _brain()
    cube = (CENTRE[0] - 300, CENTRE[1] + 100)
    g = _feed(brain, planner, tracker, ground_cube_positions=[cube])
    assert g.mode == "cube" and g.precise and g.point == cube


def test_brain_engages_a_weak_enemy_at_standoff_range():
    brain, planner, tracker = _brain("colt")
    enemy = (CENTRE[0] + 900, CENTRE[1])
    for _ in range(3):
        g = _feed(brain, planner, tracker, health=4000, enemies=[(enemy, 800, 0)])
    assert g.mode == "engage"
    d = math.hypot(g.point[0] - enemy[0], g.point[1] - enemy[1])
    reach = brain.combat.range_px((VIEW_W, VIEW_H))
    assert 0.3 * reach < d <= reach, f"standoff {d:.0f}px vs reach {reach:.0f}px"


def test_goal_to_action_inverts_target_for():
    from rl.path_planner import WaypointPlanner
    from rl.tactics import goal_to_action
    p = WaypointPlanner(localize=False)
    rect = (0, 0, VIEW_W, VIEW_H)
    for h in range(16):
        for d in range(3):
            goal = p.target_for(h, d, CENTRE, rect)
            assert goal_to_action(goal, CENTRE, rect) == (h, d)


def test_decode_goal_carries_tactic_and_a_quantised_action():
    from rl.actions import decode_goal
    from rl.path_planner import WaypointPlanner
    from rl.tactics import Goal
    world = World()
    st = GameState(player_pos=CENTRE, health=8000, frame_size=(VIEW_W, VIEW_H))
    it = decode_goal(Goal(point=(CENTRE[0] + 400, CENTRE[1]), mode="explore", reason="sweep"),
                     state=st, frame_size=(VIEW_W, VIEW_H),
                     planner=WaypointPlanner(localize=False), terrain=world.terrain())
    assert it.tactic == "explore: sweep"
    assert it.raw[0] == 0, "due east is heading 0"
    assert it.move[0] > 0.5


# --------------------------------------------------------------------------- #
# Integration
# --------------------------------------------------------------------------- #
def test_env_with_the_tactical_brain_reports_teacher_actions(showdown_frame, monkeypatch):
    import perception.liveLoop as LL
    import rl.env as E
    monkeypatch.setattr(LL, "get_game_state", lambda frame, hsv=None: {
        "state": "in_match", "brawlers_left": 8, "rank": None})
    monkeypatch.setattr(E, "navigate_to_match", lambda *a, **k: None)

    class Src:
        live = True
        def grab(self):
            return showdown_frame
        def close(self):
            pass

    env = E.BrawlStarsEnv(source_factory=Src, brain="tactical", ability="colt")
    env.reset()
    for _ in range(4):
        obs, _r, _t, _tr, info = env.step([0, 0])
    assert obs.shape == env.observation_space.shape
    assert "teacher_action" in info and len(info["teacher_action"]) == 2
    assert info["tactic"]


def test_session_config_validates_brain():
    from webui.runner import SessionConfig
    SessionConfig(brain="tactical").validate()
    with pytest.raises(ValueError):
        SessionConfig(brain="genius").validate()


def test_web_session_does_not_reset_the_map_every_second(showdown_frame, monkeypatch):
    """Each new_match() builds a fresh LivePerception that reports "unknown"
    until its smoothing confirms twice. That used to re-arm the new-match edge,
    so the world map was wiped every 1-2 seconds for the whole session."""
    import time
    import perception.liveLoop as LL
    from webui.runner import SessionConfig, SessionRunner
    monkeypatch.setattr(LL, "get_game_state", lambda frame, hsv=None: {
        "state": "in_match", "brawlers_left": 8, "rank": None})

    class Src:
        live = True
        n = 0
        def grab(self):
            Src.n += 1
            return showdown_frame if Src.n < 60 else None
        def close(self):
            pass

    r = SessionRunner()
    r._build_source = lambda cfg: Src()
    r.start(SessionConfig(brawler="colt", fps=20, brain="tactical"))
    deadline = time.time() + 20
    while r.is_running and time.time() < deadline:
        time.sleep(0.2)
    s = r.stop()
    resets = sum("new match detected" in line for line in s["log"])
    assert s["error"] is None
    assert resets <= 1, f"{resets} spurious new-match resets"


def test_sendevent_sends_one_write_per_tick_and_really_aims(monkeypatch):
    import rl.sendevent_backend as SB
    from rl.actions import Intent
    sent = []

    class Run:
        stderr = ""

    monkeypatch.setattr(SB.shutil, "which", lambda name: "/usr/bin/adb")
    monkeypatch.setattr(SB.subprocess, "run", lambda *a, **k: (sent.append(a[0][-1]), Run())[1])

    class Shell:
        def __init__(self, *a, **k):
            class In:
                def write(_, x):
                    sent.append(x.strip())
                def flush(_):
                    pass
                def close(_):
                    pass
            self.stdin = In()
        def poll(self):
            return None
        def terminate(self):
            pass

    monkeypatch.setattr(SB.subprocess, "Popen", Shell)
    ex = SB.SendeventExecutor(device="/dev/input/event2")
    ex._execute(Intent("E", (1.0, 0.0), True, True, (0, 0, 1, 1)))
    assert len(sent) == 1, "a tick must be ONE adb write, not one per event"
    ex._execute(Intent("E", (1.0, 0.0), True, False, (0, 0, 1, 0), aim=(0.0, -1.0)))
    assert len(sent) == 2
    assert ex.ATTACK_SLOT in ex._active, "aimed press must stay held for a tick"
    ex._execute(Intent("E", (0.0, 0.0), False, False, (0, 0, 0, 0)))
    assert ex.ATTACK_SLOT not in ex._active, "aimed press never released (never fired)"


def test_shipped_base_policy_loads_whatever_algorithm_wrote_it(repo_root):
    pytest.importorskip("stable_baselines3")
    from rl.checkpoints import load_policy
    model, algo = load_policy(repo_root / "brawlstars_move.zip", device="cpu")
    assert model.observation_space.shape == (60,)
    print(f"  brawlstars_move.zip loads as {algo}")


@pytest.mark.slow
def test_simulated_match_runs_and_is_deterministic():
    from sim.harness import run_match
    a = run_match("tactical", "shelly", seed=7, localize=False)
    b = run_match("tactical", "shelly", seed=7, localize=False)
    assert a.as_dict() == b.as_dict(), "same seed, different match"
    assert 1 <= a.rank <= 10 and a.survival_s > 0
    print(f"  {a.map}: rank {a.rank}, {a.cubes} cubes, {a.kills} kills")


@pytest.mark.slow
def test_sim_gym_env_passes_the_gymnasium_checker():
    from gymnasium.utils.env_checker import check_env
    from sim.gym_env import SimShowdownEnv
    check_env(SimShowdownEnv(brawler="colt", teacher=True), skip_render_check=True)
