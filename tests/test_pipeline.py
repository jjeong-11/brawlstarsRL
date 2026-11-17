"""rl/pipeline.LiveAgent and rl/timebase: one frame in, one decision out, and
tick-counted settings that mean the same number of seconds at any loop rate."""

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

cv2 = pytest.importorskip("cv2")


def test_tick_counted_settings_keep_their_duration_in_seconds():
    from rl.combat import CombatConfig
    from rl.enemy_tracker import TrackerConfig
    from rl.path_planner import PathPlannerConfig
    from rl.tactics import TacticsConfig
    from rl.timebase import (REFERENCE_TICK, combat_config_for, planner_config_for,
                             tactics_config_for, tracker_config_for)

    # At the simulator's tick nothing moves: tuned results stay valid.
    assert planner_config_for(REFERENCE_TICK) == PathPlannerConfig()
    assert tracker_config_for(REFERENCE_TICK) == TrackerConfig()
    assert tactics_config_for(REFERENCE_TICK) == TacticsConfig(tick_seconds=REFERENCE_TICK)
    assert combat_config_for(REFERENCE_TICK) == CombatConfig(tick_seconds=REFERENCE_TICK)

    base_p, base_t, base_b = PathPlannerConfig(), TrackerConfig(), TacticsConfig()
    for tick in (0.25, 0.05, 1 / 15):
        p, t, b, c = (planner_config_for(tick), tracker_config_for(tick),
                      tactics_config_for(tick), combat_config_for(tick))
        slack = tick                                   # rounding to whole ticks
        assert abs(p.stuck_ticks * tick - base_p.stuck_ticks * REFERENCE_TICK) <= slack
        assert abs(t.memory_ticks * tick - base_t.memory_ticks * REFERENCE_TICK) <= slack
        assert abs(t.hurt_ticks * tick - base_t.hurt_ticks * REFERENCE_TICK) <= slack
        assert abs(b.visible_age_ticks * tick - base_b.visible_age_ticks * REFERENCE_TICK) <= slack
        assert abs(b.strafe_ticks * tick - base_b.strafe_ticks * REFERENCE_TICK) <= slack
        # per-tick distances shrink with the tick; latency stays a time
        assert abs(t.max_speed_frac / tick - base_t.max_speed_frac / REFERENCE_TICK) < 1e-9
        assert abs(c.latency_ticks * tick - CombatConfig().latency_ticks * REFERENCE_TICK) < 1e-9
        assert b.tick_seconds == pytest.approx(tick) and c.tick_seconds == pytest.approx(tick)


def test_live_agent_turns_a_real_frame_into_a_decision():
    import shutil
    if shutil.which("tesseract") is None:
        pytest.skip("tesseract not installed (the screen classifier needs it)")
    from rl.pipeline import LiveAgent

    frame = cv2.imread(str(ROOT / "media" / "fixtures" / "showdown.png"))
    if frame is None:
        pytest.skip("fixture missing")
    agent = LiveAgent("shelly", tick_seconds=0.1, async_game_state=False)
    tk = None
    for i in range(30):                                  # 3 s on the video clock
        tk = agent.step(frame, now=i * 0.1)
    assert tk.in_match, "perception never confirmed the match"
    assert tk.state.player_pos is not None and tk.state.health == 10880
    assert tk.goal is not None and tk.intent is not None
    assert tk.combat is not None and tk.reward is not None
    for stage in ("perception", "terrain", "gas", "camera", "brain", "combat", "route"):
        assert stage in tk.timings
    lat = agent.latency()
    assert lat["total"]["n"] == 30 and lat["total"]["p50"] > 0
    assert " | " in agent.explain()                      # option scores are exposed
    # This frame has no gas anywhere. The player's own sprite must not read as
    # gas -- not even from the scans made before the player was located.
    assert tk.goal.mode != "gas", agent.explain()
    assert float(agent.planner.world.gas.max()) < 0.3
    print(f"\n  total p50 {lat['total']['p50']:.1f} ms, p95 {lat['total']['p95']:.1f} ms; "
          f"decision: {agent.explain()}")

    # Gas is rationed: scanned on its own cadence, not every tick.
    scans = 0
    for i in range(30, 50):
        scans += agent.step(frame, now=i * 0.1).gas is not None
    assert 3 <= scans <= 7                               # every 0.4 s over 2 s
