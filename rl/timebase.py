"""
rl/timebase.py
==============

Keep behaviour fixed in SECONDS when the loop rate changes.

The planner, tracker, brain and combat count many things in ticks: "stuck
after 5 motionless decisions", "an unseen enemy is remembered 30 ticks", "a
track is visible if seen within 6 ticks", "strafe every 7 ticks". Those
numbers were chosen at the simulator's 0.1 s tick and mean half a second,
three seconds, 0.6 s and 0.7 s. Run the live loop at 4 Hz and they silently
become 1.25 s, 7.5 s, 1.5 s and 1.75 s; run it at 20 Hz and they halve. The
bot that was tuned is not the bot that plays.

`*_config_for(tick_seconds)` rescales every tick-counted field so its
duration in seconds is what it was at the reference tick. At 0.1 s the
configs come back unchanged (the simulator's results do not move).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

REFERENCE_TICK = 0.1


def _scale(tick_seconds: Optional[float]) -> float:
    """Ticks per reference tick: 2.0 at 20 Hz, 0.4 at 4 Hz, 1.0 when unpaced."""
    if not tick_seconds or tick_seconds <= 0:
        return 1.0
    s = REFERENCE_TICK / float(tick_seconds)
    return 1.0 if abs(s - 1.0) < 0.05 else s


def _ticks(n: int, s: float, lo: int = 1) -> int:
    return max(lo, int(round(n * s)))


def planner_config_for(tick_seconds: Optional[float], base=None):
    from .path_planner import PathPlannerConfig
    base = base or PathPlannerConfig()
    s = _scale(tick_seconds)
    if s == 1.0:
        return base
    return replace(
        base,
        commit_ticks=tuple(_ticks(t, s) for t in base.commit_ticks),
        stuck_ticks=_ticks(base.stuck_ticks, s, 2),
        bump_ticks=_ticks(base.bump_ticks, s, 2) if base.bump_ticks > 0 else 0,
        # "barely moving" is a distance per tick: shorter ticks, shorter steps.
        stuck_speed=base.stuck_speed / s,
    )


def tracker_config_for(tick_seconds: Optional[float], base=None):
    from .enemy_tracker import TrackerConfig
    base = base or TrackerConfig()
    s = _scale(tick_seconds)
    if s == 1.0:
        return base
    return replace(
        base,
        memory_ticks=_ticks(base.memory_ticks, s, 2),
        predict_ticks=_ticks(base.predict_ticks, s, 1),
        box_memory_ticks=_ticks(base.box_memory_ticks, s, 2),
        cube_memory_ticks=_ticks(base.cube_memory_ticks, s, 2),
        hurt_ticks=_ticks(base.hurt_ticks, s, 1),
        # A believable speed is a distance per tick.
        max_speed_frac=base.max_speed_frac / s,
    )


def tactics_config_for(tick_seconds: Optional[float], base=None):
    from .tactics import TacticsConfig
    base = base or TacticsConfig(tick_seconds=tick_seconds or REFERENCE_TICK)
    s = _scale(tick_seconds)
    if s == 1.0:
        return base
    return replace(
        base,
        tick_seconds=float(tick_seconds),
        strafe_ticks=_ticks(base.strafe_ticks, s, 1),
        give_up_ticks=_ticks(base.give_up_ticks, s, 2),
        blacklist_ticks=_ticks(base.blacklist_ticks, s, 2),
        explore_hold_ticks=_ticks(base.explore_hold_ticks, s, 1),
        visible_age_ticks=_ticks(base.visible_age_ticks, s, 1),
    )


def combat_config_for(tick_seconds: Optional[float], base=None):
    from .combat import CombatConfig
    base = base or CombatConfig(tick_seconds=tick_seconds or REFERENCE_TICK)
    s = _scale(tick_seconds)
    if s == 1.0:
        return base
    return replace(
        base,
        tick_seconds=float(tick_seconds),
        max_target_age=_ticks(base.max_target_age, s, 1),
        super_cooldown_ticks=_ticks(base.super_cooldown_ticks, s, 1),
        # A latency in seconds, expressed in ticks.
        latency_ticks=base.latency_ticks * s,
    )
