"""
rl/kills.py
===========

Kill attribution — deciding when a drop in the Showdown "brawlers left" counter
was caused by THE AGENT (a knockout: reward ++) versus someone/something else —
gas, a rival's fight, fall damage (still rewarded, but only as improved
placement: +).

The screen shows no per-damage-source information, so exact attribution isn't
possible from pixels alone. This is a deliberately simple, well-documented
heuristic that needs no new perception:

    Credit a kill on a tick when ALL of these hold:
      1. brawlers_left dropped (a player died this tick),
      2. the agent dealt damage very recently — the super charge rose within a
         short window (the super only charges by dealing damage), and
      3. an enemy was near the agent very recently (someone was in fighting range).

    kills credited = min(players that died, enemies recently near, cap)

Everything else that dies is left to the placement reward, so nothing is
double-counted incorrectly: your own kill pays BOTH the kill and placement terms
(intended), while a distant death pays only placement.

Stateful across a match, auto-resets between matches. Feed it the LivePerception
`live` dict each tick via :meth:`update`. A future upgrade could read the on-screen
defeat banner for exact attribution; this heuristic is the no-new-perception path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class KillConfig:
    damage_window: int = 10       # ticks: how recently "dealt damage" still counts
    near_window: int = 12         # ticks: how recently "enemy was near" still counts
    near_radii: float = 14.0      # enemy within this many player-ring radii = near
    super_rise_eps: float = 0.02  # min super-charge rise counted as "dealt damage"
    max_credit_per_tick: int = 2  # cap kills credited in a single tick


class KillAttributor:
    """Credits agent kills from the LivePerception live-dict stream (stateful)."""

    def __init__(self, config: Optional[KillConfig] = None):
        self.config = config or KillConfig()
        self.reset()

    def reset(self) -> None:
        self._prev_left: Optional[int] = None
        self._last_super: float = 0.0
        self._since_damage: int = 10 ** 9
        self._since_near: int = 10 ** 9
        self._last_near_count: int = 0
        self.total_credited: int = 0

    def update(self, live: dict) -> int:
        """Return kills to credit the agent THIS tick (0, 1, ...)."""
        cfg = self.config
        gs = live.get("game_state") or {}
        state = gs.get("state")
        left = gs.get("brawlers_left")

        # Between matches: reset and credit nothing.
        if state != "in_match":
            if state in ("loading", "match_end"):
                self.reset()
            return 0

        # (2) recently dealt damage? — super charge rose.
        sc = live.get("super_charge")
        if sc is not None and (sc - self._last_super) >= cfg.super_rise_eps:
            self._since_damage = 0
        else:
            self._since_damage += 1
        if sc is not None:
            self._last_super = sc

        # (3) recently had an enemy near?
        near_count = self._nearby_enemy_count(live)
        if near_count > 0:
            self._since_near = 0
            self._last_near_count = near_count
        else:
            self._since_near += 1

        # (1) did a player die, and can we attribute it?
        kills = 0
        if self._prev_left is not None and left is not None:
            drop = self._prev_left - left
            if drop > 0:
                dealt = self._since_damage <= cfg.damage_window
                near = self._since_near <= cfg.near_window
                if dealt and near:
                    kills = min(drop, max(1, self._last_near_count), cfg.max_credit_per_tick)
        if left is not None:
            self._prev_left = left

        self.total_credited += kills
        return kills

    # allow use as env kills_fn: fn(frame, live) -> int
    def __call__(self, _frame, live: dict) -> int:
        return self.update(live)

    def _nearby_enemy_count(self, live: dict) -> int:
        enemies = live.get("enemies") or []
        if not enemies:
            return 0
        anchor = live.get("anchor")
        # No usable player ring -> treat any visible enemy as "near" (the camera
        # only shows a small area around the player anyway).
        if not anchor or len(anchor) < 3 or not anchor[2]:
            return len(enemies)
        ax, ay, ar = anchor[0], anchor[1], anchor[2]
        reach = self.config.near_radii * ar
        n = 0
        for e in enemies:
            c = e.get("center") if isinstance(e, dict) else e
            if c is None:
                continue
            if ((c[0] - ax) ** 2 + (c[1] - ay) ** 2) ** 0.5 <= reach:
                n += 1
        return n
