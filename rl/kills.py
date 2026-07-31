"""
rl/kills.py
===========

Kill attribution — deciding when a drop in the Showdown "brawlers left" counter
was caused by THE AGENT (a knockout: reward ++) versus someone/something else —
gas, a rival's fight, fall damage (still rewarded, but only as improved
placement: +).

The screen shows no per-damage-source information, so exact attribution is not
possible from pixels alone without reading the defeat banner (see KNOWN GAPS).
This is a deliberately simple heuristic that needs no new perception:

    Credit a kill on a tick when ALL of these hold:
      1. brawlers_left dropped (a player died this tick),
      2. the agent SHOT very recently, and
      3. an enemy was within auto-aim range very recently.

Everything else that dies is left to the placement reward, so nothing is
double-counted incorrectly: your own kill pays BOTH the kill and placement terms
(intended), while a distant death pays only placement.

WHY IT NO LONGER INFERS "DEALT DAMAGE" FROM THE SUPER CHARGE
-------------------------------------------------------------
The previous version used a rise in super charge as the damage signal, on the
reasoning that the super only charges by dealing damage. True, but it has a
blind spot that bites exactly when it matters most: THE SUPER CHARGE CAPS AT
1.0. Once it is full, dealing damage produces no rise at all, `_since_damage`
never resets, and kills stop being credited — precisely when the agent is at
its strongest and most likely to be getting kills.

Since `rl/combat.py` now decides shooting rather than the policy, we simply
KNOW when the agent fired. That is a direct observation rather than an
inference, it has no cap, and it costs nothing. The super rise is kept as a
secondary signal because a shot that missed is weaker evidence than damage that
demonstrably landed.

RANGE
-----
"Near" is expressed in the same units `rl/combat.py` uses to decide whether to
shoot at all, because they are the same question: could the agent have hit this
enemy? The old threshold was 14 player-ring radii — roughly 500px on a 2424px
frame, or most of the screen — which meant essentially any visible enemy
counted, and the gate did little.

KNOWN GAPS
----------
This is still a heuristic and it still over-credits. In a 10-player lobby many
deaths happen off-screen while the agent happens to be mid-fight, and those
will be credited. The exact fix is the on-screen defeat banner
("<name> defeated <name>", top-centre for ~2s), which is the ground truth and is
listed in `rl/state.MISSING_EXTRACTORS`. It needs an ROI calibrated against real
footage, which this repo does not currently have.

Until then, `RewardConfig.kill` being large (+5.0, second only to winning) means
a false credit is expensive. Consider lowering it until attribution is exact.
Stateful across a match, auto-resets between matches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class KillConfig:
    damage_window: int = 10       # ticks: how recently "we shot" still counts
    near_window: int = 12         # ticks: how recently "enemy was near" still counts
    # Enemy within this fraction of the frame's SHORT side counts as engageable.
    # Mirrors combat.CombatConfig.max_range_frac -- "was it in range" is the
    # same question there and here, and letting the two drift apart would mean
    # crediting kills at distances the agent would not even have fired at.
    near_frac: float = 0.42
    # Fallback for frames with no usable player ring, in player-ring radii.
    near_radii: float = 6.0
    super_rise_eps: float = 0.02  # min super-charge rise counted as "damage landed"
    max_credit_per_tick: int = 2  # cap kills credited in a single tick
    # Require the shot to have LANDED (super charge rose) as well as been fired.
    # Off by default: with auto-aim most shots at an in-range target do land, and
    # the super cap makes the landed-signal unavailable at full charge -- so
    # requiring it would reintroduce the blind spot this class exists to avoid.
    require_damage_landed: bool = False


class KillAttributor:
    """Credits agent kills from the LivePerception live-dict stream (stateful)."""

    def __init__(self, config: Optional[KillConfig] = None):
        self.config = config or KillConfig()
        self.reset()

    def reset(self) -> None:
        self._prev_left: Optional[int] = None
        self._last_super: float = 0.0
        self._since_shot: int = 10 ** 9
        self._since_damage: int = 10 ** 9
        self._since_near: int = 10 ** 9
        self._last_near_count: int = 0
        self.total_credited: int = 0
        self.total_deaths_seen: int = 0

    def note_fired(self) -> None:
        """Record that the agent attacked this tick.

        Called by the env from the combat policy's decision. This is a direct
        observation of the thing the old code had to infer from the super
        charge, and unlike that inference it does not stop working when the
        super is full.
        """
        self._since_shot = -1        # incremented to 0 by this tick's update()

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

        self._since_shot += 1

        # Did damage demonstrably land? (super charge rose). Secondary evidence:
        # useful when available, unusable at full charge -- see the module docs.
        sc = live.get("super_charge")
        if sc is not None and (sc - self._last_super) >= cfg.super_rise_eps:
            self._since_damage = 0
        else:
            self._since_damage += 1
        if sc is not None:
            self._last_super = sc

        # Was an enemy in range recently?
        near_count = self._nearby_enemy_count(live)
        if near_count > 0:
            self._since_near = 0
            self._last_near_count = near_count
        else:
            self._since_near += 1

        # Did a player die, and can we attribute it?
        kills = 0
        if self._prev_left is not None and left is not None:
            drop = self._prev_left - left
            if drop > 0:
                self.total_deaths_seen += drop
                engaged = self._since_shot <= cfg.damage_window
                if cfg.require_damage_landed:
                    engaged = engaged and self._since_damage <= cfg.damage_window
                near = self._since_near <= cfg.near_window
                if engaged and near:
                    kills = min(drop, max(1, self._last_near_count),
                                cfg.max_credit_per_tick)
        if left is not None:
            self._prev_left = left

        self.total_credited += kills
        return kills

    # allow use as env kills_fn: fn(frame, live) -> int
    def __call__(self, _frame, live: dict) -> int:
        return self.update(live)

    def _nearby_enemy_count(self, live: dict) -> int:
        """Enemies close enough that the agent could plausibly have hit them."""
        enemies = live.get("enemies") or []
        if not enemies:
            return 0
        anchor = live.get("anchor")
        frame = live.get("frame_size")
        if not anchor or len(anchor) < 3 or not anchor[2]:
            # No usable player ring. The camera only shows a small area around
            # the player, so any visible enemy is at least plausible -- but this
            # is the weakest case, so it is not treated as free evidence.
            return len(enemies)

        ax, ay, ar = anchor[0], anchor[1], anchor[2]
        if frame:
            reach = self.config.near_frac * min(frame[0], frame[1])
        else:
            reach = self.config.near_radii * ar
        n = 0
        for e in enemies:
            c = e.get("center") if isinstance(e, dict) else e
            if c is None:
                continue
            if ((c[0] - ax) ** 2 + (c[1] - ay) ** 2) ** 0.5 <= reach:
                n += 1
        return n

    def stats(self) -> str:
        """Credited vs observed, for eyeballing how loose the heuristic is."""
        seen = max(1, self.total_deaths_seen)
        return (f"credited {self.total_credited} of {self.total_deaths_seen} "
                f"deaths seen ({100 * self.total_credited / seen:.0f}%)")
