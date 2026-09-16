"""
brawlers/profiles.py
====================

The hook for per-brawler advantages — deliberately empty for now.

Separate weights (see `registry.py`) let each brawler *learn* something
different. This module is for what we can tell it without waiting for a
gradient: knowledge we already have and would otherwise pay tens of thousands
of samples to rediscover.

The precedent is gas. There is no "moved toward gas" reward term; gas is a cost
layer in the A* grid, so the agent routes around the cloud from the very first
frame at zero sample cost. The same trick applies per brawler:

    Edgar   wants to be at range 0.      Piper wants to be at range 9.
    Barley  wants a wall between them.   Bull wants no wall at all.

None of that has to be learned. It is a *prior* on the planner and combat
policy, and priors are free.

Shape of the thing
------------------
A profile is pure data — overrides layered onto the existing configs, never a
fork of the logic:

    RewardConfig(**{**asdict(base), **profile.reward_overrides})
    CombatConfig(**{**asdict(base), **profile.combat_overrides})
    PathPlannerConfig(**{**asdict(base), **profile.planner_overrides})

Data, not code, for one reason: an override that only *tunes* an existing knob
cannot introduce a behaviour the trace panel does not already display. A
per-brawler `decide()` fork could, and then a bad brawler script and a bad
policy become indistinguishable in the trace — which is exactly the confusion
`rl/debug_trace.py` exists to prevent.

Every profile is empty today, so behaviour is bit-for-bit identical to having
no profiles at all. That is on purpose: this lands the seam and the call sites
now, while there is nothing to regress, so adding Edgar's aggression later is a
three-line data change instead of a refactor.

WHEN YOU FILL THESE IN, MEASURE. `preferred_range` is not a fact about the
brawler, it is a claim about what wins — check it against a rollout the way the
attack-gating and box-shaping numbers in the README were checked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class BrawlerProfile:
    """Per-brawler priors. Empty = identical to the shared defaults."""

    id: str

    # --- descriptive (not yet consumed by anything) --------------------- #
    # Comfortable engagement distance in grid cells, or None for "no opinion".
    # A thrower's is short even though its attack is long, because its attack
    # arcs over walls -- which is why this is a separate number from range.
    preferred_range: Optional[float] = None
    # Does the attack cross walls (Barley, Dynamike, Sprout, Grom, Tick)?
    lobs_over_walls: bool = False
    # Does the super reposition the player (Mortis, Edgar, Buzz, Max)? Movement
    # supers break the camera tracker's frame-to-frame assumption, so the
    # planner may need to drop its latch on the tick one fires.
    mobility_super: bool = False

    # --- override dicts, merged onto the shared configs ------------------ #
    reward_overrides: Dict[str, Any] = field(default_factory=dict)
    combat_overrides: Dict[str, Any] = field(default_factory=dict)
    planner_overrides: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_default(self) -> bool:
        return not (self.reward_overrides or self.combat_overrides
                    or self.planner_overrides)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "preferred_range": self.preferred_range,
            "lobs_over_walls": self.lobs_over_walls,
            "mobility_super": self.mobility_super,
            "is_default": self.is_default,
            "reward_overrides": dict(self.reward_overrides),
            "combat_overrides": dict(self.combat_overrides),
            "planner_overrides": dict(self.planner_overrides),
        }


# --------------------------------------------------------------------------- #
# Per-brawler overrides go here, keyed by roster id. Everything absent gets the
# defaults, so this stays a short file even with 104 brawlers.
#
# Worked example of what a filled-in entry will look like — commented out
# because none of it is measured yet and an unmeasured prior is just a bug with
# a nice story attached:
#
#   "edgar": BrawlerProfile(
#       id="edgar",
#       preferred_range=1.0,
#       mobility_super=True,
#       # Edgar heals by dealing damage, so disengaging to regen -- which
#       # health_regain pays every other brawler for -- is actively wrong here.
#       reward_overrides={"health_regain": 1.0},
#       combat_overrides={"min_engage_cells": 0.0},
#   ),
#   "piper": BrawlerProfile(
#       id="piper",
#       preferred_range=8.5,
#       # Piper's damage scales with distance, so a close shot is nearly free
#       # damage for the opponent. Raise the cost of firing point-blank.
#       combat_overrides={"min_engage_cells": 5.0},
#   ),
# --------------------------------------------------------------------------- #
PROFILES: Dict[str, BrawlerProfile] = {}


def profile_for(brawler_id: str) -> BrawlerProfile:
    """This brawler's profile, or an empty one. Never raises."""
    return PROFILES.get(brawler_id) or BrawlerProfile(id=brawler_id)


def apply_overrides(config: Any, overrides: Dict[str, Any]) -> Any:
    """Return a copy of a dataclass config with `overrides` applied.

    Unknown keys are dropped with a warning rather than raising: a profile that
    names a knob a later refactor renamed should not take a live session down
    mid-match. It should be loud in the log and otherwise inert.
    """
    if not overrides:
        return config
    import dataclasses
    if not dataclasses.is_dataclass(config):
        raise TypeError(f"{type(config).__name__} is not a dataclass config")
    known = {f.name for f in dataclasses.fields(config)}
    unknown = set(overrides) - known
    if unknown:
        print(f"[brawlers] ignoring unknown {type(config).__name__} "
              f"override(s): {sorted(unknown)}")
    good = {k: v for k, v in overrides.items() if k in known}
    return dataclasses.replace(config, **good) if good else config
