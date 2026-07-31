"""
rl/combat.py
============

When to shoot. This is a SCRIPT, not something the policy learns.

WHY TAKE IT AWAY FROM THE POLICY
--------------------------------
Attack and super used to be two binary heads on the action space. They were
never a good use of the policy's capacity, for a reason specific to this game:
Brawl Stars AUTO-AIMS a plain tap at the nearest enemy. There is no aiming
decision to learn — no angle, no lead, no charge time. The entire content of
"should I attack" is:

    is there something to shoot at, and do I have ammo?

Both of which are already in the observation, and both of which
`BrawlStarsEnv._gate_weapons` was ALREADY enforcing by overriding the policy.
So the two heads were mostly decorative: the policy proposed, the gate
disposed, and the gate's rule was the one that ran.

Measured on a rollout before the gate existed: 89% of attacks were fired with
an empty clip and 100% of supers were fired at zero charge. That is what
learning this from scratch looks like — the policy spends real samples
discovering a rule that can simply be written down.

WHAT IT BUYS
------------
The action space drops from `MultiDiscrete([16, 3, 2, 2])` to
`MultiDiscrete([16, 3])`: 48 combinations instead of 192, and 19 logits instead
of 23. Every sample now goes into the part that is genuinely hard — where to
move. On a project whose bottleneck is samples (a live phone produces ~36k
steps an hour), that is the whole argument.

WHAT IT COSTS
-------------
Real strategy is lost at the margins. A human holds fire to stay hidden in a
bush, saves a super for a wall break or an escape, and declines a shot that
would reveal position. None of that is here. `hold_fire_in_bush` is a nod at
the first; the rest is a deliberate trade, and if movement ever stops being the
binding constraint it is worth revisiting.

The honest test is an A/B: this script against the learned heads, same number
of live steps. Until that is run, this is a well-motivated guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Optional, Tuple


@dataclass(frozen=True)
class CombatConfig:
    # Charge at which the super button does anything at all. Mirrors
    # perception.getSuper.READY_CHARGE; below it the tap is a pure no-op.
    super_ready: float = 0.9

    # Do not shoot at an enemy further away than this fraction of the play
    # area's short side. Auto-aim happily fires at a target out of range: the
    # shot leaves the barrel, travels its fixed distance, and lands on nothing.
    # That wastes a round AND reveals position, which is the worse half.
    max_range_frac: float = 0.42
    # Supers are scarcer, so spend them on something closer and more certain.
    super_range_frac: float = 0.34

    # Hold fire while hidden. Bushes are the main defensive mechanic in
    # Showdown and firing from one gives the position away, which is exactly
    # the trade a human makes deliberately. Off by default: the perception
    # stack reports the bush fraction of the player's cell but has never been
    # validated for "am I concealed", and shooting when you should not is a
    # smaller mistake than never shooting.
    hold_fire_in_bush: bool = False
    bush_conceal_frac: float = 0.6

    # Minimum decisions between supers. Without it a full charge fires on
    # several consecutive ticks before the charge readout catches up, wasting
    # the rest on empty air.
    super_cooldown_ticks: int = 12


class CombatPolicy:
    """Decides attack/super from the state. Pure and cheap; no learning."""

    def __init__(self, config: Optional[CombatConfig] = None):
        self.config = config or CombatConfig()
        self.reset()

    def reset(self) -> None:
        self._super_cooldown = 0
        self.last_reason = "idle"

    # ------------------------------------------------------------------ #
    def decide(self, state, frame_size=None, world=None) -> Tuple[bool, bool]:
        """-> (fire_attack, fire_super).

        The three gates are deliberately ASYMMETRIC about uncertainty, and the
        asymmetry is the interesting part:

          * enemy visible  — REQUIRED. Auto-aim has nothing to aim at otherwise,
            so the tap cannot possibly do anything.
          * ammo           — only enforced when the reading is TRUSTED
            (`ammo_known`). A failed read also reports 0, and the bar is only
            located on a minority of real frames, so trusting the count alone
            would block nearly every attack. Firing on an unknown clip costs a
            wasted tap; refusing to fire on a full one costs the fight.
          * super charge   — ALWAYS enforced. It comes from a fixed HUD ROI that
            is never occluded, so 0.0 means uncharged rather than unknown.
        """
        cfg = self.config
        if self._super_cooldown > 0:
            self._super_cooldown -= 1

        if state is None:
            self.last_reason = "no state"
            return (False, False)

        enemies = list(getattr(state, "enemy_positions", ()) or ())
        player = getattr(state, "player_pos", None)
        if not enemies or player is None:
            self.last_reason = "no target"
            return (False, False)

        w, h = frame_size or getattr(state, "frame_size", (1280, 720))
        short = max(1.0, min(w, h))
        px, py = player
        nearest = min(enemies, key=lambda e: (e[0] - px) ** 2 + (e[1] - py) ** 2)
        dist = hypot(nearest[0] - px, nearest[1] - py) / short

        if cfg.hold_fire_in_bush and world is not None and _in_bush(world, player,
                                                                   cfg.bush_conceal_frac):
            self.last_reason = "concealed"
            return (False, False)

        attack = dist <= cfg.max_range_frac
        if attack and getattr(state, "ammo_known", False) and getattr(state, "ammo_count", 0) < 1:
            attack = False
            self.last_reason = "empty clip"
        elif attack:
            self.last_reason = f"target at {dist:.2f}"
        else:
            self.last_reason = f"out of range ({dist:.2f})"

        charge = getattr(state, "super_charge", None) or 0.0
        fire_super = (charge >= cfg.super_ready
                      and dist <= cfg.super_range_frac
                      and self._super_cooldown <= 0)
        if fire_super:
            self._super_cooldown = cfg.super_cooldown_ticks
            self.last_reason += " +SUPER"

        return (bool(attack), bool(fire_super))


def _in_bush(world, player_px, threshold: float) -> bool:
    """Is the player standing in cover, per the fused world map?"""
    try:
        gx, gy = world.clamp_cell(world.to_cell(player_px))
        return float(world.bush[gy, gx]) >= threshold
    except Exception:
        return False
