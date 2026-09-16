"""
rl/combat.py
============

When to shoot, how far to trust the shot, and whether to aim it by hand.
This is a SCRIPT, not something the policy learns.

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

PER-BRAWLER RANGE — AND WHY THE GATE CAN ONLY EVER TIGHTEN
-----------------------------------------------------------
The range gate used to be one number for everyone: don't shoot past 42% of the
screen's short side. That is wrong in both directions at once. Edgar reaches
about 2.7 tiles and was firing at four times that, paying `attack_cost` to spray
at people he cannot touch; Piper reaches 10 and was being cut off early.

`brawlers/abilities.py` has the real per-brawler range, so the gate now uses it.
Converting tiles to pixels needs one calibration number — how many tiles the
camera shows across the play area — and that number is an ESTIMATE, so the
conversion is deliberately arranged to fail safe:

    effective = min(max_range_frac, brawler_tiles / tiles_across_screen)

The per-brawler term can only ever make the gate TIGHTER than the old global
one. If `tiles_across_screen` is wrong, the worst case is the behaviour we
already had; every improvement is on the short-range brawlers who were wasting
shots. Verify it before relying on it — see `CombatConfig.tiles_across_screen`.

AIM MODE
--------
`decide()` also reports whether this brawler's shot should be AIMED rather than
tapped, and at what. The aim mode is derived from class and range in
`brawlers/abilities.py` — a rule over published facts rather than 106
hand-written opinions. The executor turns an aim vector into a drag from the
attack button; a plain tap remains the auto-aim path.

This matters most for the eight Artillery brawlers. Their whole purpose is
arcing over walls, and auto-aim will not fire at a target it has no line of
sight to — so an auto-aiming thrower throws away the one thing it is for.
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
    #
    # This is now a CEILING rather than the rule: the per-brawler range below
    # can tighten it but never loosen it.
    max_range_frac: float = 0.42
    # Supers are scarcer, so spend them on something closer and more certain.
    super_range_frac: float = 0.34

    # --- tiles -> pixels ------------------------------------------------- #
    # How many game tiles the camera shows across the play rect. The ONE number
    # standing between a published range in tiles and a distance in pixels.
    #
    # THIS IS AN ESTIMATE and it has not been measured on this setup. It is
    # safe to ship unverified only because of how it is used: the per-brawler
    # range is applied as `min(max_range_frac, tiles / tiles_across_screen)`,
    # so a wrong value can only make the gate stricter than the old global one,
    # never more permissive. Nothing new can break; short-range brawlers simply
    # stop wasting shots by less than they could.
    #
    # TO VERIFY: run the web UI on a map with a straight wall of known length,
    # count the tiles it spans on screen, and divide the play rect's width by
    # the tiles-per-pixel that implies. Then set this and drop
    # `range_gate_is_ceiling` if you trust it.
    tiles_across_screen: float = 17.0
    # While True the per-brawler range may only tighten `max_range_frac`. Set
    # False once `tiles_across_screen` is measured and you want the real range
    # to govern in both directions (which is what lets Piper shoot at 10 tiles).
    range_gate_is_ceiling: bool = True

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

    # NOT CONSULTED YET -- the seam for rl/line_of_sight.py, recorded here so
    # the decision has somewhere to live when it is made.
    #
    # Whether a wall between us and the target should hold fire. It is a
    # PER-BRAWLER fact, not a general one: throwers arc over walls, melee have
    # no projectile, Rico banks off them, and several supers break them, so
    # "blocked -> do not fire" is correct for plain shooters and wrong for
    # everyone else. Default False = behave exactly as before.
    wall_blocks_shots: bool = False


@dataclass(frozen=True)
class CombatDecision:
    """What to fire, and where to point it.

    A richer return than the old `(attack, super)` tuple because there is now a
    third thing to say: an aimed shot needs a DIRECTION, and the trace needs to
    show which mode fired so that a bad aim and a bad target are distinguishable
    after the fact. Iterating still yields `(attack, super)`, so every existing
    caller and every existing test keeps working unchanged.
    """

    attack: bool = False
    super: bool = False
    # Unit vector from the player toward the target, in screen space, or None
    # for an auto-aimed tap. Consumed by rl/actions.Intent -> the executor.
    aim: Optional[Tuple[float, float]] = None
    # 0..1 fraction of this brawler's range to throw at. Only meaningful for
    # lobbed attacks, where the drag length sets WHERE the shot lands rather
    # than just which way it goes.
    aim_frac: float = 1.0
    mode: str = "auto"            # "auto" | "manual"
    reason: str = "idle"

    def __iter__(self):
        yield self.attack
        yield self.super

    def __len__(self) -> int:
        return 2

    def __getitem__(self, i):
        return (self.attack, self.super)[i]


class CombatPolicy:
    """Decides attack/super from the state. Pure and cheap; no learning."""

    def __init__(self, config: Optional[CombatConfig] = None,
                 ability=None):
        """`ability` is a `brawlers.abilities.Ability`, or None.

        None is the honest default rather than an error: the offline env, the
        tests and any session without a brawler selected all have no per-brawler
        facts, and they must behave exactly as they did before this existed.
        """
        self.config = config or CombatConfig()
        self.ability = ability
        self.reset()

    def reset(self) -> None:
        self._super_cooldown = 0
        self.last_reason = "idle"

    # ------------------------------------------------------------------ #
    def range_frac(self) -> float:
        """The attack gate for THIS brawler, as a fraction of the short side.

        Falls back to the global ceiling whenever the per-brawler range is
        unknown — a brand-new brawler with no published range must not end up
        gated at zero.
        """
        cfg = self.config
        tiles = getattr(self.ability, "gate_range_tiles", None)
        if not tiles or cfg.tiles_across_screen <= 0:
            return cfg.max_range_frac
        frac = float(tiles) / float(cfg.tiles_across_screen)
        return min(cfg.max_range_frac, frac) if cfg.range_gate_is_ceiling else frac

    @property
    def aim_mode(self) -> str:
        return getattr(self.ability, "aim", "auto") or "auto"

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

        def _no(reason):
            self.last_reason = reason
            return CombatDecision(reason=reason, mode=self.aim_mode)

        if state is None:
            return _no("no state")

        enemies = list(getattr(state, "enemy_positions", ()) or ())
        player = getattr(state, "player_pos", None)
        if not enemies or player is None:
            return _no("no target")

        w, h = frame_size or getattr(state, "frame_size", (1280, 720))
        short = max(1.0, min(w, h))
        px, py = player
        nearest = min(enemies, key=lambda e: (e[0] - px) ** 2 + (e[1] - py) ** 2)
        dx, dy = nearest[0] - px, nearest[1] - py
        raw = hypot(dx, dy)
        dist = raw / short

        if cfg.hold_fire_in_bush and world is not None and _in_bush(world, player,
                                                                   cfg.bush_conceal_frac):
            return _no("concealed")

        gate = self.range_frac()
        attack = dist <= gate
        if attack and getattr(state, "ammo_known", False) and getattr(state, "ammo_count", 0) < 1:
            attack = False
            self.last_reason = "empty clip"
        elif attack:
            self.last_reason = f"target at {dist:.2f}"
        else:
            self.last_reason = f"out of range ({dist:.2f} > {gate:.2f})"

        charge = getattr(state, "super_charge", None) or 0.0
        fire_super = (charge >= cfg.super_ready
                      and dist <= cfg.super_range_frac
                      and self._super_cooldown <= 0)
        if fire_super:
            self._super_cooldown = cfg.super_cooldown_ticks
            self.last_reason += " +SUPER"

        # --- where to point it ------------------------------------------- #
        # Only computed when something actually fires: an aim vector attached to
        # a decision that fires nothing is a value nobody reads and everybody
        # has to reason about.
        aim = None
        aim_frac = 1.0
        mode = self.aim_mode
        if (attack or fire_super) and mode == "manual" and raw > 1e-6:
            aim = (dx / raw, dy / raw)
            # For a lobbed attack the drag LENGTH decides where the shot lands,
            # so it has to encode distance, not just direction. For everything
            # else the direction is the whole message and full deflection is the
            # most reliable thing to send.
            if getattr(self.ability, "lobs_over_walls", False):
                aim_frac = max(0.15, min(1.0, dist / max(gate, 1e-6)))
            self.last_reason += " (aimed)"

        return CombatDecision(attack=bool(attack), super=bool(fire_super),
                              aim=aim, aim_frac=aim_frac, mode=mode,
                              reason=self.last_reason)


def _in_bush(world, player_px, threshold: float) -> bool:
    """Is the player standing in cover, per the fused world map?"""
    try:
        gx, gy = world.clamp_cell(world.to_cell(player_px))
        return float(world.bush[gy, gx]) >= threshold
    except Exception:
        return False
