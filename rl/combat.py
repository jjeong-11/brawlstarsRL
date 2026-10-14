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

BOXES, TARGET CHOICE, WALLS AND LEADING
---------------------------------------
Four upgrades, each one a way the old rule lost fights or cubes:

  * POWER-CUBE BOXES WERE NEVER SHOT. `decide()` returned "no target" whenever
    no ENEMY was visible, and boxes are not enemies -- so the only cubes the
    agent ever collected were ones other players dropped. Cubes are the whole
    progression system of Showdown (+10% health and damage each), so this
    one line capped the agent's strength for the entire match. Boxes are now
    shot whenever no enemy is in range (`shoot_boxes`).
  * THE NEAREST ENEMY IS NOT ALWAYS THE RIGHT ONE. Auto-aim picks the nearest;
    a 400 HP straggler one tile further away is a kill, the full-health tank
    in front of him is not. Targets are scored by distance, remaining health
    and whether they are already in someone else's fight, and an auto-aimer
    switches to an AIMED shot only when the better target is worth it.
  * WALLS. A plain shooter firing at an enemy behind a wall hits the wall and
    spends a full reload doing it. `rl/line_of_sight.py` existed for exactly
    this and was never consulted. It is now, for brawlers whose attack does
    not arc -- with the fail-open rule that module documents, so an unknown
    map never silences the gun.
  * LEADING. A manual aimer used to fire at where the enemy IS. With a velocity
    from `rl/enemy_tracker.py`, aimed shots go to where he WILL be when the
    projectile arrives -- which is the stated reason marksmen aim by hand.

`CombatDecision.target_kind` says whether a shot was aimed at an enemy or a
box, so kill attribution (`rl/kills.py`) only hears about the first.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import List, Optional, Tuple


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
    # the tiles-per-pixel that implies. `CombatPolicy.calibrate()` also takes a
    # MEASURED value at runtime: once the localiser locks onto an arena its
    # scale (live cells per map cell) pins this down -- see rl/env.py.
    tiles_across_screen: float = 17.0
    # While True the per-brawler range may only tighten `max_range_frac`.
    #
    # NOW FALSE BY DEFAULT, because the ceiling was doing far more than its
    # docstring claimed. The old conversion returned tiles/tiles_across --
    # a fraction of the play WIDTH -- and compared it against distance over
    # the SHORT side, understating every range by the aspect ratio (2.24x on
    # a 20:9 phone). Together with the 0.42 ceiling that capped EVERY brawler
    # at ~3.2 tiles: Piper (10 tiles) and Colt (9.4) only returned fire at
    # point blank, out-ranged in every fight they took. With the units fixed
    # the real range governs; set True to restore the old cap.
    range_gate_is_ceiling: bool = False
    # Fraction of the published range actually used. Shots right at the edge
    # of the band are the ones a slightly-wrong tiles_across_screen turns into
    # misses, so the default keeps a little in hand.
    range_safety: float = 0.95

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

    # Whether a wall between us and the target should hold fire, consulted via
    # rl/line_of_sight.py. It is a PER-BRAWLER fact, not a general one:
    # throwers arc over walls, Rico banks off them, so "blocked -> do not fire"
    # is right for plain shooters and wrong for the rest.
    #   None  -> derive it: True when the brawler is KNOWN and does not lob
    #            (so an unknown brawler behaves exactly as before)
    #   bool  -> force it
    # The line test itself fails OPEN (unknown ground is never a wall).
    wall_blocks_shots: Optional[bool] = None
    # Blocked cells the line must cross before a shot counts as walled. One
    # cell of occupancy noise between two brawlers in the open must not
    # silence the gun; a real wall is several cells thick at this grid scale.
    los_min_wall_cells: int = 2

    # --- boxes ---------------------------------------------------------- #
    # Shoot power-cube boxes when no enemy is in range. See the module notes:
    # without this the agent never broke a single box.
    shoot_boxes: bool = True

    # --- target choice ---------------------------------------------------- #
    # Score = dist/gate + hp_weight * hp_fraction - hurt_bonus * in_a_fight.
    # Lower is better. With hp_weight 0.5 a target at HALF health one gate-
    # width further away is worth the same as a full-health one at the muzzle.
    smart_targeting: bool = True
    hp_weight: float = 0.5
    hurt_bonus: float = 0.25
    # An auto-aimer only abandons the auto-aim target (nearest in range) for an
    # AIMED shot at a better one when the score improves by at least this. A
    # tap is the more reliable gesture, so the alternative has to be worth it.
    switch_margin: float = 0.35
    # Tracked enemies last seen more than this many ticks ago are not shot at
    # (they are remembered for routing, not for firing into empty bushes).
    max_target_age: int = 6

    # --- leading ---------------------------------------------------------- #
    # Aimed shots lead the target by its tracked velocity. Needs a tracker;
    # without one the aim is at the current position, exactly as before.
    lead_shots: bool = True
    # Typical straight projectile speed. Brawl Stars bullets fly ~3000-4000
    # game units/s and a tile is 300 units, so 10-13 tiles/s.
    projectile_tiles_per_s: float = 11.0
    # Lobbed attacks land after a roughly constant flight time.
    lob_flight_s: float = 0.7
    # Length of one decision step, to turn px/tick velocities into a lead.
    tick_seconds: float = 0.1
    # Cap on how far a lead may move the aim point, fraction of the short
    # side, so a velocity glitch cannot swing the shot off into space.
    max_lead_frac: float = 0.22
    # AUTO-AIM DOES NOT LEAD. A tap fires at where the target IS; at range a
    # strafing enemy has moved a tile or more by the time the shot arrives.
    # With a tracked velocity, an aimed shot at the intercept point is
    # strictly better, so auto-aim brawlers switch to it when the target is
    # moving (speed above `lead_min_speed_frac` of the short side per tick)
    # and further than `lead_min_range_frac` of the reach -- close in, the
    # lead is smaller than the target and the tap is the surer gesture.
    #
    # OFF BY DEFAULT, because measured in sim/ it loses: 30% accuracy against
    # 33% for the plain tap, even with perfect perception. The game resolves a
    # tap against the target's TRUE position at the instant it fires; an aimed
    # shot uses our perceived position, which is already a tick or more old
    # when the touch lands, and a juking enemy makes the extrapolation a coin
    # flip. The tap wins on information, not on geometry.
    lead_auto_aimers: bool = False
    lead_min_speed_frac: float = 0.008
    lead_min_range_frac: float = 0.35
    # Ticks between deciding a shot and it leaving the barrel (adb/sendevent
    # latency + the game's input handling). Aimed shots extrapolate the target
    # this much further, on top of the projectile's flight time. Measure with
    # scripts/measure_latency.py; the env passes --action-delay through.
    latency_ticks: float = 1.0
    # Fraction of the extrapolated motion to lead by. Under uncertainty about
    # whether the target keeps its heading, the expected-hit-maximising lead
    # is less than the full intercept; see the measurement in the README.
    lead_gain: float = 0.35

    # --- fire rate --------------------------------------------------------- #
    # Minimum seconds between attack taps. No brawler fires faster than about
    # three shots a second -- each attack has its own animation, and taps that
    # land inside it are dropped -- so tapping every 0.1s tick buys nothing.
    # It is not free, either: on the adb backend every tap is another `input`
    # command queued behind the movement swipe, which is exactly the backlog
    # Controls.tuned_for_tick warns about. 0 disables.
    attack_interval_s: float = 0.3

    # --- ammo discipline -------------------------------------------------- #
    # Keep the last round for a sure shot: with a TRUSTED reading of exactly
    # one round, do not spend it on a target further than this fraction of
    # the gate unless it is nearly dead. 0 disables (fire whenever possible).
    reserve_last_ammo_frac: float = 0.0


@dataclass(frozen=True)
class CombatHint:
    """What the movement layer wants from the gun this tick (rl/tactics.py).

    hold_fire   : hiding in a bush for an ambush, or regenerating with no one
                  near -- a shot would give the position away or reset regen.
    allow_boxes : False while fleeing, so the agent does not stop to plink a
                  box with an enemy on its heels.
    """
    hold_fire: bool = False
    allow_boxes: bool = True


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
    # What the shot is aimed at, for the trace and for kill attribution.
    target: Optional[Tuple[float, float]] = None
    target_kind: Optional[str] = None     # "enemy" | "box" | None
    target_id: Optional[int] = None       # tracker id, when tracked

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
        self._tiles_across: Optional[float] = None   # measured override
        self.reset()

    def reset(self) -> None:
        self._super_cooldown = 0
        self._attack_cooldown = 0
        self.last_reason = "idle"

    # ------------------------------------------------------------------ #
    def range_px(self, frame_size=None, span_px: Optional[float] = None) -> float:
        """This brawler's attack reach in frame pixels.

        `span_px` is the play rect's WIDTH -- the extent `tiles_across_screen`
        is measured over. It defaults to the frame width (the play rect fills
        the frame on a phone with no letterbox).

        Falls back to the global `max_range_frac` of the short side whenever
        the per-brawler range is unknown -- a brand-new brawler with no
        published range must not end up gated at zero.
        """
        cfg = self.config
        w, h = frame_size or (1280, 720)
        short = float(max(1, min(w, h)))
        ceiling = cfg.max_range_frac * short
        tiles = getattr(self.ability, "gate_range_tiles", None)
        across = self._tiles_across or cfg.tiles_across_screen
        if not tiles or across <= 0:
            return ceiling
        span = float(span_px) if span_px else float(max(w, h))
        px = float(tiles) * cfg.range_safety * span / float(across)
        return min(ceiling, px) if cfg.range_gate_is_ceiling else px

    def range_frac(self, frame_size=None, span_px: Optional[float] = None) -> float:
        """`range_px` as a fraction of the frame's SHORT side (the unit
        `decide()` measures distance in). Defaults to a 1280x720 frame."""
        w, h = frame_size or (1280, 720)
        return self.range_px((w, h), span_px) / float(max(1, min(w, h)))

    def super_range_px(self, frame_size=None, span_px: Optional[float] = None) -> float:
        """Reach of the super. Most supers carry about as far as the main
        attack; spend them a little inside it, where they are surer to land.
        Unknown brawler: the old global `super_range_frac`."""
        cfg = self.config
        w, h = frame_size or (1280, 720)
        short = float(max(1, min(w, h)))
        if getattr(self.ability, "gate_range_tiles", None):
            return 0.85 * self.range_px((w, h), span_px)
        return cfg.super_range_frac * short

    def calibrate(self, tiles_across_screen: Optional[float]) -> None:
        """Adopt a MEASURED tiles-across-the-screen (e.g. from the localiser).

        Rejects implausible values rather than trusting them: a units slip in
        the measurement would otherwise poison every range at once.
        """
        if tiles_across_screen is None:
            self._tiles_across = None
            return
        v = float(tiles_across_screen)
        if 9.0 <= v <= 30.0:
            self._tiles_across = v

    @property
    def aim_mode(self) -> str:
        return getattr(self.ability, "aim", "auto") or "auto"

    # ------------------------------------------------------------------ #
    def walls_block(self) -> bool:
        """Should a wall between us and a target hold fire for this brawler?"""
        cfg = self.config
        if cfg.wall_blocks_shots is not None:
            return bool(cfg.wall_blocks_shots)
        if self.ability is None:
            return False                  # unknown brawler: exactly as before
        return not getattr(self.ability, "lobs_over_walls", False)

    def decide(self, state, frame_size=None, world=None, tracker=None,
               hint: Optional[CombatHint] = None) -> "CombatDecision":
        """-> CombatDecision (iterates as (fire_attack, fire_super)).

        The three gates are deliberately ASYMMETRIC about uncertainty, and the
        asymmetry is the interesting part:

          * a target   — REQUIRED. Auto-aim has nothing to aim at otherwise,
            so the tap cannot possibly do anything. Enemies first; power-cube
            boxes when no enemy is in range.
          * ammo       — only enforced when the reading is TRUSTED
            (`ammo_known`). A failed read also reports 0, and the bar is only
            located on a minority of real frames, so trusting the count alone
            would block nearly every attack. Firing on an unknown clip costs a
            wasted tap; refusing to fire on a full one costs the fight.
          * super      — ALWAYS enforced. It comes from a fixed HUD ROI that
            is never occluded, so 0.0 means uncharged rather than unknown.

        `tracker` (rl/enemy_tracker.EnemyTracker) is optional. With it, targets
        are taken from tracks -- positions extrapolated between perception
        refreshes, velocities for leading, health trends for target choice.
        Without it, this behaves as it always did on the raw detections.
        """
        cfg = self.config
        if self._super_cooldown > 0:
            self._super_cooldown -= 1
        if self._attack_cooldown > 0:
            self._attack_cooldown -= 1

        def _no(reason):
            self.last_reason = reason
            return CombatDecision(reason=reason, mode=self.aim_mode)

        if state is None:
            return _no("no state")
        player = getattr(state, "player_pos", None)
        fs = frame_size or getattr(state, "frame_size", None)
        if player is None and fs and (getattr(state, "enemy_positions", None)
                                      or getattr(state, "box_positions", None)):
            # No anchor yet, but the camera keeps the player at the middle of
            # the frame. Refusing to shoot back for want of a ring detection
            # is the worse error.
            player = (fs[0] / 2.0, fs[1] / 2.0)
        if player is None:
            return _no("no target")
        if hint is not None and hint.hold_fire:
            return _no("holding fire")

        w, h = frame_size or getattr(state, "frame_size", (1280, 720))
        short = max(1.0, min(w, h))
        px, py = float(player[0]), float(player[1])

        if cfg.hold_fire_in_bush and world is not None and _in_bush(world, player,
                                                                   cfg.bush_conceal_frac):
            return _no("concealed")

        span = (float(world.rect[2]) if world is not None and getattr(world, "rect", None)
                else None)
        gate_px = self.range_px((w, h), span)
        gate = gate_px / short
        super_px = self.super_range_px((w, h), span)
        check_walls = self.walls_block() and world is not None

        enemies = self._enemy_targets(state, tracker, px, py)
        boxes = self._box_targets(state, tracker, px, py)
        if not enemies and not boxes:
            return _no("no target")

        def los_ok(t):
            if not check_walls:
                return True
            return not _walled(world, (px, py), t.pos, cfg.los_min_wall_cells)

        # What the game's own auto-aim would hit with a tap: the nearest enemy
        # in range, else the nearest box in range. Walls are not its concern.
        in_range_e = [t for t in enemies if t.dist <= gate_px]
        in_range_b = [t for t in boxes if t.dist <= gate_px]
        auto_target = (min(in_range_e, key=lambda t: t.dist) if in_range_e
                       else (min(in_range_b, key=lambda t: t.dist) if in_range_b else None))

        chosen = None
        shootable_e = [t for t in in_range_e if los_ok(t)]
        if shootable_e:
            if cfg.smart_targeting:
                chosen = min(shootable_e, key=lambda t: self._score(t, gate_px))
            else:
                chosen = min(shootable_e, key=lambda t: t.dist)
        elif (cfg.shoot_boxes and (hint is None or hint.allow_boxes)
              and not in_range_e):
            # Boxes only when NO enemy is in range: with one in range, the
            # bullet belongs to him even if a wall is in the way this tick.
            shootable_b = [t for t in in_range_b if los_ok(t)]
            if shootable_b:
                chosen = min(shootable_b, key=lambda t: t.dist)

        if chosen is None:
            if enemies:
                nearest = min(enemies, key=lambda t: t.dist)
                dist = nearest.dist / short
                if nearest.dist <= gate_px:
                    return _no(f"walled ({dist:.2f})")
                return _no(f"out of range ({dist:.2f} > {gate:.2f})")
            return _no("no target")

        dist = chosen.dist / short
        attack = True
        if getattr(state, "ammo_known", False) and getattr(state, "ammo_count", 0) < 1:
            attack = False
            self.last_reason = "empty clip"
        elif (attack and cfg.reserve_last_ammo_frac > 0
              and getattr(state, "ammo_known", False)
              and getattr(state, "ammo_count", 0) == 1
              and chosen.dist > cfg.reserve_last_ammo_frac * gate_px
              and (chosen.hp_frac is None or chosen.hp_frac > 0.3)):
            attack = False
            self.last_reason = f"saving last round ({dist:.2f})"
        elif self._attack_cooldown > 0:
            attack = False
            self.last_reason = f"between shots ({dist:.2f})"
        else:
            kind = "box" if chosen.kind == "box" else "target"
            self.last_reason = f"{kind} at {dist:.2f}"
        if attack and cfg.attack_interval_s > 0 and cfg.tick_seconds > 0:
            self._attack_cooldown = max(0, int(round(cfg.attack_interval_s
                                                     / cfg.tick_seconds)) - 1)

        charge = getattr(state, "super_charge", None) or 0.0
        fire_super = (chosen.kind == "enemy"
                      and charge >= cfg.super_ready
                      and chosen.dist <= super_px
                      and self._super_cooldown <= 0)
        if fire_super:
            self._super_cooldown = cfg.super_cooldown_ticks
            self.last_reason += " +SUPER"

        if not (attack or fire_super):
            return CombatDecision(mode=self.aim_mode, reason=self.last_reason,
                                  target=chosen.pos, target_kind=chosen.kind,
                                  target_id=chosen.id)

        # --- where to point it ------------------------------------------- #
        mode = self.aim_mode
        retarget = False
        if mode != "manual" and auto_target is not None and chosen is not auto_target:
            # Auto-aim would hit something else. Aim by hand only if the
            # chosen target is clearly better, or the auto target is walled.
            walled_auto = not los_ok(auto_target)
            better = (self._score(auto_target, gate_px)
                      - self._score(chosen, gate_px)) >= cfg.switch_margin
            if walled_auto or better or chosen.kind != auto_target.kind:
                retarget = True
        lead = False
        if (mode != "manual" and not retarget and cfg.lead_auto_aimers
                and cfg.lead_shots and chosen.kind == "enemy"
                and chosen.dist >= cfg.lead_min_range_frac * gate_px
                and hypot(*chosen.vel) >= cfg.lead_min_speed_frac * short):
            lead = True
        aim = None
        aim_frac = 1.0
        if mode == "manual" or retarget or lead:
            ax, ay = self._aim_point(chosen, px, py, short, world, frame_size=(w, h))
            dx, dy = ax - px, ay - py
            raw = hypot(dx, dy)
            if raw > 1e-6:
                aim = (dx / raw, dy / raw)
                if getattr(self.ability, "lobs_over_walls", False):
                    aim_frac = max(0.15, min(1.0, raw / max(gate_px, 1e-6)))
                mode = "manual"
                self.last_reason += (" (retarget)" if retarget else
                                     " (lead)" if lead else " (aimed)")
        return CombatDecision(attack=bool(attack), super=bool(fire_super),
                              aim=aim, aim_frac=aim_frac, mode=mode,
                              reason=self.last_reason, target=chosen.pos,
                              target_kind=chosen.kind, target_id=chosen.id)

    # ------------------------------------------------------------------ #
    def _score(self, t: "_Target", gate_px: float) -> float:
        cfg = self.config
        s = t.dist / max(gate_px, 1e-6)
        if t.hp_frac is not None:
            s += cfg.hp_weight * t.hp_frac
        else:
            s += cfg.hp_weight * 0.75          # unknown: assume fairly healthy
        if t.hurt:
            s -= cfg.hurt_bonus
        return s

    def _enemy_targets(self, state, tracker, px, py) -> List["_Target"]:
        cfg = self.config
        out: List[_Target] = []
        if tracker is not None:
            for tr in tracker.enemies():
                if tr.age(tracker.tick) > cfg.max_target_age:
                    continue
                sx, sy = tracker.screen_pos(tr)
                out.append(_Target("enemy", (sx, sy), hypot(sx - px, sy - py),
                                   tr.hp_frac(), tracker.in_fight(tr), tr.vel, tr.id))
            if out:
                return out
            # No live tracks: either nothing is there (then the detections
            # below are empty too) or the tracker was never fed this state.
            # Falling back keeps a mis-wired tracker from silencing the gun.
        for e in (state.enemy_list() if hasattr(state, "enemy_list")
                  else [_Pos(p) for p in getattr(state, "enemy_positions", ()) or ()]):
            ex, ey = float(e.pos[0]), float(e.pos[1])
            out.append(_Target("enemy", (ex, ey), hypot(ex - px, ey - py),
                               None, False, (0.0, 0.0), None))
        return out

    def _box_targets(self, state, tracker, px, py) -> List["_Target"]:
        if not self.config.shoot_boxes:
            return []
        out: List[_Target] = []
        if tracker is not None:
            for tr in tracker.boxes():
                if tr.age(tracker.tick) > 15:
                    continue
                sx, sy = tracker.screen_pos(tr)
                out.append(_Target("box", (sx, sy), hypot(sx - px, sy - py),
                                   None, False, (0.0, 0.0), tr.id))
            if out:
                return out
        for b in (state.box_list() if hasattr(state, "box_list")
                  else [_Pos(p) for p in getattr(state, "box_positions", ()) or ()]):
            bx, by = float(b.pos[0]), float(b.pos[1])
            out.append(_Target("box", (bx, by), hypot(bx - px, by - py),
                               None, False, (0.0, 0.0), None))
        return out

    def _aim_point(self, t: "_Target", px, py, short, world, frame_size):
        """Where to point an aimed shot: the target, led by its velocity."""
        cfg = self.config
        x, y = t.pos
        vx, vy = t.vel[0] * cfg.lead_gain, t.vel[1] * cfg.lead_gain
        if not cfg.lead_shots or (vx == 0.0 and vy == 0.0) or t.kind != "enemy":
            return (x, y)
        # Where the target will be when the touch actually lands.
        x += vx * cfg.latency_ticks
        y += vy * cfg.latency_ticks
        if world is not None and getattr(world, "rect", None):
            span = float(world.rect[2])
        else:
            span = float(frame_size[0])
        px_per_tile = span / max(1e-6, self._tiles_across or cfg.tiles_across_screen)
        if getattr(self.ability, "lobs_over_walls", False):
            lead_ticks = cfg.lob_flight_s / max(1e-6, cfg.tick_seconds)
            ax, ay = x + vx * lead_ticks, y + vy * lead_ticks
        else:
            speed = cfg.projectile_tiles_per_s * px_per_tile * cfg.tick_seconds
            ax, ay = x, y
            for _ in range(3):        # fixed-point intercept: converges fast
                t_hit = hypot(ax - px, ay - py) / max(speed, 1e-6)
                ax, ay = x + vx * t_hit, y + vy * t_hit
        lead = hypot(ax - x, ay - y)
        cap = cfg.max_lead_frac * short
        if lead > cap > 0:
            ax = x + (ax - x) * cap / lead
            ay = y + (ay - y) * cap / lead
        return (ax, ay)


@dataclass(frozen=True)
class _Target:
    kind: str
    pos: Tuple[float, float]
    dist: float
    hp_frac: Optional[float]
    hurt: bool
    vel: Tuple[float, float]
    id: Optional[int]


@dataclass(frozen=True)
class _Pos:
    pos: Tuple[float, float]


def _walled(world, a, b, min_cells: int) -> bool:
    """Does the a->b line cross at least `min_cells` observed wall cells?

    Fails OPEN, like rl/line_of_sight.line_blocked: no map, no walls observed,
    or an endpoint off the map all read as clear.
    """
    try:
        from .line_of_sight import line_cells
        cells = line_cells(world, a, b)
    except Exception:
        return False
    if not cells or len(cells) <= 2:
        return False
    occ = world.occupancy
    gh, gw = occ.shape
    n = 0
    for cx, cy in cells[1:-1]:
        if 0 <= cx < gw and 0 <= cy < gh and occ[cy, cx]:
            n += 1
            if n >= max(1, min_cells):
                return True
    return False


def _in_bush(world, player_px, threshold: float) -> bool:
    """Is the player standing in cover, per the fused world map?"""
    try:
        gx, gy = world.clamp_cell(world.to_cell(player_px))
        return float(world.bush[gy, gx]) >= threshold
    except Exception:
        return False
