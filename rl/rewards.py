"""
rl/rewards.py
=============

The reward & punishment engine — the scalar the RL agent actually maximizes.

Implements the spec exactly:

    REWARDS                                          sign / relative size
    ---------------------------------------------    --------------------
    Collect powercube                                +      (per cube)
    Charge super (rises only when dealing damage)    +      (per charge %)
    Knock out / eliminate an enemy                   ++     (event)
    Survive a match tick while zone-safe             tiny + (per tick)
    Improve placement (a player you outlived died)   +      (per player)
    Win the match (last one standing)                ++++   (terminal)

    PUNISHMENTS
    ---------------------------------------------    --------------------
    Take damage                                      -      (per HP, scaled)
    Die                                              ---    (terminal)
    Stand in the hazard / gas zone                   -      (per tick)

Reward is defined on the transition prev -> curr, so the calculator is stateful:
call :meth:`reset` at the start of every match. It de-noises each signal before
it can move the reward (a second safety net on top of liveLoop's smoothing) and
returns a labeled breakdown for debugging and tuning.

Connections:  GameState (rl/state.py)  ->  compute()  ->  scalar for env.step()
              (env.py) / the prototype loop / Stable-Baselines3 PPO.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional

from .state import GameState


@dataclass
class RewardConfig:
    """All reward/punishment magnitudes in one place. Tune freely."""

    # --- Rewards ---
    cube_pickup: float = 1.0          # "+"    per power cube gained
    super_charge_full: float = 4.0    # "++"   per full 0->1 super charge (× delta):
                                      #        strong reward for DEALING DAMAGE, which
                                      #        also pushes the agent to engage/explore
    kill: float = 5.0                 # "++"   per enemy knocked out
    survive_tick: float = 0.09        # +    per safe, moving tick, so staying alive
                                      #      is net-POSITIVE (0.02 was too small to
                                      #      offset chip damage -> living looked bad)
    placement: float = 3.0            # "+"    per player you outlived who died
    win: float = 50.0                 # "++++" terminal, last one standing

    # Regaining health. Brawl Stars regenerates HP out of combat, so "healing"
    # is not an action the agent can take -- it is the CONSEQUENCE of breaking
    # off and staying out of trouble. Nothing previously paid for that: the
    # reward engine only looked at health LOSSES, so getting back to full was
    # worth exactly zero and retreating had no upside at all, while fighting
    # always paid super charge and kills. Measured on a real rollout: HP rose on
    # 1 step and earned +0.000.
    #
    # Deliberately EQUAL to damage_taken. Healing then refunds precisely what
    # the damage cost, which makes health a potential function: a damage-then-
    # heal cycle nets zero (slightly negative once discounted), so the agent
    # cannot farm it by deliberately getting hurt. Setting it higher would
    # create exactly that exploit.
    health_regain: float = 3.0        # "+"    per FULL health bar recovered

    # --- Punishments ---
    damage_taken: float = 3.0         # "-"    per FULL health bar lost (scaled)
    death: float = 30.0               # "---"  terminal (applied negative)
    # A clear "get out of gas" nudge, but NOT so large it dominates. At 5.0/tick it
    # made surviving into the closing-gas endgame score worse than dying early
    # (perverse), and its spikes made returns too noisy for the value function to
    # learn. The gas already hurts via damage_taken + eventual death; this is just
    # shaping. Gas also gates off the survive reward, so it's still clearly bad.
    gas_tick: float = 0.5             # "-"    per tick standing in gas

    # Cost of pulling the trigger, charged per shot ACTUALLY fired.
    #
    # Firing was free, so nothing discouraged holding it down: measured on a
    # rollout, 89% of attacks were fired with an empty clip and 100% of supers
    # were fired uncharged -- both pure no-op taps. Gating (rl/env.py) removes
    # the shots that provably cannot land; this term handles the rest, the shots
    # that are merely unlikely to.
    #
    # Scale: a hit is worth roughly +0.4 to +0.8 through super_charge, so at
    # 0.05 a shot needs only a ~10% chance of connecting to be worth taking.
    # That trims spray-and-pray without making the agent gun-shy.
    attack_cost: float = 0.05         # "-"    per attack tap
    super_cost: float = 0.10          # "-"    per super tap (a scarcer resource)

    # --- De-noising guards (map to real OCR/perception failure modes) ---
    default_max_health: float = 4000.0
    max_health_cap: int = 20000
    max_cube_jump: int = 6
    max_cubes: int = 60

    # --- Kill attribution ---
    use_kill_heuristic: bool = False
    kill_heuristic_range: float = 250.0

    # --- Potential-based shaping: move toward power-cube boxes ---
    # cube_pickup only pays out at the moment of contact, which is a very sparse
    # signal to discover by random walking. This adds a dense gradient toward
    # boxes WITHOUT changing what the optimal policy is.
    #
    # The form is Ng/Harada/Russell (1999): F = gamma*Phi(s') - Phi(s), with
    #     Phi(s) = box_shaping * (1 - normalised distance to the nearest box)
    # Potential-based shaping is provably policy-invariant, and — the property
    # that matters most here — it TELESCOPES TO ZERO around any closed loop. A
    # naive "reward getting closer" term would let the agent farm reward forever
    # by stepping toward a box and back again; this cannot, because the step
    # back refunds exactly what the step forward paid.
    box_shaping: float = 0.6
    # MUST match the discount PPO is trained with (train.py: gamma=0.995).
    # Policy invariance is only guaranteed when the two agree; a mismatch
    # reintroduces a real (small) bias toward or away from boxes.
    gamma: float = 0.995
    # Largest believable change in normalised box distance in one tick. Bigger
    # jumps mean the NEAREST BOX CHANGED IDENTITY (one was picked up, occluded,
    # or newly detected) rather than the player moving, and shaping them would
    # inject reward for a perception event. Those ticks are skipped.
    max_box_dist_jump: float = 0.15


@dataclass
class RewardResult:
    total: float = 0.0
    breakdown: Dict[str, float] = field(default_factory=dict)

    def __float__(self) -> float:
        return float(self.total)


class RewardCalculator:
    """Turns GameState transitions into scalar rewards. One instance per agent."""

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()
        self.reset()

    # ---- episode lifecycle ----
    def reset(self) -> None:
        self._prev: Optional[GameState] = None
        self._last_health: Optional[int] = None
        self._max_health: Optional[int] = None
        self._last_cubes: Optional[int] = None
        self._last_super: float = 0.0
        self._last_players_left: Optional[int] = None
        self._death_counted = False
        self._win_counted = False
        self._last_box_phi: Optional[float] = None
        self._last_box_dist: Optional[float] = None
        self.episode_return: float = 0.0
        self.episode_breakdown: Dict[str, float] = defaultdict(float)

    # ---- main entry point ----
    def compute(self, state: GameState, done: bool = False,
                allow_positive: bool = True, fired_attack: bool = False,
                fired_super: bool = False) -> RewardResult:
        """Score the transition into ``state``.

        allow_positive=False zeroes out ALL positive reward components (keeping the
        punishments). The env uses this to withhold every positive unless the agent
        has moved recently (anti-idle) — see BrawlStarsEnv.move_window_seconds.

        fired_attack / fired_super say whether a shot was actually SENT TO THE
        DEVICE this step (after the env's gating), not merely whether the policy
        asked for one. Shots the gate suppressed never happened and are not
        charged for.
        """
        cfg = self.config
        b: Dict[str, float] = {}

        health = self._sanitize_health(state.health)
        cubes = self._sanitize_cubes(state.cube_count)
        supercharge = self._sanitize_super(state.super_charge)
        players_left = self._sanitize_players_left(state.players_left)

        if health is not None:
            self._max_health = max(self._max_health or health, health)
        denom = float(self._max_health or cfg.default_max_health)

        if self._prev is None:  # first tick of the match: set baseline only
            self._remember(state, health, cubes, supercharge, players_left)
            return RewardResult(0.0, {})

        prev = self._prev

        # ===================== REWARDS =====================
        # (1) Collect powercube
        if cubes is not None and self._last_cubes is not None:
            gained = cubes - self._last_cubes
            if 0 < gained <= cfg.max_cube_jump:
                b["cube_pickup"] = cfg.cube_pickup * gained

        # (2) Charge super (proxy for dealing damage)
        if supercharge is not None:
            d = supercharge - self._last_super
            if d > 0:
                b["super_charge"] = cfg.super_charge_full * d

        # (3) Knock out an enemy
        kills = self._resolve_kills(state, prev, players_left)
        if kills > 0:
            b["kill"] = cfg.kill * kills

        # (4) Survive a tick, zone-safe (tiny; never while in gas)
        if state.is_alive and not state.in_gas and not state.match_over:
            b["survive"] = cfg.survive_tick

        # (5) Improve placement
        if players_left is not None and self._last_players_left is not None:
            passed = self._last_players_left - players_left
            if passed > 0 and state.is_alive:
                b["placement"] = cfg.placement * passed

        # (6) Win the match
        if state.won and not self._win_counted:
            b["win"] = cfg.win
            self._win_counted = True

        # (7) Regain health — the payoff for disengaging and letting regen work
        if health is not None and self._last_health is not None and state.is_alive:
            gained = health - self._last_health
            if gained > 0:
                b["health_regain"] = cfg.health_regain * min(1.0, gained / denom)

        # ===================== PUNISHMENTS =====================
        # (8) Take damage — fraction of the health bar; losses only
        if health is not None and self._last_health is not None and state.is_alive:
            lost = self._last_health - health
            if lost > 0:
                b["damage_taken"] = -cfg.damage_taken * min(1.0, lost / denom)

        # (9) Cost of firing — see RewardConfig.attack_cost
        if fired_attack and cfg.attack_cost:
            b["attack_cost"] = -cfg.attack_cost
        if fired_super and cfg.super_cost:
            b["super_cost"] = -cfg.super_cost

        # (10) Die — large terminal, once
        died = (prev.is_alive and not state.is_alive) or (
            state.match_over and not state.won and not state.is_alive)
        if died and not self._death_counted:
            b["death"] = -cfg.death
            self._death_counted = True

        # (11) Standing in gas
        if state.in_gas and state.is_alive:
            b["gas"] = -cfg.gas_tick

        # Anti-idle: if the agent hasn't moved recently, drop every positive
        # component and keep only the punishments.
        #
        # box_progress is computed AFTER this filter on purpose. Potential-based
        # shaping only stays policy-invariant if BOTH signs survive; keeping the
        # negative half while discarding the positive half would turn it into a
        # plain "never approach a box" penalty — the opposite of the intent, and
        # exactly the kind of asymmetry that makes shaping terms exploitable.
        # (Standing still leaves Phi unchanged, so the term is ~0 while idle
        # anyway and cannot be farmed by doing nothing.)
        if not allow_positive:
            b = {k: v for k, v in b.items() if v < 0}

        # (12) Shaping: progress toward the nearest power-cube box
        shaping = self._box_shaping(state, picked_up="cube_pickup" in b)
        if shaping is not None:
            b["box_progress"] = shaping

        total = float(sum(b.values()))
        self._remember(state, health, cubes, supercharge, players_left)
        self.episode_return += total
        for k, v in b.items():
            self.episode_breakdown[k] += v
        return RewardResult(total, b)

    __call__ = compute

    # ---- potential-based box shaping ----
    def _box_potential(self, state: GameState):
        """Phi(s) in [0, box_shaping]: higher the closer the nearest box is.

        None when there is no box to measure against — which is NOT the same as
        distance 0. Treating "no box visible" as a potential of zero would make
        a box appearing on screen look like free reward and a box leaving look
        like a punishment, neither of which the agent caused.
        """
        if state.player_pos is None or not state.box_positions:
            return None, None
        w, h = state.frame_size
        px, py = state.player_pos
        # Normalised by the frame diagonal, so the potential is resolution
        # independent and lands in [0, 1].
        diag = (w * w + h * h) ** 0.5
        d = min(((bx - px) ** 2 + (by - py) ** 2) ** 0.5
                for bx, by in state.box_positions) / max(diag, 1e-6)
        d = min(1.0, d)
        return self.config.box_shaping * (1.0 - d), d

    def _box_shaping(self, state: GameState, picked_up: bool):
        """F = gamma*Phi(s') - Phi(s), or None when it cannot be applied safely."""
        cfg = self.config
        phi, dist = self._box_potential(state)
        prev_phi, prev_dist = self._last_box_phi, self._last_box_dist
        self._last_box_phi, self._last_box_dist = phi, dist

        if cfg.box_shaping <= 0 or phi is None or prev_phi is None:
            return None
        # A pickup DELETES the box the potential was measured against, so Phi
        # collapses on the exact tick the agent succeeds. Paying that out would
        # be a large penalty for doing the right thing, which is precisely the
        # behaviour we are trying to encourage.
        if picked_up:
            return None
        # A discontinuous jump means the nearest box changed identity, not that
        # the player moved.
        if prev_dist is not None and dist is not None and \
                abs(dist - prev_dist) > cfg.max_box_dist_jump:
            return None
        return cfg.gamma * phi - prev_phi

    # ---- kill resolution ----
    def _resolve_kills(self, state, prev, players_left) -> int:
        if state.kills_this_tick:
            return int(state.kills_this_tick)
        if not self.config.use_kill_heuristic:
            return 0
        if players_left is None or self._last_players_left is None:
            return 0
        if self._last_players_left - players_left <= 0:
            return 0
        dealt = (state.super_charge or 0.0) > self._last_super
        near = prev.nearest_enemy()
        if near is not None and prev.player_pos is not None and dealt:
            dx = near[0] - prev.player_pos[0]
            dy = near[1] - prev.player_pos[1]
            if (dx * dx + dy * dy) ** 0.5 <= self.config.kill_heuristic_range:
                return 1
        return 0

    # ---- de-noising (each maps to a real perception failure mode) ----
    def _sanitize_health(self, v):
        if v is None:
            return self._last_health
        if v < 0 or v > self.config.max_health_cap:
            return self._last_health
        return int(v)

    def _sanitize_cubes(self, v):
        if v is None:  # liveLoop returns None when the HUD badge row is hidden
            return self._last_cubes
        if v < 0 or v > self.config.max_cubes:
            return self._last_cubes
        return int(v)

    def _sanitize_super(self, v):
        if v is None:
            return self._last_super
        return max(0.0, min(1.0, float(v)))

    def _sanitize_players_left(self, v):
        if v is None:
            return self._last_players_left
        if v < 1 or v > 10:
            return self._last_players_left
        if self._last_players_left is not None and v > self._last_players_left:
            return self._last_players_left  # nobody respawns in Showdown
        return int(v)

    def _remember(self, state, health, cubes, supercharge, players_left):
        self._prev = state
        if health is not None:
            self._last_health = health
        if cubes is not None:
            self._last_cubes = cubes
        if supercharge is not None:
            self._last_super = supercharge
        if players_left is not None:
            self._last_players_left = players_left
