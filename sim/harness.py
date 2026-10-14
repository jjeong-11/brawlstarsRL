"""
sim/harness.py
==============

Turns an `Arena` into exactly what the bot's perception would have reported,
and runs the bot's REAL decision stack on it.

    Arena (tiles)  --SimView-->  GameState + terrain grid + gas grid + camera delta
                                   (frame pixels, 48x27 cells, HUD holes, stale
                                    entity lists, failed reads)
                   --AgentStack--> WaypointPlanner / EnemyTracker / CombatPolicy /
                                   TacticalBrain (or a policy)
                   <-- joystick vector + CombatDecision

The perception model is deliberately pessimistic in the ways the real stack is
known to be: the anchor is lost on some ticks, HP and ammo reads fail, enemies
are refreshed only every 0.5s and re-served stale in between (with
`entities_fresh` set exactly as `perception/liveLoop.py` sets it), the HUD hides
part of the terrain grid, and the camera delta is noisy. A decision layer that
only works on perfect inputs is not worth measuring.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from typing import Callable, List, Optional, Tuple

import numpy as np

from rl.state import EntityObs, GameState

from .arena import Arena

FRAME = (2424, 1080)                 # the project's phone
TILES_ACROSS = 17.0                  # matches CombatConfig.tiles_across_screen
GRID = (48, 27)                      # matches perception.getTerrain


@dataclass(frozen=True)
class PerceptionModel:
    anchor_miss: float = 0.12
    anchor_jitter_px: float = 5.0
    hp_read: float = 0.8
    ammo_read: float = 0.5
    entity_interval_s: float = 0.2   # liveLoop.STAGE_INTERVALS["entities"]
    pickup_interval_s: float = 0.2
    player_interval_s: float = 0.3
    game_state_interval_s: float = 1.0
    enemy_miss: float = 0.08
    entity_jitter_px: float = 8.0
    enemy_hp_read: float = 0.85
    enemy_cubes_read: float = 0.5
    camera_noise_px: float = 1.5
    hud_unknown: bool = True

    @staticmethod
    def perfect() -> "PerceptionModel":
        return PerceptionModel(anchor_miss=0.0, anchor_jitter_px=0.0, hp_read=1.0,
                               ammo_read=1.0, entity_interval_s=0.0,
                               pickup_interval_s=0.0, player_interval_s=0.0,
                               game_state_interval_s=0.0, enemy_miss=0.0,
                               entity_jitter_px=0.0, enemy_hp_read=1.0,
                               enemy_cubes_read=1.0, camera_noise_px=0.0)


class SimView:
    """Renders the arena around one brawler into perception outputs."""

    def __init__(self, arena: Arena, idx: int = 0,
                 model: Optional[PerceptionModel] = None, seed: int = 0,
                 frame=FRAME, tiles_across: float = TILES_ACROSS):
        self.arena = arena
        self.idx = idx
        self.model = model or PerceptionModel()
        self.rng = random.Random(seed * 7919 + 17)
        self.frame = frame
        self.ppt = frame[0] / float(tiles_across)
        self.cell = (frame[0] / GRID[0], frame[1] / GRID[1])
        self.rect = (0, 0, frame[0], frame[1])
        self._last_pos: Optional[Tuple[float, float]] = None
        self._t_entities = -1e9
        self._t_pickups = -1e9
        self._t_player = -1e9
        self._t_gs = -1e9
        self._ents = ([], [], [])          # enemies, boxes (EntityObs)
        self._ents_pos = ([], [])
        self._cubes: List[Tuple[int, int]] = []
        self._hp = None
        self._ammo = (0, False)
        self._players_left = None
        self._hud = self._hud_cells() if self.model.hud_unknown else np.zeros(
            (GRID[1], GRID[0]), bool)
        cxs = (np.arange(GRID[0]) + 0.5) * self.cell[0]
        cys = (np.arange(GRID[1]) + 0.5) * self.cell[1]
        self._cell_dx = (cxs - frame[0] / 2.0) / self.ppt     # tiles, per column
        self._cell_dy = (cys - frame[1] / 2.0) / self.ppt     # tiles, per row

    def _hud_cells(self) -> np.ndarray:
        try:
            import cv2
            from perception.getTerrain import _hud_mask
            w, h = self.frame
            hud = _hud_mask(h, w).astype(np.float32) / 255.0
            frac = cv2.resize(hud, GRID, interpolation=cv2.INTER_AREA)
            return frac > 0.75
        except Exception:
            return np.zeros((GRID[1], GRID[0]), bool)

    # ------------------------------------------------------------------ #
    def to_screen(self, x: float, y: float) -> Tuple[float, float]:
        me = self.arena.brawlers[self.idx]
        return (self.frame[0] / 2.0 + (x - me.x) * self.ppt,
                self.frame[1] / 2.0 + (y - me.y) * self.ppt)

    def on_screen(self, x: float, y: float, margin: float = 0.0) -> bool:
        sx, sy = self.to_screen(x, y)
        w, h = self.frame
        return -margin <= sx <= w + margin and -margin <= sy <= h + margin

    def observe(self) -> Tuple[GameState, dict, np.ndarray, Tuple[float, float]]:
        a = self.arena
        m = self.model
        me = a.brawlers[self.idx]
        w, h = self.frame
        t = a.t

        # --- camera ---------------------------------------------------- #
        if self._last_pos is None:
            cam = (0.0, 0.0)
        else:
            cam = (-(me.x - self._last_pos[0]) * self.ppt + self.rng.gauss(0, m.camera_noise_px),
                   -(me.y - self._last_pos[1]) * self.ppt + self.rng.gauss(0, m.camera_noise_px))
        self._last_pos = (me.x, me.y)

        # --- terrain + gas grid ----------------------------------------- #
        tx = me.x + self._cell_dx                     # (48,)
        ty = me.y + self._cell_dy                     # (27,)
        gx = np.floor(tx).astype(np.int64)
        gy = np.floor(ty).astype(np.int64)
        inside_x = (gx >= 0) & (gx < a.gw)
        inside_y = (gy >= 0) & (gy < a.gh)
        gxc = np.clip(gx, 0, a.gw - 1)
        gyc = np.clip(gy, 0, a.gh - 1)
        wall = a.wall[np.ix_(gyc, gxc)] | (a.box_at[np.ix_(gyc, gxc)] >= 0)
        outside = ~(inside_y[:, None] & inside_x[None, :])
        occ = wall | outside
        bush = a.bush[np.ix_(gyc, gxc)] & ~outside
        cheb = np.maximum(np.abs(tx[None, :] + 0.0 - a.cx), np.abs(ty[:, None] - a.cy))
        # The cloud is drawn over the arena, not over the void beyond it.
        inb = a.in_bounds[np.ix_(gyc, gxc)] & ~outside
        gas = ((cheb > a.gas_half()) & inb).astype(np.float32)
        unknown = self._hud.copy()
        # Like find_terrain: the player's own cell is walkable; the other
        # eight under the sprite are unknown (not evidence either way).
        pcx, pcy = GRID[0] // 2, GRID[1] // 2
        nb = np.zeros_like(unknown)
        nb[max(0, pcy - 1):pcy + 2, max(0, pcx - 1):pcx + 2] = True
        nb &= ~outside
        nb[pcy, pcx] = False
        unknown |= nb
        occ = occ & ~unknown
        occ[pcy, pcx] = False
        terrain = {"occupancy": occ, "bush": bush.astype(np.float32),
                   "unknown": unknown, "rect": self.rect, "cell": self.cell,
                   "gas": gas}

        # --- player HUD ----------------------------------------------------- #
        if t - self._t_player >= m.player_interval_s - 1e-9:
            self._t_player = t
            if self.rng.random() < m.hp_read:
                self._hp = int(me.hp)
            ammo_known = self.rng.random() < m.ammo_read
            self._ammo = (int(math.floor(me.ammo + 1e-6)) if ammo_known else self._ammo[0],
                          ammo_known)
        # liveLoop COASTS the last anchor through a failed detection (it only
        # clears it between matches), so a miss is a stale position flagged
        # not-fresh -- not a None. None only happens before the first lock.
        fresh_anchor = self.rng.random() >= m.anchor_miss
        if fresh_anchor:
            self._anchor = (int(w / 2 + self.rng.gauss(0, m.anchor_jitter_px)),
                            int(h / 2 + self.rng.gauss(0, m.anchor_jitter_px)))
        anchor = getattr(self, "_anchor", None)

        # --- entities (stale between refreshes, like liveLoop) -------------- #
        fresh = False
        if t - self._t_entities >= m.entity_interval_s - 1e-9:
            self._t_entities = t
            fresh = True
            enemies, boxes = [], []
            for o in a.brawlers:
                if o is me or not o.alive:
                    continue
                if not self.on_screen(o.x, o.y) or a.hidden(o, me.x, me.y):
                    continue
                if self.rng.random() < m.enemy_miss:
                    continue
                sx, sy = self.to_screen(o.x, o.y)
                sx += self.rng.gauss(0, m.entity_jitter_px)
                sy += self.rng.gauss(0, m.entity_jitter_px)
                hp = int(o.hp) if self.rng.random() < m.enemy_hp_read else None
                cubes = o.cubes if self.rng.random() < m.enemy_cubes_read else None
                enemies.append(EntityObs((int(sx), int(sy)), hp, cubes))
            for bx in a.boxes:
                if not bx.alive or not self.on_screen(bx.x, bx.y):
                    continue
                sx, sy = self.to_screen(bx.x, bx.y)
                boxes.append(EntityObs((int(sx), int(sy)), int(bx.hp), None))
            self._ents = (enemies, boxes, None)
        pickups_fresh = False
        if t - self._t_pickups >= m.pickup_interval_s - 1e-9:
            self._t_pickups = t
            pickups_fresh = True
            self._cubes = []
            for c in a.cubes:
                if c.alive and self.on_screen(c.x, c.y):
                    sx, sy = self.to_screen(c.x, c.y)
                    self._cubes.append((int(sx), int(sy)))
        enemies, boxes, _ = self._ents

        if t - self._t_gs >= m.game_state_interval_s - 1e-9:
            self._t_gs = t
            self._players_left = sum(1 for b in a.brawlers if b.alive)

        # --- gas summary (getGas semantics) --------------------------------- #
        in_gas = a.in_gas(me.x, me.y)
        pc_x, pc_y = GRID[0] // 2, GRID[1] // 2
        sides = (float(gas[:, :pc_x].mean()), float(gas[:, pc_x:].mean()),
                 float(gas[:pc_y, :].mean()), float(gas[pc_y:, :].mean()))
        safe = (0.0, 0.0)
        if gas.any():
            ys, xs = np.nonzero(gas)
            dx, dy = pc_x - xs.mean(), pc_y - ys.mean()
            n = math.hypot(dx, dy)
            if n > 1e-6:
                safe = (dx / n, dy / n)

        alive = me.alive
        won = alive and sum(1 for b in a.brawlers if b.alive) == 1
        state = GameState(
            tick=int(round(t * 10)),
            health=self._hp if alive else None,
            cube_count=me.cubes,
            ammo_count=self._ammo[0], ammo_known=self._ammo[1],
            player_pos=anchor if alive else None,
            anchor_source=("digits" if fresh_anchor else "ring_unverified") if anchor else "none",
            anchor_fresh=bool(anchor is not None and fresh_anchor),
            enemy_positions=[e.pos for e in enemies],
            enemies=list(enemies),
            n_boxes=len(boxes), box_positions=[b.pos for b in boxes],
            boxes=list(boxes),
            n_ground_cubes=len(self._cubes),
            ground_cube_positions=list(self._cubes),
            entities_fresh=fresh, pickups_fresh=pickups_fresh,
            in_gas=in_gas, gas_sides=sides, gas_safe=safe,
            players_left=self._players_left,
            frame_size=self.frame,
            is_alive=alive, match_over=won or not alive, won=won,
            final_rank=me.rank,
            super_charge=float(me.super_charge),
        )
        return state, terrain, gas, cam


# --------------------------------------------------------------------------- #
CONTROLLERS = ("tactical", "random", "legacy", "idle", "policy", "bot")


def legacy_combat_config(frame=FRAME):
    """CombatConfig reproducing the pre-upgrade gun exactly: the aspect-ratio
    units bug + 0.42 ceiling, no boxes, nearest target, no walls, no lead."""
    from rl.combat import CombatConfig
    w, h = frame
    return CombatConfig(range_gate_is_ceiling=True, range_safety=1.0,
                        tiles_across_screen=TILES_ACROSS * w / float(min(w, h)),
                        shoot_boxes=False, smart_targeting=False, lead_shots=False,
                        wall_blocks_shots=False, attack_interval_s=0.0)


class AgentStack:
    """The bot's decision layer, as it runs live, minus screen and touch."""

    def __init__(self, controller: str, ability=None, seed: int = 0,
                 combat_config=None, tactics_config=None,
                 policy: Optional[Callable] = None, localize: bool = True,
                 tick_seconds: float = 0.1):
        from rl.combat import CombatConfig, CombatPolicy
        from rl.enemy_tracker import EnemyTracker
        from rl.path_planner import WaypointPlanner
        from rl.tactics import TacticalBrain, TacticsConfig

        if controller not in CONTROLLERS:
            raise ValueError(f"controller must be one of {CONTROLLERS}")
        self.controller = controller
        self.rng = random.Random(seed * 31 + 7)
        if combat_config is None:
            combat_config = (legacy_combat_config() if controller == "legacy"
                             else CombatConfig(tick_seconds=tick_seconds))
        self.combat = CombatPolicy(combat_config, ability=ability)
        self.tracker = None if controller == "legacy" else EnemyTracker()
        self.planner = WaypointPlanner(localize=localize)
        tcfg = tactics_config or TacticsConfig(seed=seed, tick_seconds=tick_seconds)
        self.brain = TacticalBrain(tcfg, ability=ability, combat=self.combat)
        self.policy = policy
        self._action = [0, 0]
        self._prev_raw = None
        self._max_hp = None
        self.last_goal = None
        self.last_decision = None
        self.history = []

    def act(self, state, terrain, gas, cam):
        frame = state.frame_size
        if self.tracker is not None:
            self.tracker.update(state, cam)
        hint = None
        if self.controller == "tactical":
            goal = self.brain.decide(state, self.planner, self.tracker)
            self.last_goal = goal
            move = self.planner.plan_goal(goal.point, state, frame, terrain=terrain,
                                          camera_delta=cam, gas_grid=gas,
                                          avoid=goal.avoid, precise=goal.precise,
                                          pull_in=goal.pull_in)
            hint = goal.hint
        elif self.controller == "idle":
            move = self.planner.plan_goal(None, state, frame, terrain=terrain,
                                          camera_delta=cam, gas_grid=gas)
        else:
            if self.controller == "policy" and self.policy is not None:
                # Encoded exactly as rl/env.py does live: velocity from the
                # camera, the previous action, the planner's commitment.
                from rl.env import encode_observation
                if state.health:
                    self._max_hp = max(self._max_hp or state.health, state.health)
                obs = encode_observation(state, self._max_hp or 4000.0, frame,
                                         velocity=(-cam[0], -cam[1]),
                                         prev_action=self._prev_raw,
                                         planner_status=self.planner.status())
                action = self.policy(obs)
            else:
                action = [self.rng.randrange(16), self.rng.randrange(3)]
            self._action = action
            move = self.planner.plan(int(action[0]), int(action[1]), state, frame,
                                     terrain=terrain, camera_delta=cam, gas_grid=gas)
        decision = self.combat.decide(state, frame_size=frame, world=self.planner.world,
                                      tracker=self.tracker, hint=hint)
        self.last_decision = decision
        self._prev_raw = (int(self._action[0]), int(self._action[1]),
                          int(bool(decision.attack)), int(bool(decision.super)))
        return move, decision


_PLAYABLE = None


def playable_maps():
    """The arenas whose extraction is good enough to simulate (cached)."""
    global _PLAYABLE
    if _PLAYABLE is None:
        from perception.mapdb import database
        from .arena import playable
        _PLAYABLE = [gm for gm in database() if playable(gm)]
    return _PLAYABLE


_BUSH = {}


def build_match(seed: int, brawler: str = "shelly", map_name: Optional[str] = None,
                bot_skill: Tuple[float, float] = (0.25, 1.0)):
    """(arena, bots, ability, game_map) for a seeded match.

    Shared by `run_match` and `sim.gym_env`, so a policy pretrained in the gym
    env and a controller scored by `sim.evaluate` face identical matches.
    """
    from brawlers.abilities import default_abilities
    from perception.mapdb import database

    from .arena import Arena, bush_mask, stats_for
    from .bots import Bot, BotSkill

    rng = random.Random(seed)
    db = database()
    maps = playable_maps()
    gm = db.get(map_name) if map_name else maps[rng.randrange(len(maps))]
    if gm is None:
        raise ValueError(f"unknown map {map_name!r}")
    book = default_abilities()
    ability = book.get(brawler)
    pool = [a for a in book.all() if a.range_tiles]
    stats = [stats_for(ability)] + [stats_for(rng.choice(pool)) for _ in range(9)]
    if gm.name not in _BUSH:
        _BUSH[gm.name] = bush_mask(gm)
    arena = Arena(gm, _BUSH[gm.name], stats, seed=seed)
    bots = [Bot(arena, i, BotSkill.sample(rng, *bot_skill), random.Random(seed * 101 + i))
            for i in range(1, len(stats))]
    return arena, bots, ability, gm


@dataclass
class MatchResult:
    map: str
    seed: int
    controller: str
    brawler: str
    rank: int
    won: bool
    kills: int
    cubes: int
    boxes: int
    damage: float
    survival_s: float
    death_cause: Optional[str]
    gas_s: float
    shots: int
    hits: int
    box_hits: int = 0
    damage_taken: float = 0.0
    modes: dict = field(default_factory=dict)
    # The last few seconds before the match ended for us: (t, mode, reason,
    # hp, enemies visible). Read it to find out WHY a death happened.
    last_moments: list = field(default_factory=list)

    def as_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


def run_match(controller: str = "tactical", brawler: str = "shelly", seed: int = 0,
              map_name: Optional[str] = None, perception: Optional[PerceptionModel] = None,
              dt: float = 0.1, action_delay: int = 1, localize: bool = True,
              combat_config=None, tactics_config=None, policy=None,
              bot_skill: Tuple[float, float] = (0.25, 1.0),
              stop_on_death: bool = True, on_tick: Optional[Callable] = None
              ) -> MatchResult:
    """Play one seeded match. Same seed => same map, spawns, bots and boxes."""
    from .bots import Bot, BotSkill

    arena, bots, ability, gm = build_match(seed, brawler, map_name, bot_skill)
    view = SimView(arena, 0, perception or PerceptionModel(), seed=seed)
    agent = AgentStack("idle" if controller == "bot" else controller, ability=ability,
                       seed=seed, combat_config=combat_config,
                       tactics_config=tactics_config, policy=policy,
                       localize=localize and controller != "bot", tick_seconds=dt)
    # Reference arm: a top-skill scripted bot in the agent's seat, with
    # PERFECT information and no latency -- an upper bound on what the
    # scripted opponents' own strategy achieves from the same spawn.
    ref_bot = (Bot(arena, 0, BotSkill(aim_noise_deg=4.0, lead=0.85, reaction_ticks=1,
                                      aggression=1.0, retreat_hp=0.35),
                   random.Random(seed * 101)) if controller == "bot" else None)
    me = arena.brawlers[0]
    queue: List[tuple] = []
    modes: dict = {}
    import collections
    moments = collections.deque(maxlen=60)

    while not arena.over():
        state, terrain, gas, cam = view.observe()
        if ref_bot is not None:
            ref_bot.act(dt)
        elif me.alive:
            move, dec = agent.act(state, terrain, gas, cam)
            queue.append((move, dec))
            if agent.last_goal is not None:
                modes[agent.last_goal.mode] = modes.get(agent.last_goal.mode, 0) + 1
            near = sum(1 for o in arena.brawlers[1:] if o.alive and
                       math.hypot(o.x - me.x, o.y - me.y) < 9.0)
            g = agent.last_goal
            moments.append((round(arena.t, 1), g.mode if g else "-",
                            (g.reason if g else "")[:40], int(me.hp), near,
                            bool(dec.attack), dec.reason[:30]))
            if len(queue) > action_delay:
                mv, d = queue.pop(0)
                me.move = (float(mv[0]), float(mv[1]))
                if d.attack:
                    arena.fire(me, d.aim, d.aim_frac)
                if d.super:
                    arena.fire(me, d.aim, d.aim_frac, is_super=True)
        for bot in bots:
            bot.act(dt)
        arena.step(dt)
        if on_tick is not None:
            on_tick(arena, view, agent, state)
        if stop_on_death and not me.alive:
            break

    alive = [b for b in arena.brawlers if b.alive]
    rank = me.rank if me.rank is not None else (1 if me.alive and len(alive) == 1
                                                else len(alive))
    return MatchResult(
        map=gm.name, seed=seed, controller=controller, brawler=brawler,
        rank=int(rank), won=bool(me.alive and rank == 1), kills=me.kills,
        cubes=me.cubes_collected, boxes=me.boxes_broken,
        damage=round(me.damage_dealt), survival_s=round(me.death_t or arena.t, 1),
        death_cause=me.death_cause, gas_s=round(me.gas_time, 1),
        shots=me.shots, hits=me.hits, box_hits=me.box_hits, modes=modes,
        last_moments=list(moments)[::5])
