"""
rl/pipeline.py
==============

One frame in, one decision out: the whole live stack behind a single call.

    agent = LiveAgent("shelly", tick_seconds=0.1)
    tick = agent.step(frame)          # perception -> world model -> brain -> combat -> route
    executor.send(tick.intent)        # or don't: observing is the same call

WHY THIS EXISTS
---------------
The per-tick sequence -- perceive, adapt the state, find terrain, scan gas,
track the camera, update enemy tracks, pick a goal, decide the shot, route --
was written out separately in the web UI, `watch_live`, the footage fitter
and the behaviour-cloning tool. Four copies drift: the web UI once omitted
the tracker's camera update and the anchor was missing from the gas scan in
three of them. It also left nowhere to measure or budget the loop as a whole.

REAL TIME
---------
`step` times every stage (`tick.timings`, seconds) and keeps running
percentiles (`agent.latency()`), so "is the loop fast enough" is a number.
Stages that do not need every frame run on their own cadence:

  * perception stages -- `perception.liveLoop.STAGE_INTERVALS` (entities and
    pickups 0.2 s, player 0.3 s, screen OCR 1 s on a worker thread);
  * gas scan -- `gas_every_s` (the cloud moves a tile every few seconds; the
    world map carries the last reading in between);
  * terrain -- `terrain_every_s`; walls do not move, the camera does, and the
    world map is registered by the camera tracker every tick.

Measured in the simulator, stale information and lag cost up to half a place
while perception noise costs nothing measurable (docs/PLAN_pro_imitation.md),
so the cheap stages run every tick and the expensive ones are rationed.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np


@dataclass
class Tick:
    """Everything one `LiveAgent.step` produced."""
    state: Any                      # rl.state.GameState
    live: dict                      # raw perception outputs
    screen: str                     # "in_match" | "loading" | "match_end" | "defeated" | "unknown"
    intent: Any                     # rl.actions.Intent (what to send to the phone)
    goal: Any = None                # rl.tactics.Goal, when the brain chose the movement
    combat: Any = None              # rl.combat.CombatDecision
    reward: Any = None              # rl.rewards.RewardResult, when rewards are on
    terrain: Optional[dict] = None
    gas: Any = None                 # this tick's gas grid, None between scans
    camera_delta: tuple = (0.0, 0.0)
    action: list = field(default_factory=lambda: [0, 0])
    policy_source: str = "tactical"
    timings: Dict[str, float] = field(default_factory=dict)   # seconds per stage

    @property
    def in_match(self) -> bool:
        return self.screen == "in_match"

    @property
    def total_s(self) -> float:
        return float(sum(self.timings.values()))


class LiveAgent:
    """Perception, world model, tactics, combat and routing for one brawler.

    `policy_stepper`: optional callable obs -> [heading, distance] (see
    rl.checkpoints.PolicyStepper); used only when `use_brain` is False.
    """

    def __init__(self, brawler: Optional[str] = "shelly", tick_seconds: float = 0.1,
                 use_brain: bool = True, policy_stepper_factory: Optional[Callable] = None,
                 policy_source: str = "none", combat_config=None, planner_config=None,
                 tactics_config=None, reward_config=None, rewards: bool = True,
                 gas_every_s: float = 0.4, terrain_every_s: float = 0.0,
                 async_game_state: bool = True, latency_window: int = 600):
        from brawlers import default_abilities
        from rl.rewards import RewardConfig
        from rl.timebase import (combat_config_for, planner_config_for,
                                 tactics_config_for, tracker_config_for)

        self.brawler = brawler
        self.tick_seconds = float(tick_seconds)
        self.ability = default_abilities().get(brawler) if brawler else None
        self.use_brain = bool(use_brain)
        self._stepper_factory = policy_stepper_factory
        self._policy_source = policy_source
        # Tick-counted settings are rescaled so they last the same number of
        # SECONDS at this loop rate as at the simulator's 0.1 s tick.
        self.combat_config = combat_config or combat_config_for(self.tick_seconds)
        self.planner_config = planner_config or planner_config_for(self.tick_seconds)
        self.tactics_config = tactics_config or tactics_config_for(self.tick_seconds)
        self.tracker_config = tracker_config_for(self.tick_seconds)
        self.reward_config = reward_config or RewardConfig()
        self.rewards = bool(rewards)
        self.gas_every_s = float(gas_every_s)
        self.terrain_every_s = float(terrain_every_s)
        self._async_gs = bool(async_game_state)
        self._lat: Dict[str, deque] = {}
        self._lat_window = int(latency_window)
        self.tick = 0
        self.new_match()

    # ------------------------------------------------------------------ #
    def new_match(self) -> None:
        """Drop everything that must not survive into the next match: the
        fused world map (planner), camera registration, the terrain profile,
        perception's temporal smoothing, tracks, and the brain's memory."""
        from perception.getTerrain import ProfileSelector
        from perception.liveLoop import LivePerception
        from rl.camera_tracker import CameraTracker
        from rl.combat import CombatPolicy
        from rl.enemy_tracker import EnemyTracker
        from rl.kills import KillAttributor
        from rl.path_planner import WaypointPlanner
        from rl.rewards import RewardCalculator
        from rl.tactics import TacticalBrain

        old = getattr(self, "perception", None)
        if old is not None and hasattr(old, "close"):
            try:
                old.close()
            except Exception:
                pass
        self.perception = LivePerception(async_game_state=self._async_gs)
        self.planner = WaypointPlanner(self.planner_config)
        self.camera = CameraTracker()
        self.profiles = ProfileSelector()
        self.combat = CombatPolicy(self.combat_config, ability=self.ability)
        self.combat.reset()
        self.calc = RewardCalculator(self.reward_config)
        self.calc.reset()
        self.killer = KillAttributor()
        self.killer.reset()
        self.tracker = EnemyTracker(self.tracker_config)
        self.brain = (TacticalBrain(self.tactics_config, ability=self.ability, combat=self.combat)
                      if self.use_brain else None)
        self.stepper = self._stepper_factory() if self._stepper_factory is not None else None
        self.prev_raw = None
        self.tick = 0
        self._last_gas = self._last_terrain = -1e9
        self._terrain = None
        self._last_anchor = None

    # ------------------------------------------------------------------ #
    def step(self, frame: np.ndarray, now: Optional[float] = None) -> Tick:
        """One decision. `now` (seconds) is the clock stages are scheduled on:
        leave it None live; pass the video time when replaying a recording."""
        from perception.getGas import gas_info, player_anchor
        from perception.getTerrain import GRID_H, GRID_W, find_terrain, play_rect
        from rl.actions import decode_action, decode_goal
        from rl.state import adapt_live_state

        pc = time.perf_counter
        clock = time.time() if now is None else float(now)
        T: Dict[str, float] = {}
        fh, fw = frame.shape[:2]

        t = pc()
        live, _ = self.perception.tick(frame, now=now)
        kills = self.killer.update(live)
        state = adapt_live_state(live, tick=self.tick, kills_this_tick=kills,
                                 frame_size=(fw, fh))
        T["perception"] = pc() - t
        self.tick += 1
        screen = (live.get("game_state") or {}).get("state", "unknown")

        reward = None
        if self.rewards:
            t = pc()
            reward = self.calc.compute(state)
            T["reward"] = pc() - t

        # --- terrain (walls / bushes on screen) --------------------------- #
        t = pc()
        terrain = None
        if state.player_pos is not None:
            if self._terrain is None or clock - self._last_terrain >= self.terrain_every_s:
                self._last_terrain = clock
                try:
                    terrain = find_terrain(frame, player_pos=state.player_pos,
                                           selector=self.profiles)
                except Exception:
                    terrain = None
                self._terrain = terrain
        rect = (terrain["rect"] if terrain
                else (self._terrain["rect"] if self._terrain else play_rect(frame)))
        T["terrain"] = pc() - t

        # --- gas ---------------------------------------------------------- #
        t = pc()
        gas = None
        # The player's own ring, HP bar and name tag pass the gas colour test,
        # and gas_info can only blank them if it knows where the player is.
        # A scan without an anchor writes phantom gas next to the player into
        # the world map, where it lingers for seconds (caught by
        # tests/test_pipeline.py: "gas 0.8t away" on a frame with no gas). So
        # use the last known anchor when this tick has none -- the sprite
        # barely moves on screen -- and do not scan at all before the first.
        anchor = player_anchor(state) or self._last_anchor
        self._last_anchor = anchor
        if anchor is not None and clock - self._last_gas >= self.gas_every_s:
            self._last_gas = clock
            try:
                gas = gas_info(frame, player_pos=anchor, grid_rect=rect,
                               grid_size=(GRID_W, GRID_H))["grid"]
            except Exception:
                gas = None
        if terrain is None and rect is not None:
            self.planner.world.set_geometry(rect, (GRID_W, GRID_H))
        T["gas"] = pc() - t

        # --- camera + tracks ---------------------------------------------- #
        t = pc()
        dx, dy, ok = self.camera.update(frame, rect)
        camera_delta = (dx, dy) if ok else (0.0, 0.0)
        self.tracker.update(state, camera_delta)
        T["camera"] = pc() - t

        # --- where to go ---------------------------------------------------- #
        t = pc()
        goal = None
        if self.brain is not None:
            goal = self.brain.decide(state, self.planner, self.tracker)
            action = [0, 0]
            source = "tactical"
        elif self.stepper is not None:
            from rl.env import encode_observation
            max_hp = self.calc._max_health or self.calc.config.default_max_health
            obs = encode_observation(state, max_hp, (fw, fh),
                                     velocity=(-camera_delta[0], -camera_delta[1]),
                                     prev_action=self.prev_raw,
                                     planner_status=self.planner.status())
            action = self.stepper(obs)
            source = self._policy_source
        else:
            # Random headings still exercise relocation, A*, clearance and gas
            # escape on real frames; they say nothing about a policy.
            action = [int(np.random.randint(16)), int(np.random.randint(3))]
            source = "none"
        T["brain"] = pc() - t

        # --- whether to shoot ----------------------------------------------- #
        t = pc()
        combat = self.combat.decide(state, frame_size=(fw, fh), world=self.planner.world,
                                    tracker=self.tracker, hint=getattr(goal, "hint", None))
        if (combat.attack or combat.super) and combat.target_kind != "box":
            self.killer.note_fired()
        T["combat"] = pc() - t

        # --- route (world map update + A*) ---------------------------------- #
        t = pc()
        kwargs = dict(state=state, frame_size=(fw, fh), planner=self.planner,
                      terrain=terrain, camera_delta=camera_delta, gas_grid=gas,
                      combat=combat)
        intent = decode_goal(goal, **kwargs) if goal is not None else decode_action(action, **kwargs)
        if goal is not None:
            action = [int(intent.raw[0]), int(intent.raw[1])]
        self.prev_raw = tuple(intent.raw)
        T["route"] = pc() - t

        for k, v in T.items():
            self._lat.setdefault(k, deque(maxlen=self._lat_window)).append(v)
        self._lat.setdefault("total", deque(maxlen=self._lat_window)).append(sum(T.values()))
        return Tick(state=state, live=live, screen=screen, intent=intent, goal=goal,
                    combat=combat, reward=reward, terrain=terrain, gas=gas,
                    camera_delta=camera_delta, action=list(action), policy_source=source,
                    timings=T)

    # ------------------------------------------------------------------ #
    def latency(self) -> Dict[str, Dict[str, float]]:
        """Per-stage p50 / p95 / max over the last `latency_window` ticks, ms."""
        out = {}
        for k, v in self._lat.items():
            a = np.sort(np.fromiter(v, float)) * 1000.0
            if a.size:
                out[k] = {"p50": float(a[a.size // 2]),
                          "p95": float(a[min(a.size - 1, int(0.95 * a.size))]),
                          "max": float(a[-1]), "n": int(a.size)}
        return out

    def explain(self, k: int = 3) -> Optional[str]:
        return self.brain.explain(k) if self.brain is not None else None
