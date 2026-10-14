"""
sim/gym_env.py
==============

`BrawlStarsEnv`'s interface -- same observation vector, same action space, same
reward function -- over the simulator instead of a phone.

    from sim.gym_env import SimShowdownEnv
    env = SimShowdownEnv(brawler="piper")
    obs, info = env.reset(seed=3)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

    python scripts/train_rl.py --sim --brawler piper        # PPO in the simulator
    python -m sim.gym_env collect --episodes 200 --out bc_sim.npz
    python tools/behaviour_clone.py train bc_sim.npz --out brawlstars_move.zip

WHY THIS EXISTS
---------------
The phone produces ~36k steps an hour; this produces that in a few minutes per
core. A policy pretrained here starts its live run already knowing to leave the
gas and break boxes, which is worth far more live samples than the sim's
inaccuracies cost -- PROVIDED the interface is identical, which is the whole
design constraint of this file:

  * observations come from `rl.env.encode_observation`, fed the same planner
    status, velocity and action history the live env computes;
  * actions go through `WaypointPlanner.plan` exactly as `rl/actions.py`
    decodes them live, and combat is the same scripted `CombatPolicy`;
  * the reward is the live `RewardCalculator` over the synthesised GameState
    (with read failures and stale entities), except that kills are credited
    EXACTLY -- the simulator knows who fired the killing shot, so the noisy
    heuristic in rl/kills.py is not needed here.

Checkpoints trained here therefore load and run live unchanged, and the
README's warning stands: the simulator is an approximation, so treat it as a
warm start for live training, not a substitute for it.

TEACHER LABELS
--------------
With `teacher=True`, every step also asks rl/tactics.TacticalBrain what IT would
do and returns that, quantised to the action space, as info["teacher_action"].
`collect` drives the episode with the brain and records (obs, teacher action)
pairs in the format tools/behaviour_clone.py trains on.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import random
import sys
from collections import deque
from typing import Optional

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM = True
except Exception:  # pragma: no cover
    gym = object  # type: ignore
    spaces = None  # type: ignore
    _GYM = False

from rl.actions import make_action_space            # noqa: E402
from rl.combat import CombatConfig, CombatPolicy     # noqa: E402
from rl.enemy_tracker import EnemyTracker            # noqa: E402
from rl.env import encode_observation, obs_dim       # noqa: E402
from rl.path_planner import WaypointPlanner          # noqa: E402
from rl.rewards import RewardCalculator, RewardConfig  # noqa: E402

from sim.harness import PerceptionModel, SimView, build_match  # noqa: E402


class SimShowdownEnv(gym.Env if _GYM else object):
    metadata = {"render_modes": []}

    def __init__(self, brawler: str = "shelly", perception: Optional[PerceptionModel] = None,
                 reward_config: Optional[RewardConfig] = None, dt: float = 0.1,
                 action_delay: int = 0, latency_ticks: int = 1,
                 teacher: bool = False, drive_with_teacher: bool = False,
                 max_seconds: float = 260.0, localize: bool = False,
                 seed: Optional[int] = None):
        self.brawler = brawler
        self.perception = perception or PerceptionModel()
        self.reward_config = reward_config or RewardConfig()
        self.dt = dt
        self.latency_ticks = max(0, int(latency_ticks))
        self.action_history_len = max(1, int(action_delay) + 1)
        self.teacher = teacher or drive_with_teacher
        self.drive_with_teacher = drive_with_teacher
        self.max_ticks = int(max_seconds / dt)
        self.localize = localize
        self._seed = seed if seed is not None else random.randrange(1 << 30)
        self._episode = 0
        if _GYM:
            self.action_space = make_action_space()
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(obs_dim(self.action_history_len),),
                dtype=np.float32)
        self.arena = None

    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        if _GYM:
            super().reset(seed=seed)
        if seed is not None:
            # An explicit seed names ONE match: same seed, same map, spawns,
            # opponents and perception noise. Without a seed, successive
            # resets walk through consecutive matches.
            self._seed = int(seed)
            self._episode = 0
        match_seed = self._seed + self._episode
        self._episode += 1
        self.arena, self.bots, self.ability, self.map = build_match(match_seed, self.brawler)
        self.view = SimView(self.arena, 0, self.perception, seed=match_seed)
        self.planner = WaypointPlanner(localize=self.localize)
        self.tracker = EnemyTracker()
        self.combat = CombatPolicy(CombatConfig(tick_seconds=self.dt), ability=self.ability)
        self.brain = None
        if self.teacher:
            from rl.tactics import TacticalBrain, TacticsConfig
            self.brain = TacticalBrain(TacticsConfig(seed=match_seed, tick_seconds=self.dt),
                                       ability=self.ability, combat=self.combat)
        self.rewards = RewardCalculator(self.reward_config)
        self.rewards.reset()
        self._queue = deque()
        self._tick = 0
        self._prev_action = None
        self._history = deque([None] * self.action_history_len,
                              maxlen=self.action_history_len)
        self._velocity = (0.0, 0.0)
        self._kills_seen = 0
        self._state, self._terrain, self._gas, self._cam = self.view.observe()
        self.tracker.update(self._state, self._cam)
        self.rewards.compute(self._state)
        return self._encode(), {"state": self._state, "map": self.map.name}

    def _encode(self):
        max_hp = self.rewards._max_health or self.reward_config.default_max_health
        return encode_observation(self._state, max_hp, self._state.frame_size,
                                  velocity=self._velocity,
                                  prev_action=self._prev_action,
                                  planner_status=self.planner.status(),
                                  action_history=list(self._history))

    # ------------------------------------------------------------------ #
    def step(self, action):
        a = self.arena
        me = a.brawlers[0]
        st = self._state
        frame = st.frame_size

        teacher_action = None
        goal = None
        if self.brain is not None:
            goal = self.brain.decide(st, self.planner, self.tracker)
            from rl.tactics import goal_to_action
            rect = self.planner.world.rect or (0, 0, frame[0], frame[1])
            teacher_action = list(goal_to_action(goal.point, st.player_pos or
                                                 (frame[0] / 2, frame[1] / 2), rect))
        decision = self.combat.decide(st, frame_size=frame, world=self.planner.world,
                                      tracker=self.tracker,
                                      hint=goal.hint if (goal and self.drive_with_teacher) else None)
        if self.drive_with_teacher and goal is not None:
            move = self.planner.plan_goal(goal.point, st, frame, terrain=self._terrain,
                                          camera_delta=self._cam, gas_grid=self._gas,
                                          avoid=goal.avoid, precise=goal.precise,
                                          pull_in=goal.pull_in)
            taken = teacher_action
        else:
            h, d = int(action[0]), int(action[1])
            move = self.planner.plan(h, d, st, frame, terrain=self._terrain,
                                     camera_delta=self._cam, gas_grid=self._gas)
            taken = [h, d]
        raw = (taken[0], taken[1], int(bool(decision.attack)), int(bool(decision.super)))
        self._prev_action = raw
        self._history.appendleft(raw)

        self._queue.append((move, decision))
        if len(self._queue) > self.latency_ticks:
            mv, dec = self._queue.popleft()
            me.move = (float(mv[0]), float(mv[1]))
            if dec.attack:
                a.fire(me, dec.aim, dec.aim_frac)
            if dec.super:
                a.fire(me, dec.aim, dec.aim_frac, is_super=True)
        for bot in self.bots:
            bot.act(self.dt)
        a.step(self.dt)
        self._tick += 1

        prev_pos = (me.x, me.y)
        self._state, self._terrain, self._gas, self._cam = self.view.observe()
        self.tracker.update(self._state, self._cam)
        ppt = self.view.ppt
        self._velocity = (-self._cam[0], -self._cam[1])

        # Exact kill credit: the arena knows who landed the killing shot.
        kills = me.kills - self._kills_seen
        self._kills_seen = me.kills
        self._state.kills_this_tick = kills
        result = self.rewards.compute(self._state, fired_attack=bool(decision.attack),
                                      fired_super=bool(decision.super))
        terminated = bool(not me.alive or a.over())
        truncated = self._tick >= self.max_ticks
        info = {"reward_breakdown": result.breakdown,
                "episode_return": self.rewards.episode_return,
                "rank": me.rank, "alive": me.alive, "map": self.map.name}
        if teacher_action is not None:
            info["teacher_action"] = teacher_action
            info["tactic"] = goal.label() if goal is not None else None
        if terminated:
            alive = [b for b in a.brawlers if b.alive]
            info["rank"] = me.rank or (1 if me.alive and len(alive) == 1 else len(alive))
        return self._encode(), float(result.total), terminated, truncated, info

    def close(self):
        pass


# --------------------------------------------------------------------------- #
def collect(episodes: int, out: str, brawler: str = "shelly", seed: int = 0,
            skip_holds: bool = True) -> int:
    """Drive episodes with the tactical brain; save (obs, teacher action) pairs
    in tools/behaviour_clone.py's format (npz with `obs` and `act`)."""
    env = SimShowdownEnv(brawler=brawler, drive_with_teacher=True, seed=seed)
    obs_rows, act_rows = [], []
    ranks = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            obs_next, _r, term, trunc, info = env.step([0, 0])
            ta = info.get("teacher_action")
            tactic = info.get("tactic") or ""
            if ta is not None and not (skip_holds and (">hold" in tactic or tactic.startswith("lost"))):
                obs_rows.append(obs)
                act_rows.append(ta)
            obs = obs_next
            done = term or trunc
        ranks.append(info.get("rank"))
        if (ep + 1) % max(1, episodes // 10) == 0:
            print(f"  {ep + 1}/{episodes} episodes, {len(obs_rows)} samples, "
                  f"mean rank {np.mean([r for r in ranks if r]):.2f}", flush=True)
    np.savez_compressed(out, obs=np.asarray(obs_rows, np.float32),
                        act=np.asarray(act_rows, np.int64))
    print(f"wrote {out}: {len(obs_rows)} samples from {episodes} episodes")
    return len(obs_rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Simulator gym env utilities")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="tactical-brain demonstrations -> BC dataset")
    c.add_argument("--episodes", type=int, default=100)
    c.add_argument("--out", default="bc_sim.npz")
    c.add_argument("--brawler", default="shelly")
    c.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("smoke", help="random actions for one episode")
    s.add_argument("--brawler", default="shelly")
    args = ap.parse_args(argv)
    if args.cmd == "collect":
        collect(args.episodes, args.out, args.brawler, args.seed)
    else:
        env = SimShowdownEnv(brawler=args.brawler)
        obs, info = env.reset(seed=1)
        total, n, done = 0.0, 0, False
        while not done:
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            total += r
            n += 1
            done = term or trunc
        print(f"{info['map']}: {n} steps, return {total:+.2f}, rank {info['rank']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
