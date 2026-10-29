"""
sim/multi.py
============

Several seats played by the bot's full stack -- perception model, tracker,
planner, tactical brain, combat -- in one match. The infrastructure Phase 3
of docs/PLAN_pro_imitation.md needs: self-play means the opponents are
copies (or past versions) of the agent, not the scripted `sim/bots.py`.

    python -m sim.multi --seats 10 --seed 1          # all ten seats, one match
    python -m sim.multi --seats 3 --matches 8        # three agents + 7 bots

Each agent seat sees the arena through its OWN `SimView` (same noise model
as `sim/evaluate.py`), so agents are as blind to each other as the bot is to
real players. Seats not given a config are scripted bots.
"""

from __future__ import annotations

import argparse
import dataclasses
import multiprocessing as mp
import pathlib
import random
import sys
import time
from typing import Dict, List, Optional

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def build_multi(seed: int, map_name: Optional[str] = None, brawlers: Optional[List[str]] = None,
                bot_skill=(0.25, 1.0)):
    """(arena, abilities per seat, game_map, rng): like harness.build_match,
    but every seat's ability is kept so any seat can be an agent."""
    from brawlers.abilities import default_abilities
    from perception.mapdb import database

    from .arena import Arena, bush_mask, stats_for
    from .harness import _BUSH, playable_maps

    rng = random.Random(seed)
    maps = playable_maps()
    gm = database().get(map_name) if map_name else maps[rng.randrange(len(maps))]
    book = default_abilities()
    pool = [a for a in book.all() if a.range_tiles]
    abilities = []
    for i in range(10):
        name = brawlers[i] if brawlers and i < len(brawlers) and brawlers[i] else None
        abilities.append(book.get(name) if name else rng.choice(pool))
    if gm.name not in _BUSH:
        _BUSH[gm.name] = bush_mask(gm)
    arena = Arena(gm, _BUSH[gm.name], [stats_for(a) for a in abilities], seed=seed)
    return arena, abilities, gm, rng


def run_multi(seed: int, seats: Dict[int, dict], map_name: Optional[str] = None,
              brawlers: Optional[List[str]] = None, dt: float = 0.1,
              action_delay: int = 1, localize: bool = False,
              bot_skill=(0.25, 1.0)) -> List[dict]:
    """Play one match. `seats` maps seat index -> TacticsConfig overrides for
    the seats the agent stack plays; every other seat is a scripted bot.

    Returns one dict per seat: seat, agent (bool), rank, kills, cubes,
    survival_s, death_cause, config (overrides or None).

    `localize` is off by default: the map localiser is the most expensive
    stage and identical for every agent, and self-play wants many matches.
    """
    from rl.tactics import TacticsConfig

    from .bots import Bot, BotSkill
    from .harness import AgentStack, PerceptionModel, SimView

    arena, abilities, gm, rng = build_multi(seed, map_name, brawlers, bot_skill)
    agents, views, queues = {}, {}, {}
    for i, ov in seats.items():
        cfg = dataclasses.replace(TacticsConfig(seed=seed * 17 + i, tick_seconds=dt), **(ov or {}))
        agents[i] = AgentStack("tactical", ability=abilities[i], seed=seed * 17 + i,
                               tactics_config=cfg, localize=localize, tick_seconds=dt)
        views[i] = SimView(arena, i, PerceptionModel(), seed=seed * 13 + i)
        queues[i] = []
    bots = [Bot(arena, i, BotSkill.sample(rng, *bot_skill), random.Random(seed * 101 + i))
            for i in range(10) if i not in seats]

    while not arena.over():
        for i, agent in agents.items():
            b = arena.brawlers[i]
            if not b.alive:
                continue
            state, terrain, gas, cam = views[i].observe()
            move, dec = agent.act(state, terrain, gas, cam)
            q = queues[i]
            q.append((move, dec))
            if len(q) > action_delay:
                mv, d = q.pop(0)
                b.move = (float(mv[0]), float(mv[1]))
                if d.attack:
                    arena.fire(b, d.aim, d.aim_frac)
                if d.super:
                    arena.fire(b, d.aim, d.aim_frac, is_super=True)
        for bot in bots:
            bot.act(dt)
        arena.step(dt)

    out = []
    alive = [b for b in arena.brawlers if b.alive]
    for i, b in enumerate(arena.brawlers):
        rank = b.rank if b.rank is not None else (1 if b.alive and len(alive) == 1 else len(alive))
        out.append({"seat": i, "agent": i in seats, "rank": int(rank), "kills": b.kills,
                    "cubes": b.cubes_collected, "survival_s": round(b.death_t or arena.t, 1),
                    "death_cause": b.death_cause, "brawler": abilities[i].id if abilities[i] else None,
                    "config": seats.get(i) if i in seats else None, "map": gm.name})
    return out


def _job(args):
    seed, seats = args
    t0 = time.time()
    res = run_multi(seed, seats)
    return res, time.time() - t0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seats", type=int, default=10, help="how many seats the agent plays")
    ap.add_argument("--matches", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    seats = {i: {} for i in range(a.seats)}
    jobs = [(s, seats) for s in range(a.seed, a.seed + a.matches)]
    with mp.get_context("spawn").Pool(min(a.workers, len(jobs))) as pool:
        res = pool.map(_job, jobs)
    secs = [r[1] for r in res]
    agent_ranks = [p["rank"] for r, _ in res for p in r if p["agent"]]
    bot_ranks = [p["rank"] for r, _ in res for p in r if not p["agent"]]
    print(f"{len(res)} matches, {a.seats} agent seats: {sum(secs) / len(secs):.1f} s per match")
    if agent_ranks:
        print(f"  agent mean rank {sum(agent_ranks) / len(agent_ranks):.2f}")
    if bot_ranks:
        print(f"  bot mean rank   {sum(bot_ranks) / len(bot_ranks):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
