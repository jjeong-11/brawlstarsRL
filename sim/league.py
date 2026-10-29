"""
sim/league.py
=============

Self-play over the decision layer's settings (Phase 3 of
docs/PLAN_pro_imitation.md, first cut).

A population of TacticsConfig variants plays itself in `sim/multi.py`
matches. Each match seats `agent_seats` agents -- mostly current population
members, sometimes a past champion (OpenAI Five's 80/20 mix of latest and
past opponents) -- and fills the remaining seats with scripted bots. Fitness
is mean placement. Survivors keep their history, so their estimates sharpen
across generations; the worse half is replaced by perturbed copies of the
better half.

The policy stays a set of named settings, so whatever the league finds is
still explainable ("engages at 2.1x, avoids within 9 tiles").

    python -m sim.league --generations 12 --out league.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import pathlib
import random
import sys
import time
from typing import Dict, List

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sim.imitate import SPACE, perturb  # noqa: E402


def _match(job):
    seed, seat_cfgs = job
    from sim.multi import run_multi
    seats = {i: dict(c["cfg"]) for i, c in seat_cfgs.items()}
    res = run_multi(seed, seats)
    return [(seat_cfgs[p["seat"]]["id"], p["rank"]) for p in res if p["agent"]]


def default_cfg() -> dict:
    from rl.tactics import TacticsConfig
    d = TacticsConfig()
    return {k: getattr(d, k) for k in SPACE}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--population", type=int, default=6)
    ap.add_argument("--generations", type=int, default=12)
    ap.add_argument("--matches", type=int, default=16, help="matches per generation")
    ap.add_argument("--agent-seats", type=int, default=6)
    ap.add_argument("--past-frac", type=float, default=0.2)
    ap.add_argument("--sigma", type=float, default=0.12, help="mutation, fraction of range")
    ap.add_argument("--seed", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="league.jsonl")
    a = ap.parse_args(argv)

    rng = random.Random(a.seed)
    next_id = [0]

    def new(cfg, parent=None):
        next_id[0] += 1
        return {"id": next_id[0], "cfg": cfg, "parent": parent, "ranks": []}

    pop: List[dict] = [new(default_cfg())]
    while len(pop) < a.population:
        pop.append(new(perturb(pop[0]["cfg"], rng, a.sigma), parent=1))
    past: List[dict] = []                    # hall of fame (champions' snapshots)
    by_id: Dict[int, dict] = {c["id"]: c for c in pop}
    out = open(a.out, "a")
    t0 = time.time()
    seed = a.seed

    for gen in range(a.generations):
        jobs = []
        for _ in range(a.matches):
            seat_cfgs = {}
            for s in range(a.agent_seats):
                if past and rng.random() < a.past_frac:
                    c = rng.choice(past)
                else:
                    c = pop[(len(jobs) * a.agent_seats + s) % len(pop)]   # balanced
                seat_cfgs[s] = {"id": c["id"], "cfg": c["cfg"]}
            jobs.append((seed, seat_cfgs))
            seed += 1
        with mp.get_context("spawn").Pool(a.workers) as pool:
            results = pool.map(_match, jobs, chunksize=1)
        for res in results:
            for cid, rank in res:
                if cid in by_id:
                    by_id[cid]["ranks"].append(rank)

        def fitness(c):
            r = c["ranks"]
            return sum(r) / len(r) if r else 99.0

        pop.sort(key=fitness)
        champ = pop[0]
        n = len(champ["ranks"])
        sd = (math.sqrt(sum((x - fitness(champ)) ** 2 for x in champ["ranks"]) / max(1, n - 1))
              if n > 1 else 0.0)
        line = {"gen": gen, "t": round(time.time() - t0),
                "pop": [{"id": c["id"], "parent": c["parent"], "mean_rank": round(fitness(c), 3),
                         "n": len(c["ranks"]), "cfg": c["cfg"]} for c in pop]}
        out.write(json.dumps(line) + "\n")
        out.flush()
        print(f"[{line['t']:5d}s] gen {gen:2d}: champion #{champ['id']} mean rank "
              f"{fitness(champ):.2f}±{sd / math.sqrt(max(1, n)):.2f} (n={n})  "
              f"worst {fitness(pop[-1]):.2f}  {champ['cfg']}", flush=True)

        # Hall of fame: snapshot the champion (with its config frozen).
        snap = {"id": -champ["id"] * 1000 - gen, "cfg": dict(champ["cfg"])}
        past.append(snap)
        past = past[-6:]
        # Replace the worse half with mutants of the better half.
        keep = max(1, len(pop) // 2)
        survivors = pop[:keep]
        children = [new(perturb(survivors[i % keep]["cfg"], rng, a.sigma), parent=survivors[i % keep]["id"])
                    for i in range(len(pop) - keep)]
        pop = survivors + children
        by_id = {c["id"]: c for c in pop}

    best = min(pop, key=lambda c: (sum(c["ranks"]) / len(c["ranks"])) if c["ranks"] else 99)
    print("CHAMPION", json.dumps(best["cfg"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
