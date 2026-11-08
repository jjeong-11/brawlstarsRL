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


def _paired_job(job):
    """One candidate on one seed: a self-play match (candidate in seat 0,
    fixed rivals, bots elsewhere) AND a single-agent match against bots on
    the same seed. Every candidate in a generation gets the same jobs, so
    differences between candidates are paired."""
    seed, cand, rivals = job
    from sim.harness import run_match
    from sim.multi import run_multi
    import dataclasses
    from rl.tactics import TacticsConfig
    seats = {0: dict(cand)}
    for i, r in enumerate(rivals, start=1):
        seats[i] = dict(r)
    res = run_multi(seed, seats)
    sp = next(p["rank"] for p in res if p["seat"] == 0)
    tc = dataclasses.replace(TacticsConfig(seed=seed), **cand)
    vb = run_match("tactical", "shelly", seed=seed, tactics_config=tc).rank
    return sp, vb


def paired(a) -> int:
    """League v2: paired seeds and rivals, mixed fitness, survivors re-scored."""
    rng = random.Random(a.seed)
    pop = [default_cfg()] + [perturb(default_cfg(), rng, a.sigma) for _ in range(a.population - 1)]
    hist = {i: [] for i in range(len(pop))}       # per-candidate list of per-gen fitness
    ids = list(range(len(pop)))
    next_id = len(pop)
    cfg_of = dict(zip(ids, pop))
    fame: List[dict] = []
    out = open(a.out, "a")
    t0 = time.time()
    seed = a.seed
    for gen in range(a.generations):
        seeds = list(range(seed, seed + a.matches))
        seed += a.matches
        # Same rivals for every candidate on a given seed: current members and
        # past champions, drawn once per seed.
        rivals = {}
        for s in seeds:
            pool_ = [cfg_of[i] for i in ids] + fame
            rivals[s] = [rng.choice(pool_) for _ in range(a.agent_seats - 1)]
        jobs = [(s, cfg_of[i], rivals[s]) for i in ids for s in seeds]
        with mp.get_context("spawn").Pool(a.workers) as pool:
            res = pool.map(_paired_job, jobs, chunksize=1)
        k = 0
        scores = {}
        for i in ids:
            sp = vb = 0.0
            for _ in seeds:
                r_sp, r_vb = res[k]
                k += 1
                sp += r_sp
                vb += r_vb
            sp /= len(seeds)
            vb /= len(seeds)
            scores[i] = (a.selfplay_weight * sp + (1 - a.selfplay_weight) * vb, sp, vb)
            hist[i].append(scores[i][0])

        def fit(i):
            return sum(hist[i]) / len(hist[i])

        ranked = sorted(ids, key=fit)
        line = {"gen": gen, "t": round(time.time() - t0),
                "pop": [{"id": i, "fitness": round(fit(i), 3), "gens": len(hist[i]),
                         "this_gen": [round(v, 3) for v in scores[i]], "cfg": cfg_of[i]}
                        for i in ranked]}
        out.write(json.dumps(line) + "\n")
        out.flush()
        best = ranked[0]
        d = next((i for i in ids if cfg_of[i] == default_cfg()), None)   # defaults, while alive
        print(f"[{line['t']:5d}s] gen {gen:2d}: best #{best} fitness {fit(best):.2f} over {len(hist[best])} gen(s) "
              f"(self-play {scores[best][1]:.2f}, vs bots {scores[best][2]:.2f})"
              + (f"; defaults {fit(d):.2f}" if d is not None else "") + f"  {cfg_of[best]}", flush=True)
        # Hall of fame gets champions with at least two generations of history.
        if len(hist[best]) >= 2:
            fame.append(dict(cfg_of[best]))
            fame[:] = fame[-6:]
        keep = ranked[:a.keep]
        children = []
        for j in range(len(ids) - a.keep):
            parent = cfg_of[keep[j % len(keep)]]
            children.append(next_id)
            cfg_of[next_id] = perturb(parent, rng, a.sigma)
            hist[next_id] = []
            next_id += 1
        ids = keep + children
    scored = [i for i in hist if hist[i]]
    final = sorted([i for i in scored if len(hist[i]) >= 3] or scored,
                   key=lambda i: sum(hist[i]) / len(hist[i]))
    print("CHAMPION", json.dumps(cfg_of[final[0]]))
    return 0


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
    ap.add_argument("--paired", action="store_true",
                    help="v2: paired seeds and rivals, mixed fitness (see paired())")
    ap.add_argument("--keep", type=int, default=4)
    ap.add_argument("--selfplay-weight", type=float, default=0.5)
    a = ap.parse_args(argv)
    if a.paired:
        return paired(a)

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
