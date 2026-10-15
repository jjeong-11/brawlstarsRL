"""
sim/search.py
=============

Random search over TacticsConfig, scored in the simulator, with a held-out
validation stage so the winner is not just the luckiest draw.

    python -m sim.search --candidates 24 --screen 96 --validate 320 --top 4

Screening plays every candidate on the SAME seeds (paired), which removes most
of the map-to-map variance from the comparison. Selecting the best of many
noisy scores still overstates the winner, so the top few are re-played on
seeds none of them were selected on, against the current defaults, and only
those numbers should be believed.

Results stream to sim_search.jsonl (one line per evaluated configuration).
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

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# name: (low, high, kind)
SPACE = {
    "engage_ratio": (1.5, 4.0, float),
    "avoid_utility": (2.0, 3.5, float),
    "avoid_radius_tiles": (8.0, 14.0, float),
    "loot_threat_penalty": (0.3, 1.5, float),
    "camp_utility": (0.5, 1.3, float),
    "box_utility": (0.8, 1.6, float),
    "box_decay": (0.015, 0.05, float),
    "cube_utility": (1.6, 2.4, float),
    "retreat_hp": (0.3, 0.6, float),
    "heal_hp": (0.5, 0.8, float),
    "gas_margin_tiles": (3.5, 6.0, float),
    "threat_radius_tiles": (7.0, 12.0, float),
    "standoff_frac": (0.7, 0.95, float),
}


def sample(rng: random.Random) -> dict:
    cfg = {}
    for k, (lo, hi, kind) in SPACE.items():
        cfg[k] = round(rng.uniform(lo, hi), 3)
    cfg["reengage_hp"] = round(min(0.95, cfg["retreat_hp"] + 0.3), 3)
    return cfg


def _play(job):
    seed, overrides, brawler = job
    import dataclasses
    from rl.tactics import TacticsConfig
    from sim.harness import run_match
    tc = dataclasses.replace(TacticsConfig(seed=seed), **overrides) if overrides else None
    r = run_match("tactical", brawler, seed=seed, tactics_config=tc)
    return r.rank, r.won


def score(pool, overrides, seeds, brawler):
    res = pool.map(_play, [(s, overrides, brawler) for s in seeds], chunksize=2)
    ranks = [r for r, _ in res]
    n = len(ranks)
    mean = sum(ranks) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in ranks) / max(1, n - 1))
    return {"mean_rank": round(mean, 3), "se": round(sd / math.sqrt(n), 3),
            "win": round(sum(w for _, w in res) / n, 3),
            "top3": round(sum(r <= 3 for r in ranks) / n, 3), "n": n}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=int, default=24)
    ap.add_argument("--screen", type=int, default=96)
    ap.add_argument("--validate", type=int, default=320)
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--brawler", default="shelly")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="sim_search.jsonl")
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    screen_seeds = list(range(3000, 3000 + args.screen))
    val_seeds = list(range(7000, 7000 + args.validate))
    out = open(args.out, "a")
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        results = []
        cands = [{}] + [sample(rng) for _ in range(args.candidates)]
        for i, c in enumerate(cands):
            s = score(pool, c, screen_seeds, args.brawler)
            results.append((s["mean_rank"], c, s))
            out.write(json.dumps({"stage": "screen", "overrides": c, **s}) + "\n")
            out.flush()
            print(f"[{time.time() - t0:5.0f}s] screen {i:2d} rank {s['mean_rank']:.2f}"
                  f"±{s['se']:.2f} top3 {s['top3']:.0%}  {c or 'DEFAULTS'}", flush=True)
        results.sort(key=lambda x: x[0])
        finalists = [c for _, c, _ in results[:args.top]]
        if {} not in finalists:
            finalists.append({})
        for c in finalists:
            s = score(pool, c, val_seeds, args.brawler)
            out.write(json.dumps({"stage": "validate", "overrides": c, **s}) + "\n")
            out.flush()
            print(f"[{time.time() - t0:5.0f}s] VALIDATE rank {s['mean_rank']:.2f}"
                  f"±{s['se']:.2f} win {s['win']:.0%} top3 {s['top3']:.0%}  "
                  f"{c or 'DEFAULTS'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
