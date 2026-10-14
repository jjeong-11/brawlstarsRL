"""
sim/evaluate.py
===============

Compare controllers over many seeded matches.

    python -m sim.evaluate                              # tactical vs random vs legacy
    python -m sim.evaluate --controllers tactical --matches 100 --brawler piper
    python -m sim.evaluate --brawlers shelly,piper,bull,barley --matches 30
    python -m sim.evaluate --perfect                     # no perception noise
    python -m sim.evaluate --json results.json

Every controller plays the SAME seeds -- same map, spawns, opponents, boxes --
so differences are paired rather than drowned in map-to-map variance. Read the
DIFFERENCES; the absolute numbers belong to the simulator, not to the game.

    legacy    the bot as it was: random headings through the planner (an
              untrained policy is close to uniform) + the old combat rule
              (aspect-ratio range bug, 0.42 ceiling, never shoots boxes)
    random    random headings + the upgraded combat script
    tactical  rl/tactics.py + upgraded combat + rl/enemy_tracker.py
    idle      stands still and shoots (a floor: what does ONLY combat buy?)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time
from typing import Dict, List

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _parse_overrides(text: str) -> dict:
    """"a=1,b=true,c=0.5" -> {"a": 1, "b": True, "c": 0.5}."""
    out = {}
    for part in (text or "").split(","):
        if not part.strip():
            continue
        k, v = part.split("=", 1)
        v = v.strip()
        if v.lower() in ("true", "false"):
            val = v.lower() == "true"
        elif v.lower() == "none":
            val = None
        else:
            try:
                val = int(v)
            except ValueError:
                try:
                    val = float(v)
                except ValueError:
                    val = v
        out[k.strip()] = val
    return out


def _configs(controller, combat_over, tactics_over, seed):
    import dataclasses
    from rl.combat import CombatConfig
    from rl.tactics import TacticsConfig
    from sim.harness import legacy_combat_config
    cc = tc = None
    if combat_over:
        base = legacy_combat_config() if controller == "legacy" else CombatConfig()
        cc = dataclasses.replace(base, **combat_over)
    if tactics_over:
        tc = dataclasses.replace(TacticsConfig(seed=seed), **tactics_over)
    return cc, tc


def _play(args):
    controller, brawler, seed, perfect, localize, combat_over, tactics_over, label = args
    import dataclasses
    from sim.harness import PerceptionModel, run_match
    pm = PerceptionModel.perfect() if perfect else PerceptionModel()
    if _PERCEPTION:
        pm = dataclasses.replace(pm, **_PERCEPTION)
    cc, tc = _configs(controller, combat_over, tactics_over, seed)
    t0 = time.time()
    r = run_match(controller, brawler, seed=seed, perception=pm, localize=localize,
                  combat_config=cc, tactics_config=tc)
    d = r.as_dict()
    d["wall_s"] = time.time() - t0
    d["label"] = label
    return d


def summarise(rows: List[dict]) -> Dict[str, float]:
    n = len(rows)
    if not n:
        return {}
    ranks = [r["rank"] for r in rows]
    mean_rank = sum(ranks) / n
    sd = math.sqrt(sum((x - mean_rank) ** 2 for x in ranks) / max(1, n - 1))
    deaths = [r for r in rows if not r["won"]]
    return {
        "n": n,
        "win_rate": sum(r["won"] for r in rows) / n,
        "top3": sum(r["rank"] <= 3 for r in rows) / n,
        "mean_rank": mean_rank,
        "rank_se": sd / math.sqrt(n),
        "kills": sum(r["kills"] for r in rows) / n,
        "cubes": sum(r["cubes"] for r in rows) / n,
        "boxes": sum(r["boxes"] for r in rows) / n,
        "damage": sum(r["damage"] for r in rows) / n,
        "survival_s": sum(r["survival_s"] for r in rows) / n,
        "gas_deaths": sum(r["death_cause"] == "gas" for r in deaths) / n,
        "enemy_deaths": sum(r["death_cause"] == "enemy" for r in deaths) / n,
        "hit_rate": (sum(r["hits"] + r.get("box_hits", 0) for r in rows)
                     / max(1, sum(r["shots"] for r in rows))),
        "enemy_acc": (sum(r["hits"] for r in rows)
                      / max(1, sum(r["shots"] - r.get("box_hits", 0) for r in rows))),
    }


_PERCEPTION: dict = {}


def _init_worker(perception):
    global _PERCEPTION
    _PERCEPTION = perception


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--controllers", default="tactical,random,legacy")
    ap.add_argument("--brawlers", "--brawler", default="shelly")
    ap.add_argument("--matches", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1000, help="first seed")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--perfect", action="store_true", help="no perception noise")
    ap.add_argument("--perception", default="", metavar="K=V,...",
                    help="PerceptionModel overrides, e.g. entity_interval_s=0.2")
    ap.add_argument("--no-localize", action="store_true",
                    help="disable arena identification (faster, less realistic)")
    ap.add_argument("--json", default=None, help="write every match result here")
    ap.add_argument("--combat", action="append", default=[], metavar="K=V,...",
                    help="CombatConfig overrides for an A/B arm. Repeat the flag "
                         "for several arms; each arm plays every controller on "
                         "the same seeds. e.g. --combat lead_auto_aimers=false")
    ap.add_argument("--tactics", action="append", default=[], metavar="K=V,...",
                    help="TacticsConfig overrides, paired with --combat by "
                         "position (missing = defaults)")
    args = ap.parse_args(argv)

    controllers = [c.strip() for c in args.controllers.split(",") if c.strip()]
    brawlers = [b.strip() for b in args.brawlers.split(",") if b.strip()]
    n_arms = max(1, len(args.combat), len(args.tactics))
    arms = []
    for k in range(n_arms):
        co = _parse_overrides(args.combat[k]) if k < len(args.combat) else {}
        to = _parse_overrides(args.tactics[k]) if k < len(args.tactics) else {}
        label = ";".join(x for x in (args.combat[k] if k < len(args.combat) else "",
                                     args.tactics[k] if k < len(args.tactics) else "")
                         if x) or "default"
        arms.append((co, to, label))
    jobs = [(c, b, args.seed + i, args.perfect, not args.no_localize, co, to, label)
            for (co, to, label) in arms
            for b in brawlers for i in range(args.matches) for c in controllers]

    t0 = time.time()
    rows: List[dict] = []
    if args.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(
                args.workers, initializer=_init_worker,
                initargs=(_parse_overrides(args.perception),)) as pool:
            for k, d in enumerate(pool.imap_unordered(_play, jobs, chunksize=1)):
                rows.append(d)
                if (k + 1) % max(1, len(jobs) // 10) == 0:
                    print(f"  {k + 1}/{len(jobs)} matches ({time.time() - t0:.0f}s)",
                          flush=True)
    else:
        _init_worker(_parse_overrides(args.perception))
        for job in jobs:
            rows.append(_play(job))

    print(f"\n{len(rows)} matches in {time.time() - t0:.0f}s "
          f"({'perfect' if args.perfect else 'noisy'} perception)\n")
    cols = ("win_rate", "top3", "mean_rank", "kills", "cubes", "boxes",
            "survival_s", "gas_deaths", "enemy_deaths", "enemy_acc")
    for b in brawlers:
        print(f"== {b} ==")
        print(f"  {'controller':10s} " + " ".join(f"{c:>12s}" for c in cols))
        for (co, to, label), c in [(arm, c) for arm in arms for c in controllers]:
            sub = [r for r in rows if r["controller"] == c and r["brawler"] == b
                   and r["label"] == label]
            if n_arms > 1:
                print(f"  [{label}]")
            s = summarise(sub)
            vals = []
            for col in cols:
                v = s.get(col, 0.0)
                if col == "mean_rank":
                    vals.append(f"{v:6.2f}±{s.get('rank_se', 0):.2f}")
                elif col in ("win_rate", "top3", "gas_deaths", "enemy_deaths",
                             "hit_rate", "enemy_acc"):
                    vals.append(f"{100 * v:11.0f}%")
                else:
                    vals.append(f"{v:12.1f}")
            print(f"  {c:10s} " + " ".join(f"{v:>12s}" for v in vals))
        print()
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(rows, indent=1))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
