"""
sim/imitate.py
==============

Can the decision layer be fitted from what FOOTAGE shows? (Phase 1c,
docs/PLAN_pro_imitation.md)

Pro footage yields the player's movement (tools/label_footage.py), not the
reasons behind it. So the fit must work from movement alone:

  * A DEMONSTRATOR (a TacticalBrain with settings unknown to the fit) plays
    simulated matches.
  * The LABEL at each tick is where it actually went: its world displacement
    over the next `horizon_s` seconds, or "stayed" if under `still_tiles`.
  * Each CANDIDATE config is a TacticalBrain run in lockstep on the
    demonstrator's own states (same perception, same world map, same
    tracker) -- teacher forcing, as when scoring a policy's likelihood of
    demonstrations. Its prediction is the direction from the player to the
    destination it chose (or "stay" if it chose to hold).
  * AGREEMENT per tick: both moving -> max(0, cos(angle)); both still -> 1;
    otherwise 0. A candidate's score is the mean over all ticks.

The method is validated when the demonstrator's true settings score best
and agreement falls with parameter distance from them. Then the same scorer
runs on (perception features, labelled movement) from pro footage.

    python -m sim.imitate --matches 12 --candidates 24
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import multiprocessing as mp
import pathlib
import random
import sys
from typing import Dict, List

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# The knobs that most change WHERE the brain goes. (name: (low, high))
SPACE = {
    "engage_ratio": (1.2, 4.0),
    "avoid_utility": (1.5, 3.5),
    "avoid_radius_tiles": (7.0, 14.0),
    "camp_utility": (0.3, 1.4),
    "box_utility": (0.6, 1.8),
    "retreat_hp": (0.25, 0.6),
}
# A demonstrator that plays differently from the defaults: fights earlier,
# avoids less, barely camps, loves boxes.
SECRET = {"engage_ratio": 1.6, "avoid_utility": 2.0, "avoid_radius_tiles": 8.5,
          "camp_utility": 0.45, "box_utility": 1.5, "retreat_hp": 0.32}

HORIZON_S = 1.0
STILL_TILES = 0.5


def sample(rng: random.Random) -> dict:
    return {k: round(rng.uniform(lo, hi), 3) for k, (lo, hi) in SPACE.items()}


def distance(a: dict, b: dict) -> float:
    """Normalised parameter distance (0 = identical, 1 = across the whole box)."""
    return math.sqrt(sum(((a[k] - b[k]) / (hi - lo)) ** 2
                         for k, (lo, hi) in SPACE.items()) / len(SPACE))


def agreement(preds, labels):
    """Per-candidate agreement with labelled movement.

    preds[k][i]: candidate k's direction at tick i (unit vector) or None (hold).
    labels[i]: observed direction (unit vector), None (stayed), or "end"
    (no future to label). Returns (sums, n, weighted_sums, weight_total):
    plain agreement is sums[k] / n, disagreement-weighted is
    weighted_sums[k] / weight_total.
    """
    K = len(preds)
    sums = [0.0] * K
    wsums = [0.0] * K
    n = 0
    wtot = 0.0
    for i, lab in enumerate(labels):
        if lab == "end":
            continue
        n += 1
        # How much do the candidates disagree here? Mean pairwise agreement
        # A from the resultant of the movers (sum of cosines over pairs is
        # (|S|^2 - m) / 2) plus stay-stay pairs; weight = 1 - A. Ticks where
        # every setting would do the same thing carry no information about
        # the settings, and they are most ticks.
        mv = [preds[k][i] for k in range(K) if preds[k][i] is not None]
        stays = K - len(mv)
        sx = sum(v[0] for v in mv)
        sy = sum(v[1] for v in mv)
        pair = stays * (stays - 1) + max(0.0, sx * sx + sy * sy - len(mv))
        w = 1.0 - pair / max(1, K * (K - 1))
        wtot += w
        for k in range(K):
            p = preds[k][i]
            a = 0.0
            if lab is None and p is None:
                a = 1.0
            elif lab is not None and p is not None:
                a = max(0.0, lab[0] * p[0] + lab[1] * p[1])
            sums[k] += a
            wsums[k] += w * a
    return sums, n, wsums, wtot


def _play(job):
    """One demonstration match; returns per-candidate agreement sums."""
    seed, demo, cands = job
    from rl.tactics import TacticalBrain, TacticsConfig
    from sim.harness import run_match

    demo_cfg = dataclasses.replace(TacticsConfig(seed=seed), **demo)
    brains: List[TacticalBrain] = []
    preds: List[list] = [[] for _ in cands]     # per tick: (dx, dy) unit or None
    track: list = []                             # true player world positions

    def on_tick(arena, view, agent, state):
        me = arena.brawlers[0]
        if not me.alive or state.player_pos is None:
            return
        if not brains:
            for c in cands:
                brains.append(TacticalBrain(dataclasses.replace(TacticsConfig(seed=seed), **c),
                                            ability=agent.brain.ability,
                                            combat=agent.brain.combat))
        px, py = state.player_pos
        track.append((arena.t, me.x, me.y))
        for k, b in enumerate(brains):
            g = b.decide(state, agent.planner, agent.tracker)
            if g.point is None:
                preds[k].append(None)
            else:
                dx, dy = g.point[0] - px, g.point[1] - py
                n = math.hypot(dx, dy)
                preds[k].append(None if n < 1e-6 else (dx / n, dy / n))

    r = run_match("tactical", "shelly", seed=seed, tactics_config=demo_cfg, on_tick=on_tick)

    # Labels: displacement over the next HORIZON_S (world tiles; screen y is
    # world y, so directions compare directly).
    labels = []
    j = 0
    for i, (t, x, y) in enumerate(track):
        while j < len(track) and track[j][0] < t + HORIZON_S:
            j += 1
        if j >= len(track):
            labels.append("end")
            continue
        dx, dy = track[j][1] - x, track[j][2] - y
        n = math.hypot(dx, dy)
        labels.append(None if n < STILL_TILES else (dx / n, dy / n))

    sums, n, wsums, wtot = agreement(preds, labels)
    return sums, n, r.rank, wsums, wtot


def score(cands, seeds, workers, demo=SECRET, weighted=False):
    """Agreement of every candidate with the demonstrator over `seeds`.

    weighted=True scores only where the candidates disagree (see _play); the
    numbers are then relative to this candidate set."""
    jobs = [(s, demo, cands) for s in seeds]
    with mp.get_context("spawn").Pool(workers) as pool:
        res = pool.map(_play, jobs, chunksize=1)
    if weighted:
        wt = sum(r[4] for r in res)
        return [sum(r[3][k] for r in res) / max(1e-9, wt) for k in range(len(cands))]
    n = sum(r[1] for r in res)
    return [sum(r[0][k] for r in res) / max(1, n) for k in range(len(cands))]


def perturb(base: dict, rng: random.Random, frac: float) -> dict:
    out = {}
    for k, (lo, hi) in SPACE.items():
        v = base[k] + rng.gauss(0.0, frac * (hi - lo))
        out[k] = round(min(hi, max(lo, v)), 3)
    return out


def fit(seeds, workers, rounds=5, per_round=24, keep=4, rng_seed=11, weighted=True):
    """Random search, then shrinking local search around the leaders. The
    demonstrator's settings are never among the candidates. Leaders are
    re-scored with each new round, so weighted scores stay comparable."""
    rng = random.Random(rng_seed)
    pool_ = [sample(rng) for _ in range(per_round)]
    scored = []
    for r in range(rounds):
        sc = score(pool_, seeds, workers, weighted=weighted)
        scored = sorted(zip(sc, pool_), key=lambda t: -t[0])
        best = scored[0]
        print(f"  round {r}: best agreement {best[0]:.3f}  dist {distance(best[1], SECRET):.2f}  {best[1]}",
              flush=True)
        leaders = [c for _, c in scored[:keep]]
        frac = 0.2 / (r + 1)
        pool_ = leaders + [perturb(leaders[i % keep], rng, frac)
                           for i in range(per_round - keep)]
    return scored[0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=12)
    ap.add_argument("--candidates", type=int, default=24)
    ap.add_argument("--seed", type=int, default=9000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--unweighted", action="store_true",
                    help="score every tick equally (the first version)")
    ap.add_argument("--fit", action="store_true",
                    help="search for the settings (without the true ones) and report recovery")
    a = ap.parse_args(argv)

    if a.fit:
        seeds = list(range(a.seed, a.seed + a.matches))
        sc, best = fit(seeds, a.workers, weighted=not a.unweighted)
        print(f"fitted: agreement {sc:.3f}, distance to the demonstrator {distance(best, SECRET):.2f}")
        for k, (lo, hi) in SPACE.items():
            print(f"  {k:20s} fitted {best[k]:6.2f}   true {SECRET[k]:6.2f}   "
                  f"error {abs(best[k] - SECRET[k]) / (hi - lo):4.0%} of range")
        held = list(range(a.seed + 1000, a.seed + 1000 + a.matches))
        from rl.tactics import TacticsConfig
        d = TacticsConfig()
        default = {k: getattr(d, k) for k in SPACE}
        h = score([dict(SECRET), best, default], held, a.workers)
        print(f"held-out agreement: demonstrator itself {h[0]:.3f}, fitted {h[1]:.3f}, defaults {h[2]:.3f}")
        print("FIT", {"best": best})
        return 0

    from rl.tactics import TacticsConfig
    d = TacticsConfig()
    default = {k: getattr(d, k) for k in SPACE}
    rng = random.Random(5)
    cands: List[Dict] = [dict(SECRET), default] + [sample(rng) for _ in range(a.candidates)]
    names = ["SECRET (demonstrator)", "defaults"] + [f"random {i}" for i in range(a.candidates)]
    jobs = [(s, SECRET, cands) for s in range(a.seed, a.seed + a.matches)]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        res = pool.map(_play, jobs, chunksize=1)
    tot = [sum(r[0][k] for r in res) for k in range(len(cands))]
    n = sum(r[1] for r in res)
    scores = [t / max(1, n) for t in tot]
    order = sorted(range(len(cands)), key=lambda k: -scores[k])
    print(f"{a.matches} demonstration matches, {n} labelled ticks; "
          f"demonstrator mean rank {sum(r[2] for r in res) / len(res):.2f}")
    print(f"{'rank':>4}  {'agreement':>9}  {'dist':>5}  candidate")
    for pos, k in enumerate(order):
        print(f"{pos + 1:4d}  {scores[k]:9.3f}  {distance(cands[k], SECRET):5.2f}  {names[k]}")
    # Does agreement fall with distance from the truth? (Spearman over randoms)
    ks = list(range(2, len(cands)))
    rs = sorted(ks, key=lambda k: scores[k])
    rd = sorted(ks, key=lambda k: distance(cands[k], SECRET))
    rank_s = {k: i for i, k in enumerate(rs)}
    rank_d = {k: i for i, k in enumerate(rd)}
    m = len(ks)
    rho = 1 - 6 * sum((rank_s[k] - rank_d[k]) ** 2 for k in ks) / (m * (m * m - 1))
    print(f"Spearman(agreement, distance) over random candidates: {rho:+.2f} "
          f"(negative = agreement falls as settings move away from the truth)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
