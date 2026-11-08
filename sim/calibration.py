"""
sim/calibration.py
==================

Calibrate the simulator from real footage (Phase 2 of
docs/PLAN_pro_imitation.md).

    # stats measured from recordings
    python tools/fit_from_footage.py calibrate --cache cache/doin --out doin_calib.json
    # -> simulator settings, plus the sim's elimination curve beside the footage's
    python -m sim.calibration doin_calib.json --out sim_calib.json --matches 40
    # evaluate under that calibration
    python -m sim.evaluate --controllers tactical --matches 320 --calibration sim_calib.json

WHAT IS CALIBRATED, AND HOW FAR TO TRUST IT
-------------------------------------------
* agent_speed: the median world speed of the recorded player while moving
  (tools/label_footage.py). Exact for that player's brawler.
* gas_start_s: the EARLIEST time gas was seen across the recorded matches.
  Gas starts at the arena edge and a player only sees it once it is on
  screen, so this is an upper bound on when gas starts. More matches give a
  tighter bound.
* The elimination curve (players left over time) cannot be set directly --
  it is how hard the opponents fight -- so it is printed next to the sim's.
  If the sim's field thins much faster or slower than real matches, the
  bots' aggression (sim/bots.BotSkill) is the knob to turn.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pathlib
import sys
from typing import Dict, List

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def from_footage(stats: dict) -> dict:
    """fit_from_footage `calibrate` output -> sim.harness.CALIBRATION."""
    calib: Dict = {"arena": {}}
    if stats.get("moving_speed_tps_median"):
        calib["agent_speed"] = round(float(stats["moving_speed_tps_median"]), 2)
    gas = [m["first_gas_s"] for m in stats.get("matches", []) if m.get("first_gas_s") is not None]
    if gas:
        calib["arena"]["gas_start_s"] = round(min(gas), 1)
    return calib


def footage_curve(stats: dict) -> List[tuple]:
    """Median players-left at each 10 s mark across recorded matches."""
    by_t: Dict[int, list] = {}
    for m in stats.get("matches", []):
        for t, n in m.get("players_left", []):
            by_t.setdefault(int(t) // 10 * 10, []).append(n)
    return [(t, sorted(v)[len(v) // 2]) for t, v in sorted(by_t.items())]


def _sim_curve(job):
    seed, calib = job
    from sim import harness
    harness.CALIBRATION.clear()
    harness.CALIBRATION.update(calib)
    marks = {}

    def tick(arena, view, agent, state):
        t = int(arena.t)
        if t % 10 == 0 and t not in marks:
            marks[t] = sum(1 for b in arena.brawlers if b.alive)

    harness.run_match("tactical", "shelly", seed=seed, on_tick=tick, stop_on_death=False)
    return marks


def sim_curve(calib: dict, matches: int = 40, seed: int = 60000, workers: int = 4) -> List[tuple]:
    with mp.get_context("spawn").Pool(workers) as pool:
        res = pool.map(_sim_curve, [(s, calib) for s in range(seed, seed + matches)])
    by_t: Dict[int, list] = {}
    for marks in res:
        for t, n in marks.items():
            by_t.setdefault(t, []).append(n)
    return [(t, sorted(v)[len(v) // 2]) for t, v in sorted(by_t.items())]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("footage_stats", help="JSON from `tools/fit_from_footage.py calibrate`")
    ap.add_argument("--out", default=None, help="write the sim calibration here")
    ap.add_argument("--matches", type=int, default=40, help="sim matches for the curve (0 = skip)")
    a = ap.parse_args(argv)
    stats = json.loads(pathlib.Path(a.footage_stats).read_text())
    calib = from_footage(stats)
    print("sim calibration:", json.dumps(calib))
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(calib, indent=1))
    if a.matches:
        real = dict(footage_curve(stats))
        sim = dict(sim_curve(calib, a.matches))
        print(f"{'t (s)':>6}  {'footage':>8}  {'sim':>5}   players left (median)")
        for t in sorted(set(real) | set(sim)):
            print(f"{t:6d}  {str(real.get(t, '-')):>8}  {str(sim.get(t, '-')):>5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
