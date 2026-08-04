"""
rl/line_of_sight.py
===================

Does the straight line from the player to a target pass through a wall?

NOTHING CONSUMES THIS YET, AND THAT IS DELIBERATE. The answer is only half of a
fire decision, because whether a wall matters depends entirely on who you are
playing:

    throwers   (Barley, Dynamike, Tick, Sprout, Willow)   arc OVER walls
    melee      (El Primo, Rosa, Bull, Edgar)              no projectile at all
    bouncers   (Rico)                                     a wall is a BANK SHOT
    shooters   (Colt, Piper, Brock, Bea)                  wall = shot wasted
    supers     (Shelly, El Primo, and others)             BREAK walls, so a wall
                                                          in the line is a reason
                                                          to fire, not to hold

Only the fourth row wants "blocked -> hold fire". So `CombatConfig` carries a
`wall_blocks_shots` flag to be set per run, and this module supplies the
geometry it would need. Wiring the two together is a separate decision.

Meanwhile it is worth having on its own: rendered into a trace it tells you
whether the fused occupancy grid agrees with what you can see in the frame,
which is a direct read on map quality.

READ THIS BEFORE TRUSTING THE RESULT
------------------------------------
The occupancy grid is fused around the PLAYER ANCHOR, and the anchor is the
least reliable part of the perception stack (missing on ~24% of steps in the
45-minute session of 2026-08-01, and occasionally locked onto an enemy). A
line-of-sight test computed on a map built from a bad origin is a confident
wrong answer.

So the failure mode is chosen on purpose: when anything is unknown -- no world,
no geometry, no walls ever observed, an endpoint off the map -- this reports
NOT BLOCKED. A false "clear" wastes a shot. A false "blocked" silences the gun,
and would do it most often exactly when perception is already struggling, which
is when the agent most needs to fight back. `rl/combat.py` makes the same trade
for the ammo reading, for the same reason.
"""

from __future__ import annotations

from math import floor
from typing import Iterator, Optional, Tuple

import numpy as np

Cell = Tuple[int, int]


def line_blocked(world, from_px, to_px, *,
                 treat_unknown_as_blocked: bool = False,
                 ignore_endpoints: bool = True) -> bool:
    """True if a wall lies between `from_px` and `to_px`.

    Both points are FRAME PIXELS (the coordinate system `GameState.player_pos`
    and `enemy_positions` are already in); conversion to map cells is done here.

    Parameters
    ----------
    world : rl.world_map.WorldMap or None
    treat_unknown_as_blocked : bool
        Whether never-observed ground counts as a wall. Default False -- see the
        module docstring on why this direction was chosen.
    ignore_endpoints : bool
        Skip the cells the two points themselves stand in. Default True: a
        brawler hugging a wall often has his own centre land in a cell the grid
        calls solid, and that wall is beside the target rather than between you.

    Returns False whenever the question cannot be answered.
    """
    cells = line_cells(world, from_px, to_px)
    if cells is None:
        return False
    if ignore_endpoints and len(cells) > 2:
        cells = cells[1:-1]
    if not cells:
        return False

    occ = world.occupancy
    gh, gw = occ.shape
    blocked = occ
    if treat_unknown_as_blocked:
        blocked = occ | world.unexplored

    for cx, cy in cells:
        if 0 <= cx < gw and 0 <= cy < gh and blocked[cy, cx]:
            return True
    return False


def line_cells(world, from_px, to_px) -> Optional[list]:
    """The map cells the segment crosses, or None if the map cannot answer.

    Separate from `line_blocked` so a trace overlay can draw the ray it tested
    rather than a straight line it assumes was tested -- when the two disagree
    the bug is here, and that is invisible if the drawing is independent.
    """
    if world is None or not world.ready():
        return None
    # No terrain observation has been fused yet: every cell reads as passable,
    # so a "clear" answer would be vacuous rather than informative.
    has_walls = getattr(world, "has_walls", None)
    if callable(has_walls) and not has_walls():
        return None
    if from_px is None or to_px is None:
        return None
    try:
        ax, ay = world.to_cell(from_px)
        bx, by = world.to_cell(to_px)
    except Exception:
        return None
    if not all(np.isfinite(v) for v in (ax, ay, bx, by)):
        return None
    return list(supercover(ax, ay, bx, by))


def supercover(x0: float, y0: float, x1: float, y1: float) -> Iterator[Cell]:
    """Every cell the segment touches, in order from (x0, y0) to (x1, y1).

    WHY NOT BRESENHAM. Bresenham draws a THIN line: at a diagonal step it moves
    x and y at once and never visits the two cells flanking that corner. Two
    walls meeting at a diagonal seam therefore look like a gap, and the shot is
    reported clear straight through the join -- a rare-looking bug that is
    actually common, because tile-based maps are full of diagonal seams.

    This is the Amanatides-Woo voxel traversal instead: it steps one axis at a
    time and so visits every cell whose area the segment enters. For a fire
    decision that conservatism is the right side to err on; for drawing it would
    be too thick, which is why this is not shared with the path renderer.

    Coordinates are FRACTIONAL CELLS (what `WorldMap.to_cell` returns).
    """
    cx, cy = int(floor(x0)), int(floor(y0))
    end_x, end_y = int(floor(x1)), int(floor(y1))

    dx, dy = x1 - x0, y1 - y0
    step_x = 1 if dx > 0 else -1
    step_y = 1 if dy > 0 else -1

    inf = float("inf")
    # Parametric distance (in units of the whole segment) between successive
    # boundary crossings on each axis, and to the FIRST crossing.
    t_delta_x = abs(1.0 / dx) if dx != 0 else inf
    t_delta_y = abs(1.0 / dy) if dy != 0 else inf
    if dx > 0:
        t_max_x = (cx + 1 - x0) / dx
    elif dx < 0:
        t_max_x = (cx - x0) / dx
    else:
        t_max_x = inf
    if dy > 0:
        t_max_y = (cy + 1 - y0) / dy
    elif dy < 0:
        t_max_y = (cy - y0) / dy
    else:
        t_max_y = inf

    yield (cx, cy)
    # Bounded: the segment cannot cross more cells than its extent in cells,
    # plus one per axis. The cap also stops a pathological float case spinning.
    max_steps = abs(end_x - cx) + abs(end_y - cy) + 2
    for _ in range(max_steps):
        if (cx, cy) == (end_x, end_y):
            break
        if t_max_x < t_max_y:
            if t_max_x > 1.0:
                break
            cx += step_x
            t_max_x += t_delta_x
        else:
            if t_max_y > 1.0:
                break
            cy += step_y
            t_max_y += t_delta_y
        yield (cx, cy)


# --- offline check ----------------------------------------------------------- #
if __name__ == "__main__":
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

    print("supercover, straight and diagonal:")
    for a, b in (((0.5, 0.5), (4.5, 0.5)),
                 ((0.5, 0.5), (0.5, 3.5)),
                 ((0.5, 0.5), (3.5, 3.5))):
        print(f"  {a} -> {b}: {list(supercover(a[0], a[1], b[0], b[1]))}")

    # The case Bresenham gets wrong: a wall seam crossed exactly at a corner.
    print("\ndiagonal corner (a thin line would skip the flanking cells):")
    print(" ", list(supercover(0.5, 0.5, 2.5, 2.5)))
