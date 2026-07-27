"""
rl/path_planner.py
==================

Turns a polar waypoint choice into a joystick vector, via a latched destination
and A* over the terrain grid.

THE THREE-LAYER SPLIT
---------------------
    policy   picks WHERE to go, occasionally      (heading, distance)
    planner  keeps that destination fixed in the world and routes to it
    executor drives the joystick                  (dx, dy)

The middle layer is the whole point. In the previous design the policy chose a
cell in a player-centred grid and the target was recomputed as
`player_pos + offset` every single tick -- so the destination moved with the
player and was never reached. That is a compass with extra steps, and it made
exploration *harder* (225 actions instead of 9) while delivering none of the
commitment that was supposed to fix the jittery movement.

Here the waypoint is LATCHED. Once chosen it is held in world space (advanced
each tick by the camera delta from `rl/camera_tracker.py`) and the policy's
movement heads are IGNORED until the commitment ends. It ends when:

    * the player arrives (within `arrive_frac` of the destination), or
    * the commitment times out (`commit_ticks` for the chosen distance tier), or
    * the route is blocked and no progress is possible, or
    * something urgent happens -- gas, or an enemy first coming into view.

That makes each decision a small option/macro-action rather than a single
twitch, which is what produces smooth committed movement instead of jitter.

BECAUSE THE POLICY IS SOMETIMES IGNORED, IT MUST BE ABLE TO SEE THAT.
`status()` exposes whether a commitment is active and how much of it is left,
and `env.encode_observation` feeds that to the network. Without it the agent
would be learning from steps where its movement action had no effect and no way
to tell which ones those were.

A* runs over the occupancy grid from `perception/getTerrain.py`, with soft
costs that push routes away from enemies and mildly prefer bushes (cover). When
no terrain is available (offline smoke tests, a frame where segmentation
failed) it degrades to direct steering with local obstacle avoidance.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from math import atan2, cos, hypot, pi, sin
from typing import Optional, Sequence, Tuple

import numpy as np

# 8-connected moves: (dx, dy, base cost). Diagonals cost sqrt(2).
_SQRT2 = 1.4142135623730951
_NEIGHBOURS = (
    (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
    (1, 1, _SQRT2), (1, -1, _SQRT2), (-1, 1, _SQRT2), (-1, -1, _SQRT2),
)


@dataclass(frozen=True)
class PathPlannerConfig:
    # --- action space shape (mirrored by rl/actions.make_action_space) ---
    n_headings: int = 16
    # Distance tiers as a fraction of the play rect's SHORTER side.
    distances: Tuple[float, ...] = (0.18, 0.34, 0.55)
    # How many DECISION STEPS a commitment may last, per distance tier. Sized
    # from a brawler covering roughly a quarter of the short side per second at
    # this zoom: tier 0 is ~0.8s of travel, tier 2 ~2.4s. These are timeouts,
    # not targets -- arrival normally ends the commitment sooner. If you raise
    # --action-repeat, one decision step is several ticks, so scale these down.
    commit_ticks: Tuple[int, ...] = (8, 16, 26)

    # --- arrival / steering ---
    arrive_frac: float = 0.09      # "close enough", fraction of the short side
    lookahead: float = 0.14        # how far along the path to aim, same units
    edge_margin: float = 0.06      # keep waypoints this far inside the play rect

    # --- A* soft costs ---
    grid_size: Tuple[int, int] = (48, 27)
    enemy_radius: float = 0.20     # enemy influence radius, fraction of short side
    enemy_cost: float = 3.0        # extra cost at an enemy's exact position
    bush_discount: float = 0.15    # <1 multiplier for routing through cover
    unknown_cost: float = 0.4      # mild penalty for cells hidden behind HUD
    # Gas is a COST, not an obstacle, and that distinction is deliberate. Making
    # it impassable would strand the agent when the safe zone is only reachable
    # through the cloud — which is exactly the endgame situation where being
    # wrong is fatal. As a cost, A* takes the long way round whenever one
    # exists, and cuts through only when the detour is genuinely worse. 8.0
    # makes a gas cell about as expensive as eight clear ones, so it will
    # happily walk ~8 cells out of its way per gas cell avoided.
    gas_cost: float = 8.0
    gas_avoid_threshold: float = 0.25   # cell coverage above which to relocate a
                                        # destination that landed in the cloud

    # --- interrupts ---
    interrupt_on_enemy: bool = True   # drop the latch when enemies first appear
    interrupt_on_gas: bool = True     # gas always overrides a walking plan
    # Drop the latch when power-cube boxes first come into view.
    #
    # Committing to a destination is the whole point of this planner, but it
    # has a cost the 8-direction policy did not pay: the agent cannot react to
    # something it only notices mid-commitment. Boxes are exactly that case --
    # they appear as the camera scrolls, and under a far-tier commitment the
    # agent would walk straight past one for up to 26 decisions (~2.5s).
    # Enemies and gas already interrupt for the same reason; leaving boxes out
    # was an oversight, and it visibly suppressed cube collecting.
    #
    # This fires only on the none-visible -> visible TRANSITION, so it cannot
    # shred commitments while boxes stay on screen.
    interrupt_on_box: bool = True
    # Stuck detection. Without this, an agent wedged against a wall keeps
    # pushing for the WHOLE commitment (up to 26 decisions) because A* still
    # reports a valid route and nothing checks whether the route is actually
    # being followed. That is what grinding into a map border looks like, and
    # it survives even a perfect terrain grid: the grid is coarse, so a cell can
    # be walkable while the specific pixel the agent is pressed against is not.
    stuck_ticks: int = 5              # consecutive barely-moving decisions
    stuck_speed: float = 0.015        # movement below this fraction of the
                                      # short side per tick counts as "stuck"

    # --- direct-steering fallback (no terrain grid) ---
    candidate_headings: int = 24
    entity_clearance: float = 0.10


@dataclass
class PlannerStatus:
    """What the planner is currently doing — surfaced to the observation."""
    active: bool = False
    progress: float = 0.0        # 0 = just latched, 1 = commitment exhausted
    waypoint_dx: float = 0.0     # to the destination, fraction of frame width
    waypoint_dy: float = 0.0     # fraction of frame height
    distance: float = 0.0        # 0..1, normalised by the short side
    blocked: bool = False
    replanned: bool = False      # a new waypoint was latched this step


class WaypointPlanner:
    """Latches a destination, routes to it with A*, emits a joystick vector."""

    def __init__(self, config: Optional[PathPlannerConfig] = None):
        self.config = config or PathPlannerConfig()
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self.waypoint: Optional[Tuple[float, float]] = None
        self.ticks_left: int = 0
        self.tier: int = 0
        self.path: list = []
        self.blocked: bool = False
        self._had_enemies: bool = False
        self._had_boxes: bool = False
        self._stuck_for: int = 0
        self._last_screen_pos: Optional[Tuple[float, float]] = None
        self._last_status = PlannerStatus()

    def status(self) -> PlannerStatus:
        return self._last_status

    # ------------------------------------------------------------------ #
    def target_for(self, heading_idx: int, dist_idx: int, player, rect) -> Tuple[float, float]:
        """(heading, distance tier) -> an absolute destination in frame pixels.

        Heading 0 points RIGHT and indices advance clockwise on screen (y grows
        downward, matching image and touch coordinates).
        """
        cfg = self.config
        n = cfg.n_headings
        heading_idx = int(heading_idx) % n
        dist_idx = max(0, min(len(cfg.distances) - 1, int(dist_idx)))

        angle = 2.0 * pi * heading_idx / n
        short = min(rect[2], rect[3])
        reach = cfg.distances[dist_idx] * short
        tx = player[0] + cos(angle) * reach
        ty = player[1] + sin(angle) * reach

        mx, my = cfg.edge_margin * rect[2], cfg.edge_margin * rect[3]
        tx = min(max(tx, rect[0] + mx), rect[0] + rect[2] - mx)
        ty = min(max(ty, rect[1] + my), rect[1] + rect[3] - my)
        return (tx, ty)

    # ------------------------------------------------------------------ #
    def plan(self, heading_idx: int, dist_idx: int, state, frame_size,
             terrain=None, camera_delta=(0.0, 0.0)) -> Tuple[float, float]:
        """Advance one decision step and return a unit joystick vector.

        Parameters
        ----------
        heading_idx, dist_idx : the policy's movement action. CONSULTED ONLY
            when no commitment is active -- see the module docstring.
        state : rl.state.GameState (needs player_pos, enemy_positions, in_gas)
        terrain : dict from perception.getTerrain.find_terrain, or None
        camera_delta : (dx, dy) world scroll since the last step, from
            rl.camera_tracker.CameraTracker.update()
        """
        cfg = self.config
        w, h = frame_size
        rect = terrain["rect"] if terrain else (0, 0, w, h)
        short = min(rect[2], rect[3])

        player = getattr(state, "player_pos", None) if state is not None else None
        if player is None:
            # The anchor detector lost the player: we have no frame of reference
            # for a destination, so drop the latch and steer by raw heading.
            self.waypoint = None
            self.ticks_left = 0
            self._last_status = PlannerStatus(active=False)
            angle = 2.0 * pi * (int(heading_idx) % cfg.n_headings) / cfg.n_headings
            return (cos(angle), sin(angle))

        px, py = float(player[0]), float(player[1])

        # 1) Carry the latched waypoint with the world as the camera scrolls.
        if self.waypoint is not None:
            self.waypoint = (self.waypoint[0] + camera_delta[0],
                             self.waypoint[1] + camera_delta[1])
            self.ticks_left -= 1

        # 2) Am I actually going anywhere? Movement is measured in WORLD terms
        # (screen delta minus camera scroll) because the camera chases the
        # player: while walking normally his screen position barely changes, so
        # screen movement alone would read as "stuck" the entire time.
        if self._last_screen_pos is not None:
            # World displacement = how far the player moved ON SCREEN, minus how
            # far the world itself scrolled underneath him. Walking normally,
            # the first term is ~0 and the second is large; wedged against a
            # wall, both are ~0.
            sdx = px - self._last_screen_pos[0] - camera_delta[0]
            sdy = py - self._last_screen_pos[1] - camera_delta[1]
            if hypot(sdx, sdy) < cfg.stuck_speed * short:
                self._stuck_for += 1
            else:
                self._stuck_for = 0
        self._last_screen_pos = (px, py)
        stuck = self._stuck_for >= cfg.stuck_ticks

        # 3) Decide whether the current commitment is over.
        in_gas = bool(getattr(state, "in_gas", False))
        enemies = list(getattr(state, "enemy_positions", ()) or ())
        enemies_appeared = cfg.interrupt_on_enemy and enemies and not self._had_enemies
        self._had_enemies = bool(enemies)

        boxes = list(getattr(state, "box_positions", ()) or ())
        boxes_appeared = cfg.interrupt_on_box and boxes and not self._had_boxes
        self._had_boxes = bool(boxes)

        relatch = (
            self.waypoint is None
            or self.ticks_left <= 0
            or self.blocked
            or stuck
            or hypot(self.waypoint[0] - px, self.waypoint[1] - py) < cfg.arrive_frac * short
            or enemies_appeared
            or boxes_appeared
            or (in_gas and cfg.interrupt_on_gas)
        )
        if stuck:
            # Hand control back to the policy immediately rather than spending
            # the rest of the commitment shoving at whatever is in the way.
            self._stuck_for = 0
        if relatch:
            target = self.target_for(heading_idx, dist_idx, (px, py), rect)
            # Relocate the destination out of walls and gas AT LATCH TIME, not
            # per-tick inside A*. The latched point is what arrival is measured
            # against and what the observation reports, so leaving it inside a
            # wall or a gas cloud would mean committing to somewhere the agent
            # can never stand — it would walk to the edge and idle out the
            # timeout instead of arriving.
            self.waypoint = self._relocate(target, terrain)
            tier = max(0, min(len(cfg.commit_ticks) - 1, int(dist_idx)))
            self.tier = tier
            self.ticks_left = cfg.commit_ticks[tier]
            self.blocked = False

        tx, ty = self.waypoint

        # 4) Route there.
        if terrain is not None:
            move, self.path, self.blocked = self._astar_step(
                (px, py), (tx, ty), terrain, enemies, short)
            if self.blocked:
                # Nothing reachable in that direction — end the commitment now
                # so the policy gets to choose again on the next step instead of
                # grinding into a wall for the rest of the timeout.
                self.ticks_left = 0
        else:
            move = self._steer_direct((px, py), (tx, ty), state, rect, short)
            self.path = []

        # 5) Gas beats everything: walk out of it regardless of the plan.
        if in_gas:
            sx, sy = getattr(state, "gas_safe", (0.0, 0.0))
            if (sx, sy) != (0.0, 0.0):
                move = _unit(sx, sy)

        total = max(1, self.config.commit_ticks[self.tier])
        self._last_status = PlannerStatus(
            active=self.ticks_left > 0,
            progress=float(np.clip(1.0 - self.ticks_left / total, 0.0, 1.0)),
            waypoint_dx=(tx - px) / max(w, 1),
            waypoint_dy=(ty - py) / max(h, 1),
            distance=float(min(1.0, hypot(tx - px, ty - py) / max(short, 1))),
            blocked=self.blocked,
            replanned=relatch,
        )
        return move

    def _relocate(self, target, terrain):
        """Move a destination out of a wall or gas cloud, if it landed in one."""
        if terrain is None:
            return target
        from perception.getTerrain import to_grid, to_pixels

        occ = terrain["occupancy"]
        gh, gw = occ.shape
        rect, cell = terrain["rect"], terrain["cell"]
        gx, gy = to_grid(target, rect, cell)
        gx = int(np.clip(gx, 0, gw - 1))
        gy = int(np.clip(gy, 0, gh - 1))

        gas = terrain.get("gas")
        avoid = occ
        if gas is not None:
            avoid = occ | (np.asarray(gas) > self.config.gas_avoid_threshold)
        if not avoid[gy, gx]:
            return target
        # Prefer a cell that is neither wall nor gas; settle for merely not a
        # wall if the whole neighbourhood is gassy.
        cellxy = _nearest_free(avoid, (gx, gy)) or _nearest_free(occ, (gx, gy))
        return to_pixels(cellxy, rect, cell) if cellxy else target

    # ------------------------------------------------------------------ #
    def _cost_grid(self, terrain, enemies, short) -> np.ndarray:
        """Per-cell multiplier: cheap through cover, expensive near enemies.

        Blocked cells are np.inf so A* never expands them.
        """
        cfg = self.config
        occ = terrain["occupancy"]
        gh, gw = occ.shape
        rect, cell = terrain["rect"], terrain["cell"]

        cost = np.ones((gh, gw), dtype=np.float32)
        cost *= (1.0 - cfg.bush_discount * terrain["bush"])
        cost += cfg.unknown_cost * terrain["unknown"].astype(np.float32)

        gas = terrain.get("gas")
        if gas is not None and cfg.gas_cost > 0:
            cost += cfg.gas_cost * np.asarray(gas, dtype=np.float32)

        if enemies and cfg.enemy_cost > 0:
            radius = cfg.enemy_radius * short
            ys, xs = np.mgrid[0:gh, 0:gw]
            cx = rect[0] + (xs + 0.5) * cell[0]
            cy = rect[1] + (ys + 0.5) * cell[1]
            risk = np.zeros((gh, gw), dtype=np.float32)
            for ex, ey in enemies:
                d = np.hypot(cx - ex, cy - ey)
                risk = np.maximum(risk, np.clip(1.0 - d / max(radius, 1e-6), 0.0, 1.0))
            cost += cfg.enemy_cost * risk ** 2

        cost[occ] = np.inf
        return cost

    def _astar_step(self, player, target, terrain, enemies, short):
        """A* from the player's cell to the target's. Returns (move, path, blocked)."""
        from perception.getTerrain import to_grid, to_pixels

        occ = terrain["occupancy"]
        gh, gw = occ.shape
        rect, cell = terrain["rect"], terrain["cell"]
        cost = self._cost_grid(terrain, enemies, short)

        start = to_grid(player, rect, cell)
        goal = to_grid(target, rect, cell)
        start = (int(np.clip(start[0], 0, gw - 1)), int(np.clip(start[1], 0, gh - 1)))
        goal = (int(np.clip(goal[0], 0, gw - 1)), int(np.clip(goal[1], 0, gh - 1)))

        # A goal inside a wall is normal (the policy picks a raw direction, not a
        # legal cell). Snap to the nearest free cell instead of failing.
        #
        # Gas is treated as unwalkable HERE, for destination selection only —
        # deliberately choosing to stand in the cloud is never right. The route
        # to the destination still merely pays gas_cost, so crossing gas to
        # reach safety remains possible. If everything nearby is gassy we fall
        # back to ignoring gas, because refusing to move at all is worse.
        gas = terrain.get("gas")
        avoid = occ
        if gas is not None:
            avoid = occ | (np.asarray(gas) > self.config.gas_avoid_threshold)
        if avoid[goal[1], goal[0]]:
            goal = _nearest_free(avoid, goal) or _nearest_free(occ, goal) or goal

        came, reached = self._astar(cost, start, goal)
        if reached is None:
            return ((0.0, 0.0), [], True)

        path = _reconstruct(came, start, reached)
        # `blocked` means we could not get meaningfully closer, not merely that
        # the exact goal cell was unreachable — a partial route is still useful.
        blocked = len(path) < 2 and reached != goal

        aim = _lookahead_point(path, player, rect, cell, self.config.lookahead * short,
                               to_pixels)
        if aim is None:
            return ((0.0, 0.0), path, True)
        return (_unit(aim[0] - player[0], aim[1] - player[1]), path, blocked)

    def _astar(self, cost, start, goal):
        """Returns (came_from, best_node). Falls back to the closest node reached."""
        gh, gw = cost.shape
        if not np.isfinite(cost[start[1], start[0]]):
            # Standing in a cell we think is solid (perception hiccup, or the
            # player sprite covering its own tile). Treat it as free.
            cost = cost.copy()
            cost[start[1], start[0]] = 1.0

        # Admissible: no edge can be cheaper than the smallest possible
        # multiplier times a straight step.
        h_scale = 1.0 - self.config.bush_discount

        def heuristic(n):
            dx, dy = abs(n[0] - goal[0]), abs(n[1] - goal[1])
            return h_scale * (max(dx, dy) + (_SQRT2 - 1.0) * min(dx, dy))

        g = {start: 0.0}
        came = {}
        best, best_h = start, heuristic(start)
        open_heap = [(best_h, 0.0, start)]
        seen = set()

        while open_heap:
            _, gc, node = heapq.heappop(open_heap)
            if node in seen:
                continue
            seen.add(node)
            if node == goal:
                return came, goal
            hn = heuristic(node)
            if hn < best_h:
                best, best_h = node, hn

            nx0, ny0 = node
            for dx, dy, step in _NEIGHBOURS:
                nx, ny = nx0 + dx, ny0 + dy
                if not (0 <= nx < gw and 0 <= ny < gh):
                    continue
                c = cost[ny, nx]
                if not np.isfinite(c):
                    continue
                # No cutting diagonally through the gap between two walls.
                if dx and dy and (not np.isfinite(cost[ny0, nx]) or
                                  not np.isfinite(cost[ny, nx0])):
                    continue
                ng = gc + step * float(c)
                nb = (nx, ny)
                if ng < g.get(nb, float("inf")):
                    g[nb] = ng
                    came[nb] = node
                    heapq.heappush(open_heap, (ng + heuristic(nb), ng, nb))

        return came, (best if best != start else None)

    # ------------------------------------------------------------------ #
    def _steer_direct(self, player, target, state, rect, short):
        """Terrain-free fallback: head for the target, dodging visible entities.

        This is the old v4 planner's behaviour, kept for offline runs and for
        frames where terrain segmentation returns nothing.
        """
        cfg = self.config
        px, py = player
        desired = _unit(target[0] - px, target[1] - py)
        if desired == (0.0, 0.0):
            return desired

        step = cfg.lookahead * short
        best, best_score = desired, float("inf")
        base = atan2(desired[1], desired[0])
        for i in range(cfg.candidate_headings):
            angle = base + (i - cfg.candidate_headings // 2) * (2 * pi / cfg.candidate_headings)
            d = (cos(angle), sin(angle))
            nx, ny = px + d[0] * step, py + d[1] * step
            score = hypot(target[0] - nx, target[1] - ny)
            score += self._edge_cost(nx, ny, rect, step)
            score += self._entity_cost(nx, ny, getattr(state, "enemy_positions", ()), step, 1.6)
            score += self._entity_cost(nx, ny, getattr(state, "box_positions", ()), step, 0.7)
            if score < best_score:
                best, best_score = d, score
        return _unit(*best)

    def _edge_cost(self, x, y, rect, step):
        rx, ry, rw, rh = rect
        m = self.config.edge_margin
        edge = min((x - rx) / rw, 1 - (x - rx) / rw, (y - ry) / rh, 1 - (y - ry) / rh)
        return max(0.0, m - edge) * step * 12.0

    def _entity_cost(self, x, y, positions, step, weight):
        clearance = self.config.entity_clearance * step / self.config.lookahead
        total = 0.0
        for ex, ey in positions or ():
            d = hypot(x - ex, y - ey)
            if d < clearance:
                total += weight * ((clearance - d) / clearance) ** 2 * step * 3.0
        return total


# --- helpers ---------------------------------------------------------------- #
def _unit(x: float, y: float) -> Tuple[float, float]:
    length = hypot(x, y)
    return (0.0, 0.0) if length < 1e-6 else (x / length, y / length)


def _nearest_free(occ: np.ndarray, cell, max_r: int = 6):
    """Closest non-blocked cell to `cell`, searched in growing rings."""
    gh, gw = occ.shape
    cx, cy = cell
    for r in range(1, max_r + 1):
        best, best_d = None, None
        for y in range(max(0, cy - r), min(gh, cy + r + 1)):
            for x in range(max(0, cx - r), min(gw, cx + r + 1)):
                if max(abs(x - cx), abs(y - cy)) != r or occ[y, x]:
                    continue
                d = (x - cx) ** 2 + (y - cy) ** 2
                if best_d is None or d < best_d:
                    best, best_d = (x, y), d
        if best is not None:
            return best
    return None


def _reconstruct(came, start, goal):
    path = [goal]
    node = goal
    while node != start:
        node = came.get(node)
        if node is None:
            break
        path.append(node)
    path.reverse()
    return path


def _lookahead_point(path, player, rect, cell, lookahead, to_pixels):
    """Pure pursuit: the first point on the path at least `lookahead` px away.

    Steering at the path's far end rather than its next cell is what keeps
    movement smooth — aiming one cell ahead produces the same stair-stepping
    zigzag the 8-direction policy had.
    """
    if not path:
        return None
    for node in path[1:]:
        p = to_pixels(node, rect, cell)
        if hypot(p[0] - player[0], p[1] - player[1]) >= lookahead:
            return p
    return to_pixels(path[-1], rect, cell)


# Back-compat alias: the module previously exported `LocalPathPlanner`.
LocalPathPlanner = WaypointPlanner
