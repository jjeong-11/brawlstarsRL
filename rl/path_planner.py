"""
rl/path_planner.py
==================

Turns a polar waypoint choice into a joystick vector, via a latched destination
and A* over a persistent, camera-registered terrain map.

THE THREE-LAYER SPLIT
---------------------
    policy   picks WHERE to go, occasionally      (heading, distance)
    planner  keeps that destination fixed in the world and routes to it
    executor drives the joystick                  (dx, dy)

The middle layer is the whole point. In the original design the policy chose a
cell in a player-centred grid and the target was recomputed as
`player_pos + offset` every single tick -- so the destination moved with the
player and was never reached. That is a compass with extra steps, and it made
exploration *harder* (225 actions instead of 9) while delivering none of the
commitment that was supposed to fix the jittery movement.

Here the waypoint is LATCHED, and it is latched in MAP CELL COORDINATES on the
`rl/world_map.WorldMap`, which stays registered to the world by itself. The
policy's movement heads are IGNORED until the commitment ends. It ends when:

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

WHAT CHANGED, AND WHY
---------------------
Three failure modes drove the current design. Each one has a named fix below;
the fixes are listed here because none of them is obvious from the code alone.

1. WEDGING ON WALLS AND CORNERS.
   A* planned for a dimensionless point on a grid whose cells are about as wide
   as the brawler is. The optimal route between two points either side of a
   wall therefore runs exactly along the wall face, and around an outside
   corner it cuts the corner exactly. The agent, being a real object with a
   real width, grinds into it. The old code's answer was stuck DETECTION --
   notice after five motionless decisions and hand control back -- which is
   reactive, costs half a second every time, and never stops it happening
   again.
   FIX: `clearance_cost` / `hard_clearance`. The world map keeps a distance
   transform of free space, so every cell knows how far it is from the nearest
   obstacle. Cells closer than the agent's half-width are removed from the
   graph outright, and a soft cost decaying with clearance pulls routes toward
   the middle of gaps rather than the edges. Stuck detection is still there,
   but as a backstop rather than the primary mechanism.
   The inflation RELAXES automatically when it would disconnect the goal, so a
   genuinely one-cell-wide doorway stays usable.

2. SHORT-SIGHTEDNESS.
   The occupancy grid was rebuilt per frame in screen space, so nothing was
   remembered, ~20% of the play area behind the HUD was never resolved, and a
   waypoint that scrolled off-screen had its goal clamped to the grid border.
   FIX: `rl/world_map.WorldMap`, a fused map three screens wide that persists
   across frames. Waypoints live in its cell coordinates, so they no longer
   need advancing by the camera delta by hand and they stay valid off-screen.

3. WALKING INTO THE GAS.
   Three separate causes, all fixed here:
     (a) The escape direction came from `getGas.safe_vector`, which is
         `player - gas_centroid`. Showdown's cloud closes inward as a RING, and
         the centroid of a ring is the middle of the safe zone -- so "away from
         the centroid" points OUTWARD, straight into the gas. The bug was
         invisible early (a partial cloud has an off-centre centroid) and
         lethal late, and it overrode the entire A* plan whenever `in_gas` was
         set. FIX: `_escape_gas`, a Dijkstra outward from the player over the
         gas-weighted cost field that stops at the first genuinely safe,
         reachable cell. Correct for any cloud geometry, because it asks "where
         is safe ground" instead of assuming the cloud is a blob.
     (b) Cached gas was rolled by the camera delta with newly exposed edges
         filled with ZERO -- ground the agent had just fled came back into view
         labelled clean. FIX: the world map remembers gas and fuses it
         asymmetrically (fast up, slow down), because the cloud never retreats.
     (c) Gas was only computed when a terrain profile matched, so on an
         uncalibrated map the fallback steering had no gas awareness at all.
         FIX: geometry and gas are established independently of the terrain
         profile (`WorldMap.set_geometry`), and `_steer_direct` now costs gas.

A* also emits an any-angle path (`_string_pull`), because an 8-connected grid
can only express headings in 45-degree steps and the residual zigzag was
showing up in the joystick.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from math import atan2, cos, hypot, pi, sin
from typing import List, Optional, Tuple

import numpy as np

from .world_map import WorldMap, WorldMapConfig

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
    # Blend factor toward the new steering direction each tick. The planner
    # replans every step, and a one-cell change in the A* frontier can swing the
    # aim point by 45 degrees; feeding that straight to the joystick reproduces
    # the twitching the planner exists to remove. 0.55 follows real direction
    # changes within two or three ticks while filtering single-tick flapping.
    steer_smoothing: float = 0.55

    # --- agent footprint (see failure mode 1 in the module docstring) --- #
    # The brawler's half-width in CELLS. A cell is ~1/48 of the play width; a
    # brawler is roughly one cell across, so half-width ~0.5. 0.9 is
    # deliberately larger than that: the occupancy grid is coarse and its walls
    # are only accurate to about half a cell, so the footprint has to cover the
    # segmentation error as well as the sprite.
    agent_radius_cells: float = 0.9
    # Cells with less clearance than this are removed from the graph entirely,
    # UNLESS doing so would disconnect the goal (then it relaxes -- see
    # `_astar_with_relaxation`).
    hard_clearance: float = 0.9
    # Soft cost applied out to `clearance_falloff` cells from any wall. Keeps
    # routes off wall faces even where there is technically room.
    clearance_cost: float = 2.5
    clearance_falloff: float = 2.6

    # --- A* soft costs ---
    grid_size: Tuple[int, int] = (48, 27)
    enemy_radius: float = 0.20     # enemy influence radius, fraction of short side
    enemy_cost: float = 3.0        # extra cost at an enemy's exact position
    bush_discount: float = 0.15    # <1 multiplier for routing through cover
    unknown_cost: float = 0.4      # mild penalty for ground never yet observed
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
    # Gas is DILATED by this many cells before being costed, so cells the cloud
    # is about to reach are already expensive. The zone only ever shrinks, so
    # ground next to gas is not neutral ground -- it is ground that is about to
    # be gas. Without this the agent only reacts once the cloud is literally on
    # top of it, which in the endgame is far too late to walk out of.
    gas_dilate_cells: int = 2
    # How far the cloud is assumed to CONTINUE into ground that has never been
    # observed. A cloud does not stop at the edge of the screen, but the map
    # only knows about gas it has seen, so an unobserved cell beside a known
    # gas cell reads as perfectly clean — and A* will happily plan a detour
    # "around" a band by routing through the unknown ground just past the edge
    # of the view. That is a route straight into the part of the cloud we simply
    # have not looked at yet. Assuming the cloud continues into unexplored
    # ground is the conservative reading, and being wrong about it only costs a
    # slightly longer route.
    gas_unknown_dilate_cells: int = 7
    # Coverage below which a cell counts as safe to STOP on when escaping.
    gas_safe_threshold: float = 0.06
    # How far to search for safe ground when standing in gas. 40 cells is most
    # of the map at this scale; the search is Dijkstra over ~12k cells and costs
    # well under a millisecond, and giving up early is how you die at the edge
    # of a cloud that was two cells thick.
    gas_escape_radius: int = 40

    # --- staying away from the map border ---
    # Out-of-bounds decoration reads as unwalkable, so "how much blocked area
    # surrounds a cell" is a reliable proxy for "how close to the map edge is
    # this". Openness is the walkable fraction within openness_radius cells.
    #
    # This matters because the border is actively lethal in Showdown: the zone
    # closes inward, so the edge is where gas arrives FIRST. And the planner had
    # a bias straight towards it -- destinations aimed off-map get relocated to
    # the nearest legal cell, which is the border ring. Measured from a spot
    # near the edge, 46% of all 48 heading/distance combinations landed in the
    # outer 3 cells, and distinct destinations collapsed from 48 to 43.
    openness_radius: int = 4
    border_cost: float = 6.0        # A* penalty at zero openness
    min_destination_openness: float = 0.45   # pull a destination inward below this

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
    # Stuck detection, now a BACKSTOP behind the clearance costs rather than the
    # primary anti-wedging mechanism. Without it an agent wedged against a wall
    # keeps pushing for the WHOLE commitment because A* still reports a valid
    # route and nothing checks whether the route is actually being followed.
    stuck_ticks: int = 5              # consecutive barely-moving decisions
    stuck_speed: float = 0.015        # movement below this fraction of the
                                      # short side per tick counts as "stuck"

    # --- direct-steering fallback (no map at all) ---
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
    escaping_gas: bool = False   # the A* plan was overridden to flee the cloud
    mode: str = "idle"           # astar | escape | direct | lost


class WaypointPlanner:
    """Latches a destination, routes to it with A*, emits a joystick vector."""

    def __init__(self, config: Optional[PathPlannerConfig] = None,
                 world_config: Optional[WorldMapConfig] = None,
                 localize: bool = True):
        self.config = config or PathPlannerConfig()
        self.world = WorldMap(world_config)
        # Arena identification against the published layouts. Entirely
        # optional: without it (no map database, never enough explored, a skin
        # we cannot segment) the planner behaves exactly as it did before, on
        # the locally fused map alone.
        self.localizer = None
        if localize:
            try:
                from perception.localize import Localizer
                self.localizer = Localizer()
            except Exception:
                self.localizer = None
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self.world.reset()
        if self.localizer is not None:
            self.localizer.reset()      # the next match may be a different arena
        # The waypoint lives in WORLD MAP CELLS, not screen pixels. The map
        # keeps itself registered, so the destination needs no per-tick camera
        # correction and stays meaningful after it scrolls off screen.
        self.waypoint_cell: Optional[Tuple[float, float]] = None
        self.ticks_left: int = 0
        self.tier: int = 0
        self.path: List[Tuple[int, int]] = []       # map cells
        self.path_pixels: List[Tuple[float, float]] = []
        self.blocked: bool = False
        self._had_enemies: bool = False
        self._had_boxes: bool = False
        self._stuck_for: int = 0
        self._last_player_cell: Optional[Tuple[float, float]] = None
        self._smoothed_move: Tuple[float, float] = (0.0, 0.0)
        self._last_status = PlannerStatus()

    def status(self) -> PlannerStatus:
        return self._last_status

    # ------------------------------------------------------------------ #
    def waypoint_pixels(self) -> Optional[Tuple[float, float]]:
        """The latched destination in frame pixels, for drawing and rewards."""
        if self.waypoint_cell is None or not self.world.ready():
            return None
        return self.world.to_pixels(self.waypoint_cell)

    @property
    def waypoint(self) -> Optional[Tuple[float, float]]:
        """The destination in frame pixels.

        The latch itself now lives in map cells (`waypoint_cell`), because the
        world map keeps those registered by itself. This stays as the pixel
        view of the same thing: it is what callers, tests and the reward engine
        actually want, and it means "where is the agent going" has one answer
        rather than one per coordinate system.
        """
        return self.waypoint_pixels()

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

        # Clamped to the play rect only loosely: the world map extends past the
        # screen, so a destination just off the edge is now a legitimate place
        # to walk to rather than something that has to be folded back inside.
        mx, my = cfg.edge_margin * rect[2], cfg.edge_margin * rect[3]
        tx = min(max(tx, rect[0] - mx), rect[0] + rect[2] + mx)
        ty = min(max(ty, rect[1] - my), rect[1] + rect[3] + my)
        return (tx, ty)

    # ------------------------------------------------------------------ #
    def plan(self, heading_idx: int, dist_idx: int, state, frame_size,
             terrain=None, camera_delta=(0.0, 0.0), gas_grid=None
             ) -> Tuple[float, float]:
        """Advance one decision step and return a unit joystick vector.

        Parameters
        ----------
        heading_idx, dist_idx : the policy's movement action. CONSULTED ONLY
            when no commitment is active -- see the module docstring.
        state : rl.state.GameState (needs player_pos, enemy_positions, in_gas)
        terrain : dict from perception.getTerrain.find_terrain, or None
        camera_delta : (dx, dy) world scroll since the last step, from
            rl.camera_tracker.CameraTracker.update()
        gas_grid : gas coverage on the screen grid, used when `terrain` is None
            or carries no "gas" key. Gas detection does not need a terrain
            profile, so it must not be gated behind one.
        """
        cfg = self.config
        w, h = frame_size
        rect = terrain["rect"] if terrain else (self.world.rect or (0, 0, w, h))
        short = min(rect[2], rect[3])

        # --- keep the persistent map registered and fused ----------------- #
        gas = gas_grid
        if terrain is not None and terrain.get("gas") is not None:
            gas = terrain["gas"]
        self.world.update(terrain, gas, camera_delta)
        self._localize(camera_delta)

        player = getattr(state, "player_pos", None) if state is not None else None
        if player is None:
            # The anchor detector lost the player: we have no frame of reference
            # for a destination, so drop the latch and steer by raw heading.
            self.waypoint_cell = None
            self.ticks_left = 0
            self.path = []
            self.path_pixels = []
            self._last_status = PlannerStatus(active=False, mode="lost")
            angle = 2.0 * pi * (int(heading_idx) % cfg.n_headings) / cfg.n_headings
            return self._smooth((cos(angle), sin(angle)))

        px, py = float(player[0]), float(player[1])
        use_map = self.world.ready()

        if self.waypoint_cell is not None:
            self.ticks_left -= 1

        # --- am I actually going anywhere? -------------------------------- #
        # Measured in MAP CELLS, which are world-stable, so this no longer has
        # to subtract the camera delta by hand -- and it is correct even on
        # ticks where the camera estimate failed.
        stuck = self._update_stuck(px, py, short, use_map)

        # --- decide whether the current commitment is over ---------------- #
        in_gas = bool(getattr(state, "in_gas", False))
        enemies = list(getattr(state, "enemy_positions", ()) or ())
        enemies_appeared = cfg.interrupt_on_enemy and enemies and not self._had_enemies
        self._had_enemies = bool(enemies)

        boxes = list(getattr(state, "box_positions", ()) or ())
        boxes_appeared = cfg.interrupt_on_box and boxes and not self._had_boxes
        self._had_boxes = bool(boxes)

        arrived = False
        if self.waypoint_cell is not None and use_map:
            wp = self.world.to_pixels(self.waypoint_cell)
            arrived = hypot(wp[0] - px, wp[1] - py) < cfg.arrive_frac * short

        relatch = (
            self.waypoint_cell is None
            or not use_map
            or self.ticks_left <= 0
            or self.blocked
            or stuck
            or arrived
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
            # Relocate the destination out of walls, out of gas and away from
            # the border AT LATCH TIME, not per-tick inside A*. The latched
            # point is what arrival is measured against and what the observation
            # reports, so leaving it inside a wall or a gas cloud would mean
            # committing to somewhere the agent can never stand -- it would walk
            # to the edge and idle out the timeout instead of arriving.
            self.waypoint_cell = (self._relocate_cell(target, (px, py))
                                  if use_map else None)
            tier = max(0, min(len(cfg.commit_ticks) - 1, int(dist_idx)))
            self.tier = tier
            self.ticks_left = cfg.commit_ticks[tier]
            self.blocked = False

        # --- route there --------------------------------------------------- #
        mode = "direct"
        if use_map and self.waypoint_cell is not None:
            move, self.path, self.blocked = self._astar_step(
                (px, py), self.waypoint_cell, enemies, short)
            mode = "astar"
            if self.blocked:
                # Nothing reachable in that direction — end the commitment now
                # so the policy gets to choose again on the next step instead of
                # grinding into a wall for the rest of the timeout.
                self.ticks_left = 0
        else:
            target = self.target_for(heading_idx, dist_idx, (px, py), rect)
            move = self._steer_direct((px, py), target, state, rect, short)
            self.path = []

        # --- gas beats everything: walk OUT, by the shortest real route ---- #
        escaping = False
        if in_gas:
            escape = self._escape_gas((px, py)) if use_map else None
            if escape is None:
                # No map, or no reachable safe cell found. Fall back to the
                # perception hint, which is better than nothing but is only a
                # centroid direction -- see failure mode 3(a) in the module
                # docstring for why it cannot be trusted on its own.
                sx, sy = getattr(state, "gas_safe", (0.0, 0.0))
                if (sx, sy) != (0.0, 0.0):
                    move, escaping = _unit(sx, sy), True
            else:
                move, self.path = escape
                escaping, mode = True, "escape"

        self.path_pixels = [self.world.to_pixels(c) for c in self.path] if use_map else []

        move = self._smooth(move)
        self._publish_status(px, py, w, h, short, relatch, escaping, mode, use_map)
        return move

    # ------------------------------------------------------------------ #
    def _localize(self, camera_delta) -> None:
        """Try to identify the arena, and adopt its layout once we have.

        Runs on the ACCUMULATED map rather than this frame, because a single
        viewport does not carry enough structure to pick one arena out of 71 --
        see the measured table in `perception/localize.py`. That also means
        this does nothing for the first few seconds of a match, which is fine:
        the local map is what the planner used before this existed.
        """
        if self.localizer is None or not self.world.ready():
            return
        cw, ch = self.world.cell
        delta_cells = (camera_delta[0] / max(cw, 1e-6),
                       camera_delta[1] / max(ch, 1e-6))
        try:
            pose = self.localizer.update(self.world.occ_p > self.world.config.occ_threshold,
                                         self.world.seen > 0.15, delta_cells)
        except Exception:
            return
        if pose is None:
            if self.world.have_reference and self.localizer.pose is None:
                self.world.clear_reference()   # lock was abandoned
            return
        self.world.apply_reference(pose)

    def _smooth(self, move) -> Tuple[float, float]:
        """Low-pass the joystick direction (see `steer_smoothing`)."""
        a = self.config.steer_smoothing
        if move == (0.0, 0.0):
            self._smoothed_move = (0.0, 0.0)
            return move
        sx, sy = self._smoothed_move
        if (sx, sy) == (0.0, 0.0):
            self._smoothed_move = move
            return move
        self._smoothed_move = _unit(sx + a * (move[0] - sx), sy + a * (move[1] - sy))
        return self._smoothed_move

    def _update_stuck(self, px, py, short, use_map) -> bool:
        cfg = self.config
        cur = self.world.to_cell((px, py)) if use_map else (px / max(short, 1), py / max(short, 1))
        if self._last_player_cell is not None:
            dx = cur[0] - self._last_player_cell[0]
            dy = cur[1] - self._last_player_cell[1]
            # stuck_speed is a fraction of the short side; in cells that is the
            # same fraction of the grid's short dimension.
            limit = cfg.stuck_speed * (self.world.screen_gh if use_map else 1.0)
            if hypot(dx, dy) < limit:
                self._stuck_for += 1
            else:
                self._stuck_for = 0
        self._last_player_cell = cur
        return self._stuck_for >= cfg.stuck_ticks

    def _publish_status(self, px, py, w, h, short, relatch, escaping, mode, use_map) -> None:
        wp = self.waypoint_pixels()
        if wp is None:
            wp = (px, py)
        tx, ty = wp
        total = max(1, self.config.commit_ticks[self.tier])
        self._last_status = PlannerStatus(
            active=self.ticks_left > 0 and use_map,
            progress=float(np.clip(1.0 - self.ticks_left / total, 0.0, 1.0)),
            waypoint_dx=(tx - px) / max(w, 1),
            waypoint_dy=(ty - py) / max(h, 1),
            distance=float(min(1.0, hypot(tx - px, ty - py) / max(short, 1))),
            blocked=self.blocked,
            replanned=relatch,
            escaping_gas=escaping,
            mode=mode,
        )

    # ------------------------------------------------------------------ #
    def _relocate_cell(self, target_px, player_px) -> Tuple[float, float]:
        """Make a destination legal, and never park it on the map border.

        Three corrections, in order:

        1. A destination in a CLOSED-IN area -- the outer ring, a dead end --
           is pulled back along the ray toward the player until it reaches open
           ground. This is the important one: headings that point off-map used
           to be relocated to the nearest legal cell, which is by definition
           the border ring, so a large share of all destinations collapsed onto
           the edge. In Showdown the edge is where the gas arrives first, so
           the planner was reliably steering into the most lethal part of the
           map.
        2. A destination inside a wall, inside gas, or too close to a wall for
           the agent to physically stand there, is moved to the nearest cell
           that is none of those.
        3. Failing that, it settles for merely not-a-wall.
        """
        cfg = self.config
        world = self.world
        target = world.to_cell(target_px)
        px, py = world.to_cell(player_px)

        occ = world.occupancy
        clear = world.clearance()
        gassy = world.gas > cfg.gas_avoid_threshold
        # "Somewhere the agent can actually stand": not a wall, not inside the
        # cloud, and with room for its own width.
        standable = (~occ) & (~gassy) & (clear >= cfg.agent_radius_cells)

        # (1) walk back toward the player until the ground is open enough
        if cfg.min_destination_openness > 0 and world.has_walls():
            open_frac = _openness(occ, cfg.openness_radius, world.seen > 0.15)
            best, best_open = target, -1.0
            for frac in (1.0, 0.8, 0.62, 0.48, 0.36, 0.26, 0.18):
                cx = px + (target[0] - px) * frac
                cy = py + (target[1] - py) * frac
                gx, gy = world.clamp_cell((cx, cy))
                o = float(open_frac[gy, gx])
                if o >= cfg.min_destination_openness:
                    target = (cx, cy)
                    break
                if o > best_open:
                    best, best_open = (cx, cy), o
            else:
                # Nowhere along the ray is open: the player is already boxed in,
                # so aim at the most open point available rather than the edge.
                target = best

        # (2)/(3) legality
        gx, gy = world.clamp_cell(target)
        if standable[gy, gx]:
            return (float(gx), float(gy))
        cell = (_nearest_true(standable, (gx, gy))
                or _nearest_true(~occ, (gx, gy))
                or (gx, gy))
        return (float(cell[0]), float(cell[1]))

    # ------------------------------------------------------------------ #
    def _gas_field(self) -> np.ndarray:
        """Gas coverage to plan against: what was seen, plus where it is going.

        Two extrapolations, for two different reasons:

          * `gas_dilate_cells` everywhere — the zone only ever shrinks, so a
            clear cell beside the cloud is not neutral ground, it is ground that
            is about to be gas.
          * `gas_unknown_dilate_cells` into UNEXPLORED cells only — the cloud
            does not stop at the edge of the view, and without this A* routes
            "around" a band by cutting through the unobserved ground just past
            the screen edge, which is the same cloud we have not looked at.
        """
        cfg = self.config
        world = self.world
        near = _dilate_grid(world.gas, cfg.gas_dilate_cells)
        if cfg.gas_unknown_dilate_cells > cfg.gas_dilate_cells:
            far = _dilate_grid(world.gas, cfg.gas_unknown_dilate_cells)
            # Only into ground the GAS DETECTOR has never covered. Keying this
            # off terrain exploration instead would smear an imaginary cloud
            # across the whole map on any skin without a colour profile, where
            # terrain is unreadable but gas is read perfectly well.
            near = np.where(world.gas_unobserved, np.maximum(near, far), near)
        return near.astype(np.float32)

    def _cost_grid(self, enemies, short, inflate: bool = True) -> np.ndarray:
        """Per-cell multiplier over the WHOLE world map.

        Blocked cells are np.inf so A* never expands them. `inflate` applies the
        agent-footprint clearance rule; `_astar_with_relaxation` retries with it
        off when it would disconnect the goal.
        """
        cfg = self.config
        world = self.world
        occ = world.occupancy
        clear = world.clearance()

        cost = np.ones(occ.shape, dtype=np.float32)
        cost *= (1.0 - cfg.bush_discount * world.bush)
        cost += cfg.unknown_cost * world.unexplored.astype(np.float32)

        # --- agent footprint (failure mode 1) --- #
        # Soft term: expensive right next to a wall, free once there is room.
        if cfg.clearance_cost > 0 and cfg.clearance_falloff > 0:
            near = np.clip(1.0 - clear / cfg.clearance_falloff, 0.0, 1.0)
            cost += cfg.clearance_cost * near ** 2

        if cfg.gas_cost > 0:
            cost += cfg.gas_cost * self._gas_field()

        # Push routes away from the map edge (see PathPlannerConfig.border_cost).
        if cfg.border_cost > 0 and world.has_walls():
            open_frac = _openness(occ, cfg.openness_radius, world.seen > 0.15)
            cost += cfg.border_cost * (1.0 - open_frac) ** 2

        if enemies and cfg.enemy_cost > 0:
            radius_cells = max(1e-6, cfg.enemy_radius * short / world.cell_size_px())
            ys, xs = np.mgrid[0:occ.shape[0], 0:occ.shape[1]]
            risk = np.zeros(occ.shape, dtype=np.float32)
            for ex, ey in enemies:
                ecx, ecy = world.to_cell((ex, ey))
                d = np.hypot(xs - ecx, ys - ecy)
                risk = np.maximum(risk, np.clip(1.0 - d / radius_cells, 0.0, 1.0))
            cost += cfg.enemy_cost * risk ** 2

        cost[occ] = np.inf
        # Outside the arena is not expensive, it is impossible. Known exactly
        # once the map is identified; all-False otherwise, so this is a no-op
        # until then.
        oob = world.out_of_bounds
        if oob.any():
            cost[oob] = np.inf
        if inflate and cfg.hard_clearance > 0:
            cost[clear < cfg.hard_clearance] = np.inf
        return cost

    def _astar_step(self, player_px, goal_cell, enemies, short):
        """A* from the player's cell to the goal cell. -> (move, path, blocked)."""
        world = self.world
        start = world.clamp_cell(world.to_cell(player_px))
        goal = world.clamp_cell(goal_cell)

        came, reached, cost = self._astar_with_relaxation(start, goal, enemies, short)
        if reached is None:
            return ((0.0, 0.0), [], True)

        path = _reconstruct(came, start, reached)
        path = _string_pull(path, cost)
        # `blocked` means we could not get meaningfully closer, not merely that
        # the exact goal cell was unreachable — a partial route is still useful.
        blocked = len(path) < 2 and reached != goal

        aim = self._lookahead_point(path, player_px, short)
        if aim is None:
            return ((0.0, 0.0), path, True)
        return (_unit(aim[0] - player_px[0], aim[1] - player_px[1]), path, blocked)

    def _astar_with_relaxation(self, start, goal, enemies, short):
        """A* with the agent footprint on; retry without it if that fails.

        Inflating obstacles by the agent's half-width is what stops corner
        clipping, but it can also seal a legitimate one-cell doorway — and a
        planner that refuses to move is worse than one that scrapes a wall. So
        the strict pass runs first, and only if it cannot reach the goal do we
        fall back to the un-inflated graph. Costs one extra search in the rare
        case, nothing at all in the common one.
        """
        cost = self._cost_grid(enemies, short, inflate=True)
        came, reached = self._astar(cost, start, goal)
        if reached == goal:
            return came, reached, cost

        relaxed = self._cost_grid(enemies, short, inflate=False)
        came2, reached2 = self._astar(relaxed, start, goal)
        if reached2 == goal or (reached is None and reached2 is not None):
            return came2, reached2, relaxed
        return came, reached, cost

    # Padding around the start/goal bounding box that A* may expand into, in
    # cells. The map is 144x81 = 11664 cells but a waypoint is at most ~15 cells
    # away (0.55 of the short side), so an unbounded search spends most of its
    # time exploring ground in the opposite direction. 14 cells of slack is
    # roughly a quarter-screen of room to detour around a wall, which is more
    # than any real obstacle needs; a route that genuinely has to go further
    # than that is one the commitment would time out on anyway.
    _SEARCH_PAD = 14

    def _astar(self, cost, start, goal):
        """Returns (came_from, best_node). Falls back to the closest node reached."""
        gh, gw = cost.shape
        # Bound the search to a box around start and goal (see _SEARCH_PAD).
        pad = self._SEARCH_PAD
        bx0 = max(0, min(start[0], goal[0]) - pad)
        bx1 = min(gw - 1, max(start[0], goal[0]) + pad)
        by0 = max(0, min(start[1], goal[1]) - pad)
        by1 = min(gh - 1, max(start[1], goal[1]) + pad)
        if not np.isfinite(cost[start[1], start[0]]):
            # Standing in a cell we think is solid (perception hiccup, the
            # player sprite covering its own tile, or our own inflation). Treat
            # it and its immediate neighbours as free so the search can escape.
            cost = cost.copy()
            y0, y1 = max(0, start[1] - 1), min(gh, start[1] + 2)
            x0, x1 = max(0, start[0] - 1), min(gw, start[0] + 2)
            patch = cost[y0:y1, x0:x1]
            patch[~np.isfinite(patch)] = 1.0

        # The inner loop reads the cost grid several thousand times per search.
        # Scalar indexing into a numpy array costs ~150ns each (it builds a
        # 0-d array and boxes it), against ~25ns for a flat Python list, and
        # the search is 95% of the planner's runtime — so flatten once up
        # front. Measured on a 144x81 map: 9.0ms -> 2.4ms per call, with
        # identical routes.
        flat = cost.ravel().tolist()
        inf = float("inf")

        # Admissible: no edge can be cheaper than the smallest possible
        # multiplier times a straight step.
        h_scale = 1.0 - self.config.bush_discount
        gx_goal, gy_goal = goal

        def heuristic(nx, ny):
            dx, dy = abs(nx - gx_goal), abs(ny - gy_goal)
            if dx < dy:
                dx, dy = dy, dx
            return h_scale * (dx + (_SQRT2 - 1.0) * dy)

        g = {start: 0.0}
        came = {}
        best, best_h = start, heuristic(start[0], start[1])
        open_heap = [(best_h, 0.0, start)]
        seen = set()

        while open_heap:
            _, gc, node = heapq.heappop(open_heap)
            if node in seen:
                continue
            seen.add(node)
            if node == goal:
                return came, goal
            nx0, ny0 = node
            hn = heuristic(nx0, ny0)
            if hn < best_h:
                best, best_h = node, hn

            row0 = ny0 * gw
            for dx, dy, step in _NEIGHBOURS:
                nx, ny = nx0 + dx, ny0 + dy
                if nx < bx0 or nx > bx1 or ny < by0 or ny > by1:
                    continue
                c = flat[ny * gw + nx]
                if c >= inf:
                    continue
                # No cutting diagonally through the gap between two walls.
                if dx and dy and (flat[row0 + nx] >= inf or
                                  flat[ny * gw + nx0] >= inf):
                    continue
                ng = gc + step * c
                nb = (nx, ny)
                if ng < g.get(nb, inf):
                    g[nb] = ng
                    came[nb] = node
                    heapq.heappush(open_heap, (ng + heuristic(nx, ny), ng, nb))

        return came, (best if best != start else None)

    # ------------------------------------------------------------------ #
    def _escape_gas(self, player_px):
        """Shortest route from inside the cloud to genuinely safe ground.

        THIS REPLACES `getGas.safe_vector`, which was `player - gas_centroid`.
        Showdown's cloud closes inward as a ring; the centroid of a ring is the
        middle of the SAFE ZONE, so that vector points outward -- into the gas --
        exactly when the ring has closed enough to matter. See failure mode
        3(a) in the module docstring.

        Dijkstra outward from the player over a gas-weighted cost field, halting
        at the first cell whose gas coverage is below `gas_safe_threshold`. It
        asks "where is the nearest safe ground I can actually walk to", which is
        the right question regardless of the cloud's shape, and because gas is a
        cost rather than a wall it will cut through a thin band of cloud when
        that is genuinely the shortest way out.

        Returns (move, path_cells) or None if no safe cell is reachable.
        """
        cfg = self.config
        world = self.world
        gas = world.gas
        occ = world.occupancy
        clear = world.clearance()

        start = world.clamp_cell(world.to_cell(player_px))
        gh, gw = gas.shape

        # Walls block; the agent footprint is a soft preference here rather than
        # a hard rule, because refusing a tight gap while standing in poison is
        # not a trade worth making.
        cost = 1.0 + cfg.gas_cost * self._gas_field()
        cost += cfg.clearance_cost * np.clip(1.0 - clear / max(cfg.clearance_falloff, 1e-6),
                                             0.0, 1.0) ** 2
        cost[occ] = np.inf

        safe = (gas < cfg.gas_safe_threshold) & (~occ)
        if not safe.any():
            return None

        came = {}
        g = {start: 0.0}
        heap = [(0.0, start)]
        seen = set()
        limit = float(cfg.gas_escape_radius)
        goal = None
        while heap:
            gc, node = heapq.heappop(heap)
            if node in seen:
                continue
            seen.add(node)
            if safe[node[1], node[0]] and node != start:
                goal = node
                break
            nx0, ny0 = node
            if hypot(nx0 - start[0], ny0 - start[1]) > limit:
                continue
            for dx, dy, step in _NEIGHBOURS:
                nx, ny = nx0 + dx, ny0 + dy
                if not (0 <= nx < gw and 0 <= ny < gh):
                    continue
                c = cost[ny, nx]
                if not np.isfinite(c):
                    continue
                ng = gc + step * float(c)
                nb = (nx, ny)
                if ng < g.get(nb, float("inf")):
                    g[nb] = ng
                    came[nb] = node
                    heapq.heappush(heap, (ng, nb))

        if goal is None:
            return None
        path = _string_pull(_reconstruct(came, start, goal), cost)
        aim = self._lookahead_point(path, player_px,
                                    min(world.screen_gw, world.screen_gh)
                                    * world.cell_size_px())
        if aim is None:
            return None
        return (_unit(aim[0] - player_px[0], aim[1] - player_px[1]), path)

    # ------------------------------------------------------------------ #
    def _lookahead_point(self, path, player_px, short):
        """Pure pursuit: the first point on the path at least `lookahead` away.

        Steering at the path's far end rather than its next cell is what keeps
        movement smooth — aiming one cell ahead produces the same stair-stepping
        zigzag the 8-direction policy had.
        """
        if not path:
            return None
        lookahead = self.config.lookahead * short
        for node in path[1:]:
            p = self.world.to_pixels(node)
            if hypot(p[0] - player_px[0], p[1] - player_px[1]) >= lookahead:
                return p
        return self.world.to_pixels(path[-1])

    # ------------------------------------------------------------------ #
    def _steer_direct(self, player, target, state, rect, short):
        """Map-free fallback: head for the target, dodging what we can see.

        Used before the first terrain observation and on frames where nothing
        at all could be established. Unlike the previous version this also
        costs GAS, using the per-side coverage the gas detector reports — that
        detector needs no terrain profile, so there is no reason for the
        fallback to be blind to the cloud (failure mode 3(c)).
        """
        cfg = self.config
        px, py = player
        desired = _unit(target[0] - px, target[1] - py)
        if desired == (0.0, 0.0):
            return desired

        step = cfg.lookahead * short
        gl, gr, gt, gb = getattr(state, "gas_sides", (0.0, 0.0, 0.0, 0.0)) or (0, 0, 0, 0)
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
            # Discourage heading into whichever half-planes hold the most gas.
            gas_bias = (gr if d[0] > 0 else gl) * abs(d[0]) + \
                       (gb if d[1] > 0 else gt) * abs(d[1])
            score += cfg.gas_cost * gas_bias * step
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
def _openness(occ: np.ndarray, radius: int, seen: np.ndarray = None) -> np.ndarray:
    """Walkable fraction within `radius` cells — high in the open, low at edges.

    Out-of-bounds decoration segments as unwalkable, so this doubles as a
    "distance from the map border" field without needing any map knowledge.

    `seen` MATTERS, and leaving it out silently breaks border avoidance. The
    map is three screens wide but only the current screen has been observed, and
    unobserved cells sit below the blocked threshold — so without weighting they
    vote "open". A destination pressed right against a wall of out-of-bounds
    decoration then measures as perfectly open, because the never-looked-at
    ground beyond it outnumbers the blocked cells beside it. Measured on the
    synthetic hard-border test: 25% of destinations landed in the out-of-bounds
    ring with unweighted openness, 0% with it weighted.

    So openness is the free fraction of the cells actually OBSERVED nearby, and
    ground with no observed neighbours at all comes back neutral (0.5) rather
    than confidently open.
    """
    import cv2
    k = 2 * int(radius) + 1
    if seen is None:
        free = (~occ).astype(np.float32)
        return cv2.blur(free, (k, k), borderType=cv2.BORDER_CONSTANT)

    weight = seen.astype(np.float32)
    free = ((~occ).astype(np.float32)) * weight
    num = cv2.blur(free, (k, k), borderType=cv2.BORDER_CONSTANT)
    den = cv2.blur(weight, (k, k), borderType=cv2.BORDER_CONSTANT)
    out = np.full(occ.shape, 0.5, np.float32)
    ok = den > 1e-3
    np.divide(num, den, out=out, where=ok)
    return np.clip(out, 0.0, 1.0)


def _dilate_grid(grid: np.ndarray, cells: int) -> np.ndarray:
    """Spread a 0..1 field outward by `cells`, keeping the peak value."""
    if cells <= 0:
        return grid
    import cv2
    k = 2 * int(cells) + 1
    return cv2.dilate(np.ascontiguousarray(grid, dtype=np.float32),
                      np.ones((k, k), np.uint8))


def _unit(x: float, y: float) -> Tuple[float, float]:
    length = hypot(x, y)
    return (0.0, 0.0) if length < 1e-6 else (x / length, y / length)


def _nearest_true(mask: np.ndarray, cell, max_r: int = 8):
    """Closest cell where `mask` is True, searched in growing rings."""
    gh, gw = mask.shape
    cx, cy = cell
    for r in range(1, max_r + 1):
        best, best_d = None, None
        for y in range(max(0, cy - r), min(gh, cy + r + 1)):
            for x in range(max(0, cx - r), min(gw, cx + r + 1)):
                if max(abs(x - cx), abs(y - cy)) != r or not mask[y, x]:
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


def _line_of_sight(free: np.ndarray, a, b) -> bool:
    """Bresenham visibility test between two cells over a boolean free mask."""
    x0, y0 = int(a[0]), int(a[1])
    x1, y1 = int(b[0]), int(b[1])
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    err = dx - dy
    gh, gw = free.shape
    while True:
        if not (0 <= x0 < gw and 0 <= y0 < gh) or not free[y0, x0]:
            return False
        if x0 == x1 and y0 == y1:
            return True
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def _string_pull(path, cost: np.ndarray, tolerance: float = 1.15):
    """Drop intermediate cells that the path does not need (any-angle route).

    An 8-connected grid can only express headings in 45-degree steps, so even a
    perfectly straight corridor produces a staircase and the joystick inherits
    the zigzag. Greedily skipping to the furthest cell still in line of sight
    turns the staircase back into the straight line it was approximating, at
    the cost of one Bresenham walk per kept vertex.

    SHORTCUTS ARE COST-AWARE, NOT JUST WALL-AWARE. Testing visibility against
    walls alone is wrong here and quietly undoes the planner's soft costs: gas
    is deliberately passable-but-expensive, so A* would correctly detour around
    a cloud and then string-pulling would notice both ends were mutually
    visible and straighten the route right back through it. (This is not
    hypothetical -- it broke the "gas costs but does not wall" test the moment
    smoothing was added.)

    So a cell may only be crossed if it is no more expensive than the most
    expensive cell A* already chose to walk through, times a small tolerance.
    A route that avoids gas therefore cannot be shortcut into gas, while a
    route that had to cross gas anyway is still free to straighten.
    """
    if len(path) < 3:
        return path
    # The START cell is excluded from the budget. The agent is already standing
    # there, so its cost is not something A* chose to accept -- and when the
    # agent is standing at the edge of a gas cloud (which is precisely when this
    # matters) that one cell is expensive enough on its own to license
    # shortcuts through the entire cloud. Measured: with the start cell
    # included, a correctly-detouring 26-cell route around a gas band was
    # straightened back into a 2-cell dash straight through it.
    vals = [cost[c[1], c[0]] for c in path[1:] if np.isfinite(cost[c[1], c[0]])]
    limit = (max(vals) * tolerance) if vals else np.inf
    passable = np.isfinite(cost) & (cost <= limit)

    out = [path[0]]
    i = 0
    n = len(path)
    while i < n - 1:
        j = n - 1
        while j > i + 1 and not _line_of_sight(passable, path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out


# Back-compat alias: the module previously exported `LocalPathPlanner`.
LocalPathPlanner = WaypointPlanner
