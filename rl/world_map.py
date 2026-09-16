"""
rl/world_map.py
===============

A persistent, camera-registered map of the terrain around the player.

WHY THIS EXISTS
---------------
`perception/getTerrain.find_terrain` returns a fresh 48x27 occupancy grid for
the CURRENT frame, in SCREEN space. That has three consequences the planner
used to live with, and all three of them hurt:

  1. NO MEMORY. Every frame the map is rebuilt from scratch, so a wall the
     agent walked past a moment ago does not exist any more. Worse, roughly a
     fifth of the play area is permanently hidden behind the HUD (joystick,
     attack button, cube counter), and those cells are marked `unknown` and
     treated as free forever -- they are never filled in, even though the
     camera scrolls them out from under the HUD several times a second.

  2. NO REACH. A destination that scrolls off the edge of the screen has no
     terrain behind it, so A* clamped the goal to the grid border. The "far"
     distance tier is 0.55 of the short side, which on a 20:9 panel is further
     than half the screen height -- so far waypoints regularly degenerated
     into "walk to the edge of the view".

  3. NO GAS MEMORY. `env._gas_for` rolled its cached gas grid by the camera
     delta and filled the newly exposed edge with ZERO -- i.e. "no gas here".
     Ground the agent had just fled because it was full of gas came back into
     view labelled clean. In the endgame, when the cloud is closing and the
     safe pocket is small, that is the difference between living and dying.

This module fixes all three by keeping ONE map that persists across frames and
is kept registered to the world using the camera translation that
`rl/camera_tracker.py` already measures for waypoint latching.

HOW REGISTRATION WORKS
----------------------
The map array never moves (except for the rare recentre below). What moves is
the ORIGIN: the map-cell coordinate that the play rect's top-left corner
currently sits at. Each tick the camera reports how far world content slid
across the screen; the screen window therefore slid the opposite way over the
world, so `origin -= camera_delta_in_cells`.

Tracking a float origin rather than `np.roll`-ing the array every tick matters
for accuracy. Rolling can only move by whole cells, so at a typical scroll of
a few pixels per tick the sub-cell remainder is either discarded (the map
drifts out of registration) or accumulated by hand. Moving the origin keeps
full sub-cell precision for free, and the array is only ever rolled when the
window approaches an edge -- a few times a match instead of ten times a second.

FUSION
------
Occupancy is stored as a probability and updated with an exponential moving
average over the cells the current frame could actually see. Cells hidden
behind the HUD, or outside the view, are simply not updated -- they keep
whatever earlier frames established, which is the entire point.

Gas is fused ASYMMETRICALLY: it rises fast and falls slowly. Showdown's cloud
only ever grows, so a cell that reads gassy once is almost certainly gassy
from then on, while a cell reading clean is often just a detection dropout in
a thin patch. Treating those two symmetrically is what let the old cached grid
flicker between "deadly" and "fine" from frame to frame.

CLEARANCE
---------
`clearance()` is a distance transform over free space, in cells. This is what
the planner uses to stop clipping corners: a Brawl Stars brawler is roughly as
wide as one grid cell, but A* plans for a dimensionless point, so an optimal
route runs exactly along the wall face and the agent grinds into it. Costing
(and, close in, forbidding) low-clearance cells routes down the middle of gaps
instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class WorldMapConfig:
    # How many screen-widths of map to keep. 3x the 48x27 screen grid is
    # 144x81 cells = 12k cells: still trivial to distance-transform (~0.1ms)
    # but it holds a full screen of context in every direction, which is more
    # than the longest commitment can travel.
    expand: int = 3

    # --- fusion rates, per observation ---
    occ_alpha: float = 0.35        # EMA rate toward the current frame
    occ_threshold: float = 0.55    # p(blocked) above this counts as a wall
    # Cells never observed sit at this probability. Deliberately BELOW the
    # threshold: unexplored ground must be passable or the agent would refuse
    # to walk anywhere it has not already been, which in a scrolling view means
    # refusing to explore at all.
    occ_prior: float = 0.35

    gas_rise: float = 0.75         # EMA rate when the new reading is HIGHER
    gas_fall: float = 0.10         # ... and when it is LOWER (see module docs)
    # Per-tick decay for cells NOT currently visible. Deliberately gentle: at
    # 0.997 and a 0.1s tick, remembered gas keeps ~74% of its value after 10
    # seconds, which is the right order for a cloud that never retreats.
    gas_memory_decay: float = 0.997

    # Recentre once the screen window comes within this fraction of a map edge.
    recentre_margin: float = 0.18


class WorldMap:
    """Fused occupancy + gas over a window of the world, in grid cells.

    All public coordinates are FRAME PIXELS unless a name says `_cell`.
    """

    def __init__(self, config: Optional[WorldMapConfig] = None):
        self.config = config or WorldMapConfig()
        self._reset_arrays(48, 27)

    # ------------------------------------------------------------------ #
    def _reset_arrays(self, screen_gw: int, screen_gh: int) -> None:
        cfg = self.config
        self.screen_gw, self.screen_gh = int(screen_gw), int(screen_gh)
        self.gw = int(screen_gw * cfg.expand)
        self.gh = int(screen_gh * cfg.expand)
        self.occ_p = np.full((self.gh, self.gw), cfg.occ_prior, np.float32)
        self.bush = np.zeros((self.gh, self.gw), np.float32)
        self.gas = np.zeros((self.gh, self.gw), np.float32)
        self.seen = np.zeros((self.gh, self.gw), np.float32)
        # Gas observation is tracked SEPARATELY from terrain observation,
        # because the two fail independently: on a map with no colour profile
        # the gas detector works perfectly while `find_terrain` returns nothing
        # at all. Folding them together would mean "we have never looked at
        # this ground" whenever the walls were unreadable, and the planner
        # extrapolates unseen gas outward from known gas -- so it would invent
        # a cloud over the entire map on every uncalibrated skin.
        self.seen_gas = np.zeros((self.gh, self.gw), np.float32)
        # Map-cell coordinate of the play rect's top-left corner.
        self.origin = np.array([(self.gw - self.screen_gw) / 2.0,
                                (self.gh - self.screen_gh) / 2.0], np.float64)
        self.rect: Optional[Tuple[int, int, int, int]] = None
        self.cell: Tuple[float, float] = (1.0, 1.0)
        self._clearance: Optional[np.ndarray] = None
        self._clearance_stamp = -1
        self._stamp = 0
        self.have_terrain = False
        # Ground-truth layout from perception.localize, once the arena is
        # identified. None until then -- everything works without it.
        self.reference_occ: Optional[np.ndarray] = None
        self.reference_in_bounds: Optional[np.ndarray] = None
        self.have_reference = False

    def reset(self) -> None:
        """Forget everything (new match, possibly a new map)."""
        self._reset_arrays(self.screen_gw, self.screen_gh)

    # ------------------------------------------------------------------ #
    def set_geometry(self, rect, grid_size) -> None:
        """Establish the play rect / cell size WITHOUT a terrain observation.

        `perception.getTerrain.play_rect` works on any frame -- it only looks
        for the black letterbox padding -- while `find_terrain` needs a
        matching colour profile and returns None without one. Gas detection is
        likewise profile-independent (the cloud is an engine overlay drawn the
        same way on every map, see `perception/getGas.py`).

        So on a map with no profile we still know exactly where the play area
        is and exactly where the gas is; only the walls are missing. Letting
        geometry be set separately is what allows the gas half of the map to
        keep working there, instead of the planner going completely blind
        because one of the two inputs failed.
        """
        gw, gh = int(grid_size[0]), int(grid_size[1])
        if (gw, gh) != (self.screen_gw, self.screen_gh):
            self._reset_arrays(gw, gh)
        self.rect = rect
        self.cell = (rect[2] / gw, rect[3] / gh)

    def update(self, terrain: Optional[dict], gas_grid: Optional[np.ndarray],
               camera_delta=(0.0, 0.0)) -> None:
        """Slide the map by the camera, then fuse this frame's observation.

        `terrain` is a `find_terrain` dict, or None when no profile matched
        (the map then simply ages -- it does not get wiped, because the map did
        not change just because we failed to classify one frame).
        `gas_grid` is a (screen_gh, screen_gw) 0..1 coverage grid, or None.
        """
        self._stamp += 1

        # Geometry can change: the first frame establishes it, and a different
        # capture resolution invalidates the whole map.
        if terrain is not None:
            gh, gw = terrain["occupancy"].shape
            if (gw, gh) != (self.screen_gw, self.screen_gh):
                self._reset_arrays(gw, gh)
            self.rect, self.cell = terrain["rect"], terrain["cell"]
            self.have_terrain = True

        # --- 1) registration -------------------------------------------- #
        # World content moved by camera_delta across the screen, so the screen
        # window moved by -camera_delta over the world.
        cw, ch = self.cell
        self.origin[0] -= camera_delta[0] / max(cw, 1e-6)
        self.origin[1] -= camera_delta[1] / max(ch, 1e-6)
        self._recentre_if_needed()

        # --- 2) age unobserved gas -------------------------------------- #
        if self.config.gas_memory_decay < 1.0:
            self.gas *= self.config.gas_memory_decay

        # --- 3) fuse this frame ----------------------------------------- #
        if terrain is not None:
            self._fuse_terrain(terrain)
        if gas_grid is not None:
            self._fuse_gas(np.asarray(gas_grid, dtype=np.float32))

        self._clearance = None      # invalidate the cached distance transform

    def _recentre_if_needed(self) -> None:
        """Roll the arrays so the screen window sits back near the middle.

        Only fires when the window nears an edge, so it costs a handful of
        `np.roll`s per match rather than one per tick. Cells rolled off the far
        side are gone; cells rolled in are reset to the priors, which is
        correct -- they have genuinely never been observed.
        """
        cfg = self.config
        mx = cfg.recentre_margin * self.gw
        my = cfg.recentre_margin * self.gh
        ox, oy = self.origin
        need_x = ox < mx or (ox + self.screen_gw) > (self.gw - mx)
        need_y = oy < my or (oy + self.screen_gh) > (self.gh - my)
        if not (need_x or need_y):
            return

        tx = (self.gw - self.screen_gw) / 2.0
        ty = (self.gh - self.screen_gh) / 2.0
        sx, sy = int(round(tx - ox)), int(round(ty - oy))
        if sx == 0 and sy == 0:
            return

        rolling = [(self.occ_p, cfg.occ_prior), (self.bush, 0.0),
                   (self.gas, 0.0), (self.seen, 0.0), (self.seen_gas, 0.0)]
        if self.reference_occ is not None:
            # The reference is registered to THIS array, so it has to move with
            # it. Newly exposed edges are filled BLOCKED rather than free: they
            # are off the side of everything we know, and the localiser will
            # re-lay the reference on its next update anyway.
            rolling.append((self.reference_occ, True))
            rolling.append((self.reference_in_bounds, False))
        for arr, fill in rolling:
            arr[...] = np.roll(arr, (sy, sx), axis=(0, 1))
            if sx > 0:
                arr[:, :sx] = fill
            elif sx < 0:
                arr[:, sx:] = fill
            if sy > 0:
                arr[:sy, :] = fill
            elif sy < 0:
                arr[sy:, :] = fill
        self.origin[0] += sx
        self.origin[1] += sy

    # ------------------------------------------------------------------ #
    def _window(self, gw: int, gh: int):
        """Slice of the map covered by a (gh, gw) screen grid, and its overlap.

        Returns (map_slice, obs_slice) or None when the window has drifted
        entirely off the map (which the recentre above should prevent).
        """
        ox, oy = int(round(self.origin[0])), int(round(self.origin[1]))
        mx0, my0 = max(0, ox), max(0, oy)
        mx1, my1 = min(self.gw, ox + gw), min(self.gh, oy + gh)
        if mx1 <= mx0 or my1 <= my0:
            return None
        ox0, oy0 = mx0 - ox, my0 - oy
        return ((slice(my0, my1), slice(mx0, mx1)),
                (slice(oy0, oy0 + (my1 - my0)), slice(ox0, ox0 + (mx1 - mx0))))

    def _fuse_terrain(self, terrain: dict) -> None:
        cfg = self.config
        occ = terrain["occupancy"].astype(np.float32)
        bush = np.asarray(terrain["bush"], dtype=np.float32)
        # `unknown` = hidden behind the HUD. These cells carry NO information,
        # so they must not be fused at all -- neither as free nor as blocked.
        # Skipping them is what lets the map fill them in later, from a frame
        # where the camera has scrolled that ground out from under the button.
        visible = ~np.asarray(terrain["unknown"], dtype=bool)

        win = self._window(occ.shape[1], occ.shape[0])
        if win is None:
            return
        msl, osl = win
        vis = visible[osl]
        if not vis.any():
            return

        a = cfg.occ_alpha
        tgt = self.occ_p[msl]
        tgt[vis] += a * (occ[osl][vis] - tgt[vis])
        tb = self.bush[msl]
        tb[vis] += a * (bush[osl][vis] - tb[vis])
        ts = self.seen[msl]
        ts[vis] += 0.5 * (1.0 - ts[vis])

    def _fuse_gas(self, gas: np.ndarray) -> None:
        """Fuse a gas reading — but only where there was ground to read it off.

        THE POCKET BUG. Gas is an overlay the engine draws on the FLOOR. A wall
        block has no floor showing, so the detector reads gas=0 over it no
        matter how deep inside the cloud it sits. The old code fused that zero
        and stamped the cell `seen_gas`, which reads as "we looked, and it is
        clear" — the strongest possible claim, from the one place we cannot
        make it. A cluster of blocks inside the cloud therefore became an
        island of apparent safety, and since it is enclosed by gas it also
        looks like the *nearest* safe ground, so the planner steered into it.

        The reading is still fused (a wall cell's own cost is irrelevant — A*
        will not expand it anyway), but it is NOT credited as observed. That
        leaves it `gas_unobserved`, which is exactly the state the planner's
        `gas_unknown_dilate_cells` extrapolation already exists to handle: the
        cloud gets extended across it from the ground either side.

        Deliberately keyed on occupancy rather than on terrain-explored: on a
        skin with no colour profile nothing is occupied, so this correctly does
        nothing rather than declaring the whole map gas-unobserved.
        """
        cfg = self.config
        win = self._window(gas.shape[1], gas.shape[0])
        if win is None:
            return
        msl, osl = win
        cur = self.gas[msl]
        obs = gas[osl]
        # Asymmetric: believe an increase quickly, a decrease slowly.
        rate = np.where(obs > cur, cfg.gas_rise, cfg.gas_fall).astype(np.float32)
        cur += rate * (obs - cur)

        readable = self.occ_p[msl] <= cfg.occ_threshold
        ts = self.seen_gas[msl]
        ts[readable] += 0.5 * (1.0 - ts[readable])

    # ------------------------------------------------------------------ #
    def apply_reference(self, pose) -> None:
        """Adopt a ground-truth layout from `perception.localize`.

        Once the arena is identified, everything the agent has NOT looked at
        stops being a guess. Two things change, and the second is the one that
        matters:

          * Unobserved cells fall back to the published layout instead of the
            optimistic `occ_prior`. Slightly conservative — the layout counts
            bushes as blocked (see `perception/mapdb.py`) — but it is real
            information about ground the agent has never seen, which nothing
            else in the pipeline can supply.
          * Everything outside the arena becomes BLOCKED, exactly. The planner
            currently infers "near the edge" from how much unwalkable
            decoration surrounds a cell and spends a whole `border_cost` term
            compensating for the fact that this is only a proxy. With a real
            boundary it is not a proxy any more.

        Observed cells keep their FUSED value. The live grid is the authority
        on what is walkable right now — it knows bushes are passable and it
        sees boxes that the static layout cannot.
        """
        gm = pose.game_map
        ref = gm.occupancy
        rh, rw = ref.shape
        ys, xs = np.mgrid[0:self.gh, 0:self.gw]
        mx = np.floor(pose.origin[0] + xs / pose.scale).astype(np.int32)
        my = np.floor(pose.origin[1] + ys / pose.scale).astype(np.int32)
        inside = (mx >= 0) & (mx < rw) & (my >= 0) & (my < rh)

        # Off the edge of the arena is blocked, not unknown.
        out = np.ones((self.gh, self.gw), bool)
        out[inside] = ref[my[inside], mx[inside]]
        self.reference_occ = out
        self.reference_in_bounds = inside & np.where(
            inside, gm.in_bounds[np.clip(my, 0, rh - 1), np.clip(mx, 0, rw - 1)], False)
        self.have_reference = True

    def clear_reference(self) -> None:
        self.reference_occ = None
        self.reference_in_bounds = None
        self.have_reference = False

    @property
    def occupancy(self) -> np.ndarray:
        """Boolean blocked mask over the whole map.

        Observed cells come from fused observation; unobserved cells come from
        the reference layout when one is attached, and from the prior otherwise.
        """
        occ = self.occ_p > self.config.occ_threshold
        if self.have_reference and self.reference_occ is not None:
            unseen = self.seen < 0.15
            occ = np.where(unseen, self.reference_occ, occ)
        return occ

    @property
    def out_of_bounds(self) -> np.ndarray:
        """Cells outside the arena. All False without a reference map."""
        if self.have_reference and self.reference_in_bounds is not None:
            return ~self.reference_in_bounds
        return np.zeros((self.gh, self.gw), bool)

    @property
    def unexplored(self) -> np.ndarray:
        """Ground whose TERRAIN has never been classified."""
        return self.seen < 0.15

    @property
    def gas_unobserved(self) -> np.ndarray:
        """Ground the gas detector has never been pointed at.

        Distinct from `unexplored`: gas needs no map profile, so on an
        uncalibrated skin every cell is terrain-unexplored while most are
        gas-observed.
        """
        return self.seen_gas < 0.15

    def clearance(self) -> np.ndarray:
        """Distance in cells from each free cell to the nearest obstacle.

        Cached per update, because the planner asks for it more than once per
        tick (cost grid, destination relocation, gas escape).
        """
        if self._clearance is not None and self._clearance_stamp == self._stamp:
            return self._clearance
        free = (~self.occupancy).astype(np.uint8)
        # Border cells are treated as walls so the map edge does not read as
        # wide-open ground the agent should happily route along.
        free[0, :] = free[-1, :] = free[:, 0] = free[:, -1] = 0
        dist = cv2.distanceTransform(free, cv2.DIST_L2, 3)
        self._clearance = dist.astype(np.float32)
        self._clearance_stamp = self._stamp
        return self._clearance

    # --- coordinate conversion ----------------------------------------- #
    def ready(self) -> bool:
        """Enough geometry to convert pixels to cells and plan on the map."""
        return self.rect is not None

    def has_walls(self) -> bool:
        """True once at least one terrain observation has been fused.

        Without this the map is still useful -- it carries gas and geometry --
        but every cell reads as passable, so the planner should not pretend the
        empty route it finds means anything about walls.
        """
        return self.have_terrain

    def to_cell(self, point) -> Tuple[float, float]:
        """Frame pixel -> fractional map cell."""
        rx, ry = (self.rect[0], self.rect[1]) if self.rect else (0.0, 0.0)
        cw, ch = self.cell
        return (self.origin[0] + (point[0] - rx) / max(cw, 1e-6),
                self.origin[1] + (point[1] - ry) / max(ch, 1e-6))

    def to_pixels(self, cell_xy) -> Tuple[float, float]:
        """Map cell (centre) -> frame pixel."""
        rx, ry = (self.rect[0], self.rect[1]) if self.rect else (0.0, 0.0)
        cw, ch = self.cell
        return (rx + (cell_xy[0] + 0.5 - self.origin[0]) * cw,
                ry + (cell_xy[1] + 0.5 - self.origin[1]) * ch)

    def clamp_cell(self, cell_xy) -> Tuple[int, int]:
        return (int(np.clip(cell_xy[0], 0, self.gw - 1)),
                int(np.clip(cell_xy[1], 0, self.gh - 1)))

    def in_bounds(self, cell_xy) -> bool:
        return 0 <= cell_xy[0] < self.gw and 0 <= cell_xy[1] < self.gh

    def cell_size_px(self) -> float:
        """Nominal cell size in pixels, for converting radii and distances."""
        return float(max(1e-6, min(self.cell)))
