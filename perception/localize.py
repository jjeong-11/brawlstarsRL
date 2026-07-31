"""
perception/localize.py
======================

Works out WHICH Showdown map is in play and WHERE on it the camera is, by
matching the live occupancy grid against the reference layouts in
`perception/mapdb.py`.

WHAT THIS BUYS
--------------
Absolute position is the difference between a bot that can see one screen and a
bot that knows the arena. With it:

  * A* plans across the whole map instead of the visible window, so a
    destination behind the camera is reachable rather than clamped to the
    screen edge.
  * The map BORDER is known exactly, instead of being inferred from how much
    unwalkable decoration surrounds a cell. That proxy is why the planner
    carries a `border_cost` term at all.
  * The gas can be modelled properly. Showdown's cloud closes toward the arena
    centre, and "the arena centre" is meaningless without absolute coordinates
    — which is why the old escape heuristic had to guess from a centroid, and
    why it guessed backwards.
  * Terrain no longer depends on a hand-calibrated HSV profile surviving a skin
    rotation.

HOW IT WORKS
------------
Matching is done on OCCUPANCY, not pixels. The renders and the live camera view
share no palette, no lighting and no projection, so correlating images directly
is hopeless — but "where are the walls" is the same question in both, and
`getTerrain` already answers it for the live frame.

So: resample the observed occupancy to the reference map's cell scale and slide
it over each candidate map with `cv2.matchTemplate`. The best
normalised-correlation peak over all maps and positions is the pose.

ONE VIEWPORT IS NOT ENOUGH, AND THAT IS THE WHOLE DESIGN
--------------------------------------------------------
The obvious implementation matches the current frame's 48x27 grid. It does not
work, and it fails in the worst possible way — confidently.

Measured against all 71 reference maps, with 8% of cells flipped to simulate
segmentation noise, identifying a square patch of N x N MAP cells:

    patch     correct     mean margin over the runner-up
    8x8       19/30       +0.073
    12x12     30/30       +0.252
    16x16     30/30       +0.372
    24x24     30/30       +0.524

And on a real screenshot, matching a single viewport scored 0.60-0.71 against
EVERY map in the database, with 0.017 between first and second place. That is
not recognition, it is a small template finding a lucky spot in 71 large
haystacks — the same overfitting any small-template match suffers.

One live viewport covers roughly 8x8 map cells. That is precisely the row where
identification is a coin flip.

So this does not match a frame. It matches the ACCUMULATED occupancy from
`rl/world_map.WorldMap`, which fuses frames across time and already spans three
screens. `min_patch_cells` refuses to even attempt identification until the
explored region clears the 12x12 threshold the table above establishes —
because a wrong lock is unrecoverable, while waiting a few seconds costs
nothing (the planner runs on the local map meanwhile, exactly as it did before
this module existed).

THE SCALE PROBLEM
-----------------
The live grid's cell size in MAP cells is not known a priori: it depends on the
camera's zoom, which is fixed per device but not something we are told. So the
search runs over a range of scales, and once a map has locked, the winning
scale is reused — it cannot change mid-match.

WHY IT IS STICKY, AND WHY IT STILL RE-CHECKS
--------------------------------------------
A match cannot change maps mid-game, so re-scoring 71 candidates every frame
would be pure waste — and worse, it would let a single bad frame (a super
effect, a death overlay) throw the pose away. So identification happens once,
against accumulated evidence rather than one frame, and after that only the
POSITION is tracked, seeded from the previous pose plus the camera delta.

But identification is not permanent either. Confidence is monitored, and a
sustained collapse re-opens the search — otherwise a wrong early lock (the
first frames of a match can be a loading screen) would poison the whole match.

DEGRADATION
-----------
Everything here is optional. `Localizer.update` returns None when it cannot
lock, and the planner falls back to the local world map exactly as before. A
bot that is merely as good as it was yesterday is the correct failure mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .mapdb import GameMap, MapDatabase, database


@dataclass(frozen=True)
class LocalizerConfig:
    # Live-grid cells per reference-map cell. Depends on the camera zoom, which
    # is a device constant we are not told, so it is searched once and then
    # frozen — it cannot change mid-match.
    scales: Tuple[float, ...] = (2.2, 2.6, 3.0, 3.4, 3.8, 4.4, 5.0)

    # Smallest explored region, in MAP cells per side, that may be used for
    # identification. 12 is measured, not guessed: see the table in the module
    # docstring. Below it, identification is a coin flip against 71 maps.
    min_patch_cells: int = 12

    # matchTemplate correlation below which a pose is not believed. Occupancy
    # grids are noisy (bushes disagree by construction, see mapdb), so this is
    # deliberately not near 1.0.
    min_confidence: float = 0.42
    # A first lock has to clear a higher bar than a tracked update, because
    # getting the map wrong is unrecoverable in a way that a jittery position
    # is not.
    min_lock_confidence: float = 0.55
    # ... and beat the runner-up by this much. Also measured: at a 12x12 patch
    # the true map leads by +0.25 on average, while a too-small patch leads by
    # +0.07. Requiring 0.15 rejects the second case without rejecting the first.
    min_margin: float = 0.15

    # Attempts (spaced by `attempt_every`) that must agree before locking.
    vote_frames: int = 3
    # Identification is expensive — 71 maps x 7 scales — and the explored region
    # only grows slowly, so there is nothing to gain from retrying every tick.
    attempt_every: int = 20
    # Consecutive low-confidence tracked frames before the lock is abandoned.
    relock_after: int = 45

    # Half-width of the local search window, in map cells, once tracking.
    track_radius: int = 6


@dataclass
class Pose:
    """Where the live viewport sits on the reference map."""
    game_map: GameMap
    # Map-cell coordinate of the live grid's top-left corner. Fractional: the
    # camera does not move in whole cells.
    origin: Tuple[float, float]
    scale: float          # live cells per map cell
    confidence: float
    tracked: bool = False   # False = found by full search, True = incremental

    def to_map_cell(self, live_cell) -> Tuple[float, float]:
        return (self.origin[0] + live_cell[0] / self.scale,
                self.origin[1] + live_cell[1] / self.scale)

    def to_live_cell(self, map_cell) -> Tuple[float, float]:
        return ((map_cell[0] - self.origin[0]) * self.scale,
                (map_cell[1] - self.origin[1]) * self.scale)


def _resample(grid: np.ndarray, scale: float) -> np.ndarray:
    """Live occupancy -> reference-map cell scale, as float 0..1."""
    gh, gw = grid.shape
    tw, th = max(2, int(round(gw / scale))), max(2, int(round(gh / scale)))
    return cv2.resize(grid.astype(np.float32), (tw, th),
                      interpolation=cv2.INTER_AREA)


def _match(patch: np.ndarray, reference: np.ndarray):
    """Best (score, x, y) of `patch` inside `reference` by normalised correlation.

    TM_CCOEFF_NORMED subtracts the mean of each window before correlating, which
    matters here: a patch that is 40% blocked and a map region that is 40%
    blocked would score highly on raw overlap no matter how the walls were
    arranged. Removing the mean makes the score about the PATTERN.
    """
    rh, rw = reference.shape
    ph, pw = patch.shape
    if ph > rh or pw > rw or ph < 2 or pw < 2:
        return (-1.0, 0, 0)
    res = cv2.matchTemplate(reference, patch, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    return (float(score), int(loc[0]), int(loc[1]))


class Localizer:
    """Sticky map identification + incremental pose tracking."""

    def __init__(self, config: Optional[LocalizerConfig] = None,
                 db: Optional[MapDatabase] = None):
        self.config = config or LocalizerConfig()
        self.db = db if db is not None else database()
        self.reset()

    def reset(self) -> None:
        self.pose: Optional[Pose] = None
        self._votes = {}
        self._attempts = 0
        self._calls = 0
        self._low_confidence_for = 0
        self.explored_cells: float = 0.0     # side length, in map cells

    # ------------------------------------------------------------------ #
    @staticmethod
    def _explored_patch(occupancy, seen):
        """Crop the accumulated map to what has actually been observed.

        Unobserved cells inside the crop are filled with the MEAN of the
        observed ones. That is the neutral value for `TM_CCOEFF_NORMED`, which
        subtracts each window's mean before correlating — so unexplored ground
        contributes nothing instead of voting for "open".
        """
        occ = np.asarray(occupancy, dtype=np.float32)
        seen = np.asarray(seen, dtype=bool)
        if not seen.any():
            return None, (0, 0)
        ys, xs = np.nonzero(seen)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        patch = occ[y0:y1, x0:x1].copy()
        mask = seen[y0:y1, x0:x1]
        if not mask.all():
            patch[~mask] = float(patch[mask].mean()) if mask.any() else 0.5
        return patch, (x0, y0)

    def update(self, occupancy: np.ndarray, seen: np.ndarray,
               camera_delta_cells=(0.0, 0.0)) -> Optional[Pose]:
        """Accumulated occupancy + observation mask -> a pose, or None.

        Both arrays are in LIVE grid cells and must share a coordinate frame —
        `rl.world_map.WorldMap.occupancy` and `world.seen > 0.15` are exactly
        that. `camera_delta_cells` seeds the tracking search so the window can
        stay small.

        Poses are expressed against the caller's array, so `pose.origin` is the
        map cell that array's (0, 0) sits on.
        """
        if occupancy is None or not len(self.db):
            return self.pose
        self._calls += 1
        patch, offset = self._explored_patch(occupancy, seen)
        if patch is None:
            return self.pose

        if self.pose is None:
            # Identification is 71 maps x 7 scales; the explored region grows
            # slowly, so retrying every tick buys nothing.
            if self._calls % max(1, self.config.attempt_every):
                return None
            return self._identify(patch, offset)
        return self._track(patch, offset, camera_delta_cells)

    # ------------------------------------------------------------------ #
    def _identify(self, patch: np.ndarray, offset) -> Optional[Pose]:
        """Full search over every map and scale, voting across attempts."""
        cfg = self.config
        best_overall = None

        for scale in cfg.scales:
            resampled = _resample(patch, scale)
            side = min(resampled.shape)
            self.explored_cells = max(self.explored_cells, float(side))
            # REFUSE rather than guess. Below the measured threshold the winner
            # is essentially random across 71 maps, and a wrong lock is
            # unrecoverable — see the table in the module docstring.
            if side < cfg.min_patch_cells:
                continue
            scored = []
            for gm in self.db:
                score, x, y = _match(resampled, gm.occupancy.astype(np.float32))
                scored.append((score, gm, x, y))
            scored.sort(key=lambda t: -t[0])
            top, runner_up = scored[0], (scored[1][0] if len(scored) > 1 else -1.0)
            margin = top[0] - runner_up
            if best_overall is None or top[0] > best_overall[0][0]:
                best_overall = (top, margin, scale)

        if best_overall is None:
            return None            # still too little explored to try

        (score, gm, x, y), margin, scale = best_overall
        self._attempts += 1
        if score < cfg.min_lock_confidence or margin < cfg.min_margin:
            return None

        prev = self._votes.get(gm.name, (0, 0.0))
        self._votes[gm.name] = (prev[0] + 1, prev[1] + score)
        count, total = self._votes[gm.name]
        if count < cfg.vote_frames:
            return None

        # `origin` is relative to the caller's full array, not the crop.
        self.pose = Pose(game_map=gm,
                         origin=(float(x) - offset[0] / scale,
                                 float(y) - offset[1] / scale),
                         scale=scale, confidence=total / count, tracked=False)
        return self.pose

    # ------------------------------------------------------------------ #
    def _track(self, patch_in: np.ndarray, offset, camera_delta_cells) -> Optional[Pose]:
        """Local search around the predicted pose. Cheap: one map, one scale."""
        cfg = self.config
        pose = self.pose
        gm = pose.game_map
        scale = pose.scale

        # The viewport moved opposite to the world's scroll, in map cells.
        pred = (pose.origin[0] - camera_delta_cells[0] / scale + offset[0] / scale,
                pose.origin[1] - camera_delta_cells[1] / scale + offset[1] / scale)

        patch = _resample(patch_in, scale)
        ph, pw = patch.shape
        ref = gm.occupancy.astype(np.float32)
        rh, rw = ref.shape

        r = cfg.track_radius
        x0 = int(np.clip(round(pred[0]) - r, 0, max(0, rw - pw)))
        y0 = int(np.clip(round(pred[1]) - r, 0, max(0, rh - ph)))
        x1 = int(np.clip(round(pred[0]) + r + pw, pw, rw))
        y1 = int(np.clip(round(pred[1]) + r + ph, ph, rh))
        window = ref[y0:y1, x0:x1]

        score, dx, dy = _match(patch, window)
        if score < cfg.min_confidence:
            self._low_confidence_for += 1
            if self._low_confidence_for >= cfg.relock_after:
                # A sustained collapse means the lock was wrong, or the match
                # ended. Re-open the search rather than tracking a fiction.
                self.reset()
                return None
            # Coast on dead reckoning: the camera delta is still trustworthy
            # even when this frame's segmentation is not.
            self.pose = Pose(gm, (pred[0] - offset[0] / scale,
                                  pred[1] - offset[1] / scale),
                             scale, score, tracked=True)
            return self.pose

        self._low_confidence_for = 0
        self.pose = Pose(gm, (float(x0 + dx) - offset[0] / scale,
                              float(y0 + dy) - offset[1] / scale),
                         scale, score, tracked=True)
        return self.pose

    # ------------------------------------------------------------------ #
    def status(self) -> str:
        if self.pose is None:
            if self.explored_cells < self.config.min_patch_cells:
                return (f"exploring ({self.explored_cells:.0f}/"
                        f"{self.config.min_patch_cells} map cells seen -- too "
                        f"little to identify a map reliably)")
            top = sorted(self._votes.items(), key=lambda kv: -kv[1][0])[:1]
            hint = f", leading {top[0][0]}" if top else ""
            return f"searching ({self._attempts} attempts{hint})"
        p = self.pose
        return (f"{p.game_map.name} @ ({p.origin[0]:.1f}, {p.origin[1]:.1f}) "
                f"scale {p.scale:.1f} conf {p.confidence:.2f}"
                f"{' tracked' if p.tracked else ' LOCKED'}")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true",
                    help="crop windows out of the reference maps themselves and "
                         "check the localiser puts them back")
    ap.add_argument("image", nargs="?", help="a real screenshot to localise")
    args = ap.parse_args()

    db = database()
    print(f"{len(db)} maps in the database")

    if args.selftest:
        # Machinery test: validates search, scale, the explored-area gate and
        # tracking, NOT the render-to-live-frame correspondence (nothing
        # offline can validate that -- it needs a screenshot whose map is
        # known and this project has none).
        cfg = LocalizerConfig()
        rng = np.random.default_rng(0)
        scale = 3.0
        ok = refused = wrong = trials = 0

        for gm in list(db)[:16]:
            ref = gm.occupancy
            rh, rw = ref.shape
            # An explored region a bit past the 12-cell threshold, i.e. what a
            # WorldMap holds after a few seconds of walking.
            side = cfg.min_patch_cells + 4
            if rh <= side or rw <= side:
                continue
            for _ in range(3):
                trials += 1
                ty = int(rng.integers(0, rh - side))
                tx = int(rng.integers(0, rw - side))
                crop = ref[ty:ty + side, tx:tx + side].astype(np.float32)
                live = cv2.resize(crop, (int(side * scale), int(side * scale)),
                                  interpolation=cv2.INTER_NEAREST)
                noise = rng.random(live.shape) < 0.08   # segmentation error
                live = (np.where(noise, 1.0 - live, live) > 0.5)
                seen = np.ones_like(live, bool)

                loc = Localizer()
                pose = None
                for _ in range(cfg.attempt_every * cfg.vote_frames + 1):
                    pose = loc.update(live, seen)
                    if pose is not None:
                        break
                if pose is None:
                    refused += 1
                    continue
                err = np.hypot(pose.origin[0] - tx, pose.origin[1] - ty)
                if pose.game_map.name == gm.name and err <= 2.0:
                    ok += 1
                else:
                    wrong += 1

        print(f"\nself-test ({trials} trials, {cfg.min_patch_cells + 4}-cell "
              f"explored region, 8% cell noise):")
        print(f"  located correctly : {ok}")
        print(f"  refused to guess  : {refused}")
        print(f"  WRONG LOCK        : {wrong}   <- the one that must be 0")

        # Below the measured threshold it must REFUSE, not guess. A wrong lock
        # is unrecoverable; refusing just means the planner keeps using the
        # local map, which is what it did before this module existed.
        gm = list(db)[0]
        tiny = gm.occupancy[:6, :6].astype(np.float32)
        live = cv2.resize(tiny, (18, 18), interpolation=cv2.INTER_NEAREST) > 0.5
        loc = Localizer()
        for _ in range(200):
            loc.update(live, np.ones_like(live, bool))
        gated = loc.pose is None
        print(f"  6x6 patch refused : {gated}   ({loc.status()})")

        raise SystemExit(0 if (wrong == 0 and ok >= trials * 0.8 and gated) else 1)

    if args.image:
        from .getTerrain import find_terrain
        from .getAnchor import find_player_position

        img = cv2.imread(args.image)
        if img is None:
            raise SystemExit(f"could not read {args.image}")
        a = find_player_position(img)
        terrain = find_terrain(img, player_pos=(int(a[0]), int(a[1])) if a else None)
        if terrain is None:
            raise SystemExit("no terrain profile matched — nothing to localise with")
        loc = Localizer()
        pose = None
        for _ in range(LocalizerConfig().vote_frames):
            pose = loc.update(terrain["occupancy"])
        print(f"\n{loc.status()}")
        if pose:
            print(f"  arena {pose.game_map.occupancy.shape[1]}x"
                  f"{pose.game_map.occupancy.shape[0]} cells, "
                  f"{100 * pose.game_map.occupancy.mean():.0f}% blocked")
