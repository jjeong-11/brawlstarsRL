"""
rl/debug_trace.py
=================

Shows what the agent decided, and — the part that was missing — WHERE IT
INTENDED TO WALK.

WHY
---
Everything upstream of the joystick was already observable in isolation:
`scripts/watch_live.py` prints perception readings and the reward breakdown,
`getTerrain --calibrate` dumps an occupancy overlay, `--profile` reports timings.
What none of them showed was the decision itself: the destination the policy
latched onto, the route A* chose to get there, and how the joystick vector
followed from that route. When the agent walked into a wall or into the gas,
there was no way to tell whether perception was wrong, the destination was
wrong, or the route was wrong — three completely different bugs with completely
different fixes.

This module renders one annotated frame per capture containing all three
layers, so a bad decision can be attributed by eye:

    PERCEPTION   player, enemies, boxes, cubes, and the fused occupancy /
                 gas grids the planner actually used (not the raw frame).
    DECISION     the latched waypoint, the A* path to it, and the commitment
                 clock. A path that ends somewhere silly means the DESTINATION
                 was bad; a sensible destination with a route hugging a wall
                 means the COST FIELD is bad.
    ACTUATION    the joystick vector finally sent to the device, so a route
                 that looks right but a stick that points elsewhere localises
                 the bug to steering rather than planning.

The inset in the corner is the world map (`rl/world_map.py`), which is three
screens wide. It matters because the most confusing failures involve ground
that is currently OFF SCREEN — a waypoint behind the camera, or a gas cell the
agent is remembering rather than seeing. Those are invisible in the main view
by definition.

USE
---
    python scripts/train_rl.py --live --serial <S> --trace 2      # every 2s
    python scripts/watch_live.py --serial <S> --trace 2

Frames land in `debugOutput/trace/` alongside `trace.jsonl`, one JSON record per
captured decision for grepping ("show me every step where mode was escape").

Capture is INTERVAL-BASED rather than every-step on purpose: an annotated PNG is
a few hundred KB and encoding one costs ~15ms, which at a 0.1s tick would be
15% of the loop and tens of GB per hour. Every couple of seconds is enough to
see behaviour and costs nothing measurable.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

# BGR, because OpenCV.
_C_PLAYER = (255, 128, 0)
_C_ENEMY = (0, 0, 255)
_C_BOX = (0, 165, 255)
_C_CUBE = (0, 255, 0)
_C_PATH = (0, 255, 255)
_C_WAYPOINT = (255, 0, 255)
_C_STICK = (255, 255, 255)
_C_WALL = (60, 60, 200)
_C_GAS = (120, 255, 180)
_C_TEXT = (255, 255, 255)
_C_PANEL = (24, 24, 24)

_FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class TraceConfig:
    every_seconds: float = 2.0
    out_dir: str = "debugOutput/trace"
    # Ring buffer. A long training run would otherwise fill the disk; keeping
    # the most recent N is what you actually want when something goes wrong,
    # because you go and look immediately after seeing it.
    max_frames: int = 600
    write_jsonl: bool = True
    # Long side of the written PNG. Full 2424px frames are ~1.5MB each and add
    # nothing — every annotation is legible at 1280.
    max_width: int = 1280
    draw_grid: bool = True
    draw_minimap: bool = True
    jpeg_quality: int = 82

    # --- storage guards ------------------------------------------------- #
    # max_frames caps the frame COUNT, which is not the same as capping bytes:
    # a frame is 150-400KB depending on scene complexity, the JSONL appends
    # forever, and a directory left over from a run with different settings is
    # not cleaned by the ring buffer at all (it reuses indices 0..max_frames-1
    # and simply never touches trace_0600.jpg from the run before).
    #
    # An overnight run is exactly when nobody is watching the disk fill.
    max_bytes: int = 512 * 1024 * 1024      # budget for the whole trace dir
    max_jsonl_bytes: int = 64 * 1024 * 1024  # rotated to .1 past this
    # Below this much free space, stop writing images entirely rather than take
    # the machine down with it. Training continues; only the debug output stops.
    min_free_bytes: int = 2 * 1024 * 1024 * 1024
    prune_every: int = 25                   # captures between budget checks


class DecisionTracer:
    """Writes an annotated frame + a JSON record every `every_seconds`."""

    def __init__(self, config: Optional[TraceConfig] = None, root=None):
        self.config = config or TraceConfig()
        root = pathlib.Path(root) if root else pathlib.Path(__file__).resolve().parent.parent
        self.dir = root / self.config.out_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self._last_at = 0.0
        self._n = 0
        self._jsonl = None
        self._paused_for_disk = False
        self._since_prune = 0
        # Sweep once at startup: this is where a previous run's leftovers get
        # collected, and the only place they ever would.
        self._prune()
        if self.config.write_jsonl:
            self._jsonl = open(self.dir / "trace.jsonl", "a", buffering=1)

    def close(self) -> None:
        if self._jsonl is not None:
            self._jsonl.close()
            self._jsonl = None

    # --- storage ---------------------------------------------------------- #
    def _frames(self):
        """Trace images, oldest first."""
        try:
            files = [p for p in self.dir.glob("trace_*.jpg") if p.is_file()]
        except OSError:
            return []
        return sorted(files, key=lambda p: p.stat().st_mtime)

    def _free_bytes(self) -> Optional[int]:
        try:
            return shutil.disk_usage(self.dir).free
        except OSError:
            return None

    def _prune(self) -> None:
        """Keep the trace directory inside its byte budget, oldest first.

        Deliberately deletes by MTIME rather than by index. The ring buffer
        reuses names trace_0000..trace_0599, so on disk the newest frame can
        have the lowest number -- sorting by filename would delete the frames
        you just captured and keep the stale ones, which is precisely backwards
        for a debugging aid you consult right after seeing something go wrong.
        """
        cfg = self.config
        files = self._frames()
        total = 0
        for p in files:
            try:
                total += p.stat().st_size
            except OSError:
                pass

        free = self._free_bytes()
        tight = free is not None and free < cfg.min_free_bytes
        # When the disk is tight, claw back to half the budget rather than
        # sitting at the limit and re-pruning on every single capture.
        budget = (cfg.max_bytes // 2) if tight else cfg.max_bytes

        for p in files:
            if total <= budget:
                break
            try:
                size = p.stat().st_size
                p.unlink()
                total -= size
            except OSError:
                pass

        # The JSONL is append-only and outlives any single run, so it needs its
        # own cap. One generation is kept: enough to span a rotation boundary,
        # bounded unlike the alternative.
        path = self.dir / "trace.jsonl"
        try:
            if path.exists() and path.stat().st_size > cfg.max_jsonl_bytes:
                if self._jsonl is not None:
                    self._jsonl.close()
                    self._jsonl = None
                path.replace(self.dir / "trace.jsonl.1")
                if cfg.write_jsonl:
                    self._jsonl = open(path, "a", buffering=1)
        except OSError:
            pass

        # Final guard: if the disk is STILL below the floor after pruning, the
        # problem is not us. Stop writing images and say so once -- a stalled
        # training run at 3am is a worse outcome than a gap in the traces.
        free = self._free_bytes()
        low = free is not None and free < cfg.min_free_bytes
        if low and not self._paused_for_disk:
            print(f"[trace] only {free / 1e9:.1f}GB free — pausing trace images "
                  f"(records still written to trace.jsonl)")
        elif self._paused_for_disk and not low:
            print("[trace] disk space recovered — resuming trace images")
        self._paused_for_disk = low

    def due(self) -> bool:
        return (time.time() - self._last_at) >= self.config.every_seconds

    # ------------------------------------------------------------------ #
    def capture(self, frame, planner, state, intent=None, reward=None,
                live=None, tick: int = 0, force: bool = False):
        """Render and write one trace frame. Returns the path, or None if skipped."""
        if frame is None or (not force and not self.due()):
            return None
        self._last_at = time.time()

        record = _record(planner, state, intent, reward, tick)
        try:
            out = self.render(frame, planner, state, intent, reward, live, tick)
        except Exception as e:                      # never take the loop down
            record["render_error"] = repr(e)
            out = None

        name = f"trace_{self._n % self.config.max_frames:04d}.jpg"
        path = self.dir / name
        if out is not None and not self._paused_for_disk:
            cv2.imwrite(str(path), out,
                        [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality])
            record["frame"] = name
        else:
            # The record is still worth having without the picture -- it is what
            # the post-hoc analysis actually greps.
            record["frame"] = None
            if self._paused_for_disk:
                record["frame_skipped"] = "low disk"
        if self._jsonl is not None:
            self._jsonl.write(json.dumps(record, default=str) + "\n")
        self._n += 1

        self._since_prune += 1
        if self._since_prune >= self.config.prune_every:
            self._since_prune = 0
            self._prune()
        return path

    # ------------------------------------------------------------------ #
    def render(self, frame, planner, state, intent=None, reward=None,
               live=None, tick: int = 0) -> np.ndarray:
        """The annotated frame. Pure-ish: safe to call outside the loop."""
        cfg = self.config
        out = frame.copy()
        H, W = out.shape[:2]
        world = getattr(planner, "world", None)
        status = planner.status() if planner is not None else None

        if cfg.draw_grid and world is not None and world.ready():
            _draw_grids(out, world)
        _draw_entities(out, state, live)
        _draw_plan(out, planner, state)

        scale = min(1.0, cfg.max_width / float(W))
        if scale < 1.0:
            out = cv2.resize(out, (int(W * scale), int(H * scale)),
                             interpolation=cv2.INTER_AREA)

        if cfg.draw_minimap and world is not None and world.ready():
            _draw_minimap(out, world, planner, state)
        _draw_panel(out, status, state, intent, reward, tick)
        return out


# --- layers ----------------------------------------------------------------- #
def _draw_grids(out, world) -> None:
    """Tint the FUSED occupancy and gas the planner is actually using.

    Deliberately the fused map rather than this frame's raw segmentation: when
    the two disagree it is the fused one that determines the route, and that
    disagreement is exactly the bug you are looking for.

    Blocked cells get an OUTLINE as well as a tint. A tint alone is close to
    unreadable here — Showdown's walls are already pink and its floors already
    purple, so a translucent red wash over them changes very little and you
    cannot tell at a glance whether a cell the agent walked into was believed
    solid. A hard contour along the blocked/free boundary is unambiguous.

    Also gets the cells shaded UNREACHABLE by the agent-footprint inflation
    (`hard_clearance`), because "A* refused to go through that gap" and "A*
    thought that gap was a wall" look identical without it.
    """
    rect = world.rect
    x, y, w, h = rect
    gw, gh = world.screen_gw, world.screen_gh
    ox, oy = int(round(world.origin[0])), int(round(world.origin[1]))

    def window(arr):
        """The on-screen part of a map-sized array, padded if it hangs off."""
        out_a = np.zeros((gh, gw), arr.dtype)
        mx0, my0 = max(0, ox), max(0, oy)
        mx1, my1 = min(world.gw, ox + gw), min(world.gh, oy + gh)
        if mx1 <= mx0 or my1 <= my0:
            return out_a
        out_a[my0 - oy:my1 - oy, mx0 - ox:mx1 - ox] = arr[my0:my1, mx0:mx1]
        return out_a

    occ = window(world.occupancy)
    gas = window(world.gas)
    # Cells that are free but too tight for the agent to fit through.
    tight = window((world.clearance() < 0.9) & (~world.occupancy))

    # Build the overlay on the small grid and upscale once, rather than issuing
    # ~1300 cv2.rectangle calls per frame.
    small = np.zeros((gh, gw, 3), np.float32)
    small[occ] = _C_WALL
    small[tight] = (90, 90, 90)
    gmask = gas > 0.08
    if gmask.any():
        small[gmask] = np.array(_C_GAS, np.float32) * \
            np.clip(gas[gmask] * 1.8, 0.35, 1.0)[:, None]

    alpha = np.zeros((gh, gw), np.float32)
    alpha[tight] = 0.30
    alpha[gmask] = 0.55
    alpha[occ] = 0.45

    big = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    big_a = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_NEAREST)[..., None]
    roi = out[y:y + h, x:x + w].astype(np.float32)
    out[y:y + h, x:x + w] = (roi * (1 - big_a) + big * big_a).astype(np.uint8)

    # Hard outline along the wall boundary — the part you can actually read.
    edges = cv2.resize(occ.astype(np.uint8) * 255, (w, h),
                       interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out[y:y + h, x:x + w], contours, -1, (80, 80, 255), 2)


def _draw_entities(out, state, live) -> None:
    if state is None:
        return
    if state.player_pos:
        cv2.circle(out, tuple(int(v) for v in state.player_pos), 34, _C_PLAYER, 3)
    for p in getattr(state, "enemy_positions", ()) or ():
        cv2.circle(out, (int(p[0]), int(p[1])), 36, _C_ENEMY, 3)
        cv2.circle(out, (int(p[0]), int(p[1])), 4, _C_ENEMY, -1)
    for p in getattr(state, "box_positions", ()) or ():
        cv2.circle(out, (int(p[0]), int(p[1])), 26, _C_BOX, 2)
    for p in getattr(state, "ground_cube_positions", ()) or ():
        cv2.circle(out, (int(p[0]), int(p[1])), 16, _C_CUBE, 2)


def _draw_plan(out, planner, state) -> None:
    """The route, the destination and the stick — the point of this module."""
    if planner is None or state is None or not state.player_pos:
        return
    px, py = int(state.player_pos[0]), int(state.player_pos[1])
    status = planner.status()

    pts = [(int(p[0]), int(p[1])) for p in getattr(planner, "path_pixels", []) or []]
    if len(pts) >= 2:
        colour = (0, 140, 255) if status.escaping_gas else _C_PATH
        cv2.polylines(out, [np.array(pts, np.int32)], False, (0, 0, 0), 9)
        cv2.polylines(out, [np.array(pts, np.int32)], False, colour, 4)
        for p in pts[1:-1]:
            cv2.circle(out, p, 5, colour, -1)

    wp = planner.waypoint_pixels() if hasattr(planner, "waypoint_pixels") else None
    if wp is not None:
        wx, wy = int(wp[0]), int(wp[1])
        cv2.drawMarker(out, (wx, wy), (0, 0, 0), cv2.MARKER_TILTED_CROSS, 46, 9)
        cv2.drawMarker(out, (wx, wy), _C_WAYPOINT, cv2.MARKER_TILTED_CROSS, 42, 4)
        cv2.circle(out, (wx, wy), 22, _C_WAYPOINT, 2)
        cv2.line(out, (px, py), (wx, wy), _C_WAYPOINT, 1, cv2.LINE_AA)

    move = getattr(planner, "_smoothed_move", (0.0, 0.0))
    if move != (0.0, 0.0):
        tip = (int(px + move[0] * 150), int(py + move[1] * 150))
        cv2.arrowedLine(out, (px, py), tip, (0, 0, 0), 11, tipLength=0.3)
        cv2.arrowedLine(out, (px, py), tip, _C_STICK, 5, tipLength=0.3)


def _draw_minimap(out, world, planner, state) -> None:
    """The whole fused world map, including the parts that are off screen."""
    px_per_cell = 3
    mw, mh = world.gw * px_per_cell, world.gh * px_per_cell
    mini = np.full((mh, mw, 3), 30, np.uint8)

    occ = world.occupancy
    mini[np.kron(occ, np.ones((px_per_cell, px_per_cell), bool))] = _C_WALL
    unexplored = np.kron(world.unexplored, np.ones((px_per_cell, px_per_cell), bool))
    mini[unexplored] = (48, 48, 48)
    gas_up = np.kron(world.gas, np.ones((px_per_cell, px_per_cell)))
    gmask = gas_up > 0.08
    if gmask.any():
        mini[gmask] = (np.array(_C_GAS) * np.clip(gas_up[gmask] * 1.6, 0, 1)[:, None]
                       ).astype(np.uint8)

    # The current screen window, so it is obvious what is memory vs. sight.
    ox, oy = int(round(world.origin[0])), int(round(world.origin[1]))
    cv2.rectangle(mini, (ox * px_per_cell, oy * px_per_cell),
                  ((ox + world.screen_gw) * px_per_cell,
                   (oy + world.screen_gh) * px_per_cell), (200, 200, 200), 1)

    path = getattr(planner, "path", []) or []
    if len(path) >= 2:
        pts = np.array([[int((c[0] + .5) * px_per_cell), int((c[1] + .5) * px_per_cell)]
                        for c in path], np.int32)
        cv2.polylines(mini, [pts], False, _C_PATH, 2)
    if planner is not None and getattr(planner, "waypoint_cell", None) is not None:
        wc = planner.waypoint_cell
        cv2.drawMarker(mini, (int((wc[0] + .5) * px_per_cell), int((wc[1] + .5) * px_per_cell)),
                       _C_WAYPOINT, cv2.MARKER_TILTED_CROSS, 14, 2)
    if state is not None and state.player_pos:
        pc = world.to_cell(state.player_pos)
        cv2.circle(mini, (int((pc[0] + .5) * px_per_cell), int((pc[1] + .5) * px_per_cell)),
                   4, _C_PLAYER, -1)

    H, W = out.shape[:2]
    x0, y0 = W - mw - 12, 12
    if x0 < 0 or y0 + mh > H:
        return
    cv2.rectangle(out, (x0 - 2, y0 - 2), (x0 + mw + 2, y0 + mh + 2), (200, 200, 200), 1)
    out[y0:y0 + mh, x0:x0 + mw] = mini
    cv2.putText(out, "world map (fused)", (x0, y0 + mh + 16), _FONT, 0.45, _C_TEXT, 1)


def _draw_panel(out, status, state, intent, reward, tick) -> None:
    lines = [f"tick {tick}"]
    if intent is not None:
        lines.append(f"intent  {getattr(intent, 'move_label', '?')}"
                     + ("  ATTACK" if getattr(intent, 'fire_attack', False) else "")
                     + ("  SUPER" if getattr(intent, 'fire_super', False) else ""))
        raw = getattr(intent, "raw", None)
        if raw:
            lines.append(f"action  heading={raw[0]} dist={raw[1]} atk={raw[2]} sup={raw[3]}")
    if status is not None:
        lines.append(f"planner {status.mode}"
                     f"  {'COMMITTED' if status.active else 'free'}"
                     f"  progress {status.progress:.2f}"
                     + ("  BLOCKED" if status.blocked else "")
                     + ("  GAS-ESCAPE" if status.escaping_gas else ""))
        lines.append(f"waypoint d={status.distance:.2f} "
                     f"({status.waypoint_dx:+.2f},{status.waypoint_dy:+.2f})")
    if state is not None:
        lines.append(f"hp {state.health}  ammo {state.ammo_count}"
                     f"{'' if state.ammo_known else '?'}  "
                     f"super {(state.super_charge or 0) * 100:.0f}%  "
                     f"cubes {state.cube_count}  left {state.players_left}"
                     f"{'  IN GAS' if state.in_gas else ''}")
        src = getattr(state, "anchor_source", None)
        if src:
            # On the frame itself, because "is the blue circle actually on the
            # player" is the first question to ask of any trace that looks wrong.
            lines.append(f"anchor  {src}"
                         + ("" if getattr(state, "anchor_fresh", False) else "  COASTED"))
    if reward is not None:
        total = getattr(reward, "total", None)
        bd = getattr(reward, "breakdown", {}) or {}
        top = sorted(bd.items(), key=lambda kv: -abs(kv[1]))[:4]
        lines.append(f"reward {total:+.3f}  " +
                     "  ".join(f"{k}={v:+.2f}" for k, v in top if abs(v) > 1e-9))

    pad, lh = 10, 22
    box_h = lh * len(lines) + 2 * pad
    box_w = max(int(9.2 * len(s)) for s in lines) + 2 * pad
    box_w = min(box_w, out.shape[1] - 20)
    y0 = out.shape[0] - box_h - 10
    overlay = out.copy()
    cv2.rectangle(overlay, (10, y0), (10 + box_w, y0 + box_h), _C_PANEL, -1)
    cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)
    for i, s in enumerate(lines):
        cv2.putText(out, s, (10 + pad, y0 + pad + lh * (i + 1) - 6),
                    _FONT, 0.5, _C_TEXT, 1, cv2.LINE_AA)
    _draw_legend(out)


def _draw_legend(out) -> None:
    """Without this the overlay is a colourful mystery a week later."""
    items = [("wall (fused)", _C_WALL), ("too tight", (90, 90, 90)),
             ("gas", _C_GAS), ("A* path", _C_PATH), ("waypoint", _C_WAYPOINT),
             ("joystick", _C_STICK), ("player", _C_PLAYER), ("enemy", _C_ENEMY)]
    x, y = 12, 12
    overlay = out.copy()
    cv2.rectangle(overlay, (x - 4, y - 4), (x + 132, y + 18 * len(items) + 4),
                  _C_PANEL, -1)
    cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)
    for i, (label, col) in enumerate(items):
        yy = y + 18 * i + 12
        cv2.rectangle(out, (x + 2, yy - 8), (x + 16, yy + 2), col, -1)
        cv2.putText(out, label, (x + 22, yy), _FONT, 0.42, _C_TEXT, 1, cv2.LINE_AA)


def _record(planner, state, intent, reward, tick) -> dict:
    s = planner.status() if planner is not None else None
    rec = {"t": round(time.time(), 3), "tick": tick}
    if intent is not None:
        rec["action"] = list(getattr(intent, "raw", ()) or ())
        rec["move"] = [round(v, 3) for v in getattr(intent, "move", (0, 0))]
        rec["label"] = getattr(intent, "move_label", None)
    if s is not None:
        rec["planner"] = {
            "mode": s.mode, "active": s.active, "blocked": s.blocked,
            "replanned": s.replanned, "escaping_gas": s.escaping_gas,
            "progress": round(s.progress, 3), "distance": round(s.distance, 3),
        }
    if planner is not None:
        rec["waypoint_cell"] = (list(planner.waypoint_cell)
                                if getattr(planner, "waypoint_cell", None) else None)
        rec["path_cells"] = [list(c) for c in (getattr(planner, "path", []) or [])]
    if state is not None:
        rec["state"] = {
            "hp": state.health, "ammo": state.ammo_count, "cubes": state.cube_count,
            "in_gas": state.in_gas, "enemies": len(state.enemy_positions),
            "players_left": state.players_left, "alive": state.is_alive,
            "player_pos": list(state.player_pos) if state.player_pos else None,
            # Provenance, not position: "digits" is the trustworthy path, any
            # "ring*" reading is the weak fallback and a prime suspect whenever
            # the route in this frame looks wrong.
            "anchor_source": getattr(state, "anchor_source", None),
            "anchor_fresh": getattr(state, "anchor_fresh", None),
        }
    if reward is not None:
        rec["reward"] = round(float(getattr(reward, "total", 0.0)), 4)
        rec["breakdown"] = {k: round(float(v), 4)
                            for k, v in (getattr(reward, "breakdown", {}) or {}).items()
                            if abs(v) > 1e-9}
    return rec


# --- offline smoke test ------------------------------------------------------ #
if __name__ == "__main__":
    import argparse
    import sys

    root = pathlib.Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))

    ap = argparse.ArgumentParser(description="Render one trace frame from a screenshot.")
    ap.add_argument("image", nargs="?", default=str(root / "media" / "fixtures" / "showdown.png"))
    ap.add_argument("--heading", type=int, default=2)
    ap.add_argument("--dist", type=int, default=2)
    args = ap.parse_args()

    from perception.getTerrain import find_terrain, play_rect
    from perception.getGas import gas_info
    from perception.getAnchor import find_player_position
    from rl.path_planner import WaypointPlanner
    from rl.state import GameState

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"could not read {args.image}")

    anchor = find_player_position(img)
    player = (int(anchor[0]), int(anchor[1])) if anchor else (img.shape[1] // 2,
                                                              img.shape[0] // 2)
    terrain = find_terrain(img, player_pos=player)
    rect = terrain["rect"] if terrain else play_rect(img)
    gas = gas_info(img, grid_rect=rect, grid_size=(48, 27))
    state = GameState(player_pos=player, in_gas=gas["in_gas"],
                      gas_safe=gas["safe_vector"], gas_sides=gas["sides"],
                      frame_size=(img.shape[1], img.shape[0]))

    planner = WaypointPlanner()
    for _ in range(3):        # let the map fuse a little
        move = planner.plan(args.heading, args.dist, state,
                            (img.shape[1], img.shape[0]),
                            terrain=terrain, gas_grid=gas["grid"])

    tracer = DecisionTracer(TraceConfig(every_seconds=0.0))
    from rl.actions import decode_action
    intent = decode_action([args.heading, args.dist], state=state,
                           frame_size=(img.shape[1], img.shape[0]),
                           planner=planner, terrain=terrain, gas_grid=gas["grid"])
    path = tracer.capture(img, planner, state, intent, None, tick=0, force=True)
    print(f"joystick {move}")
    print(f"path cells: {len(planner.path)}  waypoint {planner.waypoint_cell}")
    print(f"wrote {path}")
