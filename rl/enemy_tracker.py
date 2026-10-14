"""
rl/enemy_tracker.py
===================

Who is where, which way they are going, and who just walked into a bush.

WHY PER-FRAME DETECTIONS ARE NOT ENOUGH
---------------------------------------
`GameState.enemies` is a list of what perception saw on ONE frame, in SCREEN
pixels. Three things a fight depends on are not in it:

  * MOTION. Aiming a slow projectile at where an enemy IS misses a moving
    target -- the lead problem `brawlers/abilities.py` gives as the reason
    marksmen aim by hand. A lead needs a velocity, and a velocity needs the
    same enemy identified on two frames.
  * MEMORY. Bushes are the defining mechanic of Showdown: an enemy that steps
    into one vanishes from the screen while standing three tiles away. A
    per-frame view reports "no enemies" and walks straight into him. So does
    anything that reads the screen alone after the enemy scrolls off it.
  * WORLD FRAME. The camera follows the player, so a stationary enemy's
    SCREEN position moves every tick the player does. Velocity measured in
    screen pixels is the player's own motion, negated.

So detections are associated into TRACKS kept in a world frame: screen pixels
minus the accumulated camera scroll, which is exactly the quantity
`rl/camera_tracker.py` already measures for waypoint latching. A stationary
enemy then has a constant world position whatever the player does, and a
moving one has a real velocity.

STALE READINGS
--------------
`LivePerception` refreshes entities every 0.5s and RE-SERVES the cached list in
between, in the screen frame of the tick that produced it. Converting that
stale list with a NEWER camera offset makes every enemy appear to slide
backwards at exactly the player's speed. `GameState.entities_fresh` says which
ticks carry a new reading; on the others the tracker only predicts.

FAILURE DIRECTION
-----------------
Everything here errs toward REMEMBERING. A remembered enemy that has actually
left costs a slightly over-cautious route; a forgotten one standing in the
bush beside you costs the match. `memory_ticks` is sized from that asymmetry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from typing import Dict, List, Optional, Sequence, Tuple

Point = Tuple[float, float]


@dataclass(frozen=True)
class TrackerConfig:
    # Association gate: how far (fraction of the frame's short side) a
    # detection may sit from a track's PREDICTED position and still be the same
    # object. Grows with the time since the track was last seen, because a
    # longer gap allows a longer walk.
    gate_frac: float = 0.10
    # Largest believable world speed, fraction of the short side per tick.
    # A brawler covers ~2.4 tiles/s; at ~17 tiles across a 20:9 screen and a
    # 0.1s tick that is ~0.03 of the short side per tick. 0.07 leaves room for
    # dashes and supers without letting a track teleport across the screen.
    max_speed_frac: float = 0.07
    # Ceiling on the growing gate. Without it a track unseen for a couple of
    # seconds would claim any new detection anywhere on screen -- a different
    # enemy walking in from the far side would inherit the old one's identity
    # and a nonsense velocity.
    max_gate_frac: float = 0.40
    # EMA weight of a new velocity measurement. Detections are noisy to a few
    # pixels, so a single reading is not trusted outright.
    vel_alpha: float = 0.55
    # How long an unseen ENEMY is remembered, in ticks. 30 ticks = 3s at the
    # default 0.1s tick: long enough to cover the classic "stepped into the
    # bush and waited" ambush, short enough that a departed enemy stops
    # shaping routes before it matters.
    memory_ticks: int = 30
    # Beyond this many ticks unseen a track's position is no longer
    # extrapolated: the enemy has had time to change his mind, and a velocity
    # projected for seconds puts him somewhere he never went.
    predict_ticks: int = 8
    # Boxes do not move, so they are remembered far longer -- the agent should
    # be able to come back for a box it saw while it was busy fleeing.
    box_memory_ticks: int = 900
    # Ground cubes are static too, but someone else may pick them up.
    cube_memory_ticks: int = 150
    # A static object whose remembered position is ON SCREEN but has not been
    # detected on this many consecutive FRESH readings is gone (destroyed,
    # collected). Enemies are exempt: an on-screen enemy that vanishes is in a
    # bush, which is the case memory exists for.
    static_miss_limit: int = 3
    # Screen margin (fraction of each side) inside which "on screen" holds for
    # the miss rule above; detectors are unreliable right at the frame edge.
    onscreen_margin: float = 0.06
    # Recent HP drop that marks an enemy as "in a fight" (fraction of the
    # largest HP seen on that track) and how long that flag lasts.
    hurt_frac: float = 0.12
    hurt_ticks: int = 25


@dataclass
class Track:
    id: int
    kind: str                         # "enemy" | "box" | "cube"
    wpos: Point                       # world px at last observation
    vel: Point = (0.0, 0.0)           # world px per tick
    hp: Optional[int] = None
    hp_max_seen: Optional[int] = None
    cubes: Optional[int] = None
    first_tick: int = 0
    last_tick: int = 0
    hits: int = 1
    misses: int = 0                   # consecutive fresh on-screen misses
    hurt_until: int = -1              # tick until which it counts as "in a fight"
    _hp_hist: List[Tuple[int, int]] = field(default_factory=list)

    def age(self, tick: int) -> int:
        """Ticks since this track was last actually seen."""
        return tick - self.last_tick

    def visible(self, tick: int) -> bool:
        return self.age(tick) == 0

    def hp_frac(self) -> Optional[float]:
        if self.hp is None or not self.hp_max_seen:
            return None
        return max(0.0, min(1.0, self.hp / float(self.hp_max_seen)))


class EnemyTracker:
    """Associates per-tick detections into world-frame tracks.

    Call `update(state, camera_delta)` once per tick, AFTER perception and
    with the same camera delta the planner receives. Read back with
    `enemies()`, `boxes()`, `cubes()`, `screen_pos()` and `predict()`.
    """

    def __init__(self, config: Optional[TrackerConfig] = None):
        self.config = config or TrackerConfig()
        self.reset()

    def reset(self) -> None:
        self.tick = 0
        self.cam = [0.0, 0.0]         # accumulated camera scroll, px
        self.short = 720.0
        self.frame_size = (1280, 720)
        self._next_id = 1
        self._tracks: Dict[str, List[Track]] = {"enemy": [], "box": [], "cube": []}

    # ------------------------------------------------------------------ #
    def to_world(self, screen_pt) -> Point:
        return (float(screen_pt[0]) - self.cam[0], float(screen_pt[1]) - self.cam[1])

    def to_screen(self, world_pt) -> Point:
        return (float(world_pt[0]) + self.cam[0], float(world_pt[1]) + self.cam[1])

    def predict(self, track: Track, ahead_ticks: float = 0.0) -> Point:
        """World position now (+ `ahead_ticks`), extrapolating only briefly."""
        cfg = self.config
        dt = min(float(track.age(self.tick)), float(cfg.predict_ticks))
        dt += max(0.0, float(ahead_ticks))
        if track.kind != "enemy":
            dt = 0.0
        return (track.wpos[0] + track.vel[0] * dt, track.wpos[1] + track.vel[1] * dt)

    def screen_pos(self, track: Track, ahead_ticks: float = 0.0) -> Point:
        return self.to_screen(self.predict(track, ahead_ticks))

    def screen_vel(self, track: Track) -> Point:
        """Velocity in screen px/tick of the WORLD motion (camera removed)."""
        return track.vel

    # ------------------------------------------------------------------ #
    def update(self, state, camera_delta=(0.0, 0.0)) -> None:
        cfg = self.config
        self.tick += 1
        self.cam[0] += float(camera_delta[0])
        self.cam[1] += float(camera_delta[1])
        fs = getattr(state, "frame_size", None) or self.frame_size
        self.frame_size = (int(fs[0]), int(fs[1]))
        self.short = float(max(1, min(self.frame_size)))

        if state is None:
            return

        if getattr(state, "entities_fresh", True):
            enemies = state.enemy_list() if hasattr(state, "enemy_list") else []
            boxes = state.box_list() if hasattr(state, "box_list") else []
            self._associate("enemy", [(e.pos, e.hp, e.cubes) for e in enemies])
            self._associate("box", [(b.pos, b.hp, None) for b in boxes])
        if getattr(state, "pickups_fresh", True):
            cubes = list(getattr(state, "ground_cube_positions", ()) or ())
            self._associate("cube", [(c, None, None) for c in cubes])
        self._expire()

    # ------------------------------------------------------------------ #
    def _associate(self, kind: str, detections: Sequence) -> None:
        cfg = self.config
        tracks = self._tracks[kind]
        dets = [(self.to_world(p), hp, cubes, p) for p, hp, cubes in detections]

        # Greedy nearest-pair matching. There are rarely more than three or
        # four enemies on screen, where greedy and optimal assignment agree;
        # the Hungarian algorithm would buy nothing but a dependency.
        pairs = []
        for ti, t in enumerate(tracks):
            pred = self.predict(t)
            age = max(1, t.age(self.tick))
            if kind == "enemy":
                gate = min(cfg.gate_frac + cfg.max_speed_frac * age,
                           cfg.max_gate_frac) * self.short
            else:
                gate = cfg.gate_frac * self.short
            for di, (w, _hp, _c, _p) in enumerate(dets):
                d = hypot(w[0] - pred[0], w[1] - pred[1])
                if d <= gate:
                    pairs.append((d, ti, di))
        pairs.sort()
        used_t, used_d = set(), set()
        for _d, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            self._refresh(tracks[ti], dets[di])

        # Unmatched detections start new tracks.
        for di, det in enumerate(dets):
            if di in used_d:
                continue
            w, hp, cubes, _p = det
            t = Track(id=self._next_id, kind=kind, wpos=w, hp=hp,
                      hp_max_seen=hp, cubes=cubes,
                      first_tick=self.tick, last_tick=self.tick)
            if hp is not None:
                t._hp_hist.append((self.tick, hp))
            self._next_id += 1
            tracks.append(t)

        # Unmatched tracks: static objects that SHOULD have been seen count a
        # miss; enemies are left alone (bush, occlusion -- see module docs).
        if kind != "enemy":
            for ti, t in enumerate(tracks):
                if ti in used_t or t.last_tick == self.tick:
                    continue
                if self._on_screen(self.to_screen(t.wpos)):
                    t.misses += 1

    def _refresh(self, t: Track, det) -> None:
        cfg = self.config
        w, hp, cubes, _p = det
        age = max(1, t.age(self.tick))
        if t.kind == "enemy":
            vx = (w[0] - t.wpos[0]) / age
            vy = (w[1] - t.wpos[1]) / age
            vmax = cfg.max_speed_frac * self.short
            sp = hypot(vx, vy)
            if sp > vmax:
                vx, vy = vx * vmax / sp, vy * vmax / sp
            a = cfg.vel_alpha if t.hits > 1 else 1.0
            t.vel = (t.vel[0] + a * (vx - t.vel[0]), t.vel[1] + a * (vy - t.vel[1]))
        t.wpos = w
        t.last_tick = self.tick
        t.hits += 1
        t.misses = 0
        if cubes is not None:
            t.cubes = cubes
        if hp is not None:
            t.hp = hp
            t.hp_max_seen = max(t.hp_max_seen or hp, hp)
            t._hp_hist.append((self.tick, hp))
            horizon = self.tick - cfg.hurt_ticks
            t._hp_hist = [x for x in t._hp_hist if x[0] >= horizon]
            peak = max(v for _, v in t._hp_hist)
            if t.hp_max_seen and (peak - hp) >= cfg.hurt_frac * t.hp_max_seen:
                t.hurt_until = self.tick + cfg.hurt_ticks

    def _on_screen(self, sp) -> bool:
        w, h = self.frame_size
        m = self.config.onscreen_margin
        return (m * w <= sp[0] <= (1 - m) * w) and (m * h <= sp[1] <= (1 - m) * h)

    def _expire(self) -> None:
        cfg = self.config
        limits = {"enemy": cfg.memory_ticks, "box": cfg.box_memory_ticks,
                  "cube": cfg.cube_memory_ticks}
        for kind, tracks in self._tracks.items():
            keep = []
            for t in tracks:
                if t.age(self.tick) > limits[kind]:
                    continue
                if kind != "enemy" and t.misses >= cfg.static_miss_limit:
                    continue
                keep.append(t)
            self._tracks[kind] = keep

    # ------------------------------------------------------------------ #
    def tracks(self, kind: str) -> List[Track]:
        return list(self._tracks.get(kind, ()))

    def enemies(self) -> List[Track]:
        return self.tracks("enemy")

    def boxes(self) -> List[Track]:
        return self.tracks("box")

    def cubes(self) -> List[Track]:
        return self.tracks("cube")

    def hidden_enemies(self) -> List[Track]:
        """Enemies remembered but not on this tick's detections."""
        return [t for t in self.enemies() if not t.visible(self.tick)]

    def in_fight(self, t: Track) -> bool:
        """Has this enemy lost a real chunk of health recently?"""
        return t.hurt_until >= self.tick

    def nearest(self, kind: str, player_px, visible_only: bool = False) -> Optional[Track]:
        if player_px is None:
            return None
        best, best_d = None, float("inf")
        for t in self._tracks.get(kind, ()):
            if visible_only and not t.visible(self.tick):
                continue
            sx, sy = self.screen_pos(t)
            d = hypot(sx - player_px[0], sy - player_px[1])
            if d < best_d:
                best, best_d = t, d
        return best

    def summary(self) -> dict:
        """Counts for the trace panel / UI."""
        e = self.enemies()
        return {
            "enemies": len(e),
            "hidden": sum(1 for t in e if not t.visible(self.tick)),
            "boxes": len(self.boxes()),
            "cubes": len(self.cubes()),
        }
