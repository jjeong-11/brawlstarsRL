"""
sim/arena.py
============

The world: a real Showdown layout, ten brawlers, boxes, cubes, bushes, the
closing gas, projectiles, reload and regeneration. Everything here is in TILES
and SECONDS; pixels only exist in `sim/harness.py`, which renders this into
what the bot's perception would have reported.

The numbers are Brawl Stars' own where they are well known (tile = 300 game
units, normal speed 720 units/s = 2.4 tiles/s, 3 ammo, regen 13%/s after 3s
out of combat, +10% health and damage per power cube) and class archetypes
where they are not (per-brawler damage and health vary too much to matter for
comparing strategies, and inventing 106 precise stat lines would only be
false precision).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# brawler archetypes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Stats:
    name: str
    cls: str
    hp: float
    damage: float              # per shot, before cubes
    range_tiles: float
    reload_s: float = 1.6      # per ammo
    ammo: int = 3
    speed: float = 2.4         # tiles / s
    proj_speed: float = 11.0   # tiles / s
    proj_radius: float = 0.30  # tiles (half-width of the hit corridor)
    lob: bool = False
    splash: float = 0.0        # tiles, lobbed attacks
    lob_flight_s: float = 0.7
    attack_lock_s: float = 0.45   # minimum time between shots
    super_mult: float = 2.5
    super_charge_per_hit: float = 0.25


# Class -> (hp, damage, reload, speed, proj_speed). Deliberately coarse.
_ARCH = {
    "Tank": (7400, 1900, 1.5, 2.6, 10.0),
    "Damage Dealer": (4300, 1500, 1.5, 2.4, 11.0),
    "Marksman": (3400, 1900, 1.9, 2.4, 13.0),
    "Artillery": (3500, 1350, 1.8, 2.4, 0.0),
    "Assassin": (4300, 1700, 1.4, 2.7, 11.0),
    "Support": (4400, 1300, 1.5, 2.4, 11.0),
    "Controller": (4000, 1350, 1.6, 2.4, 11.0),
}


def stats_for(ability) -> Stats:
    """Archetype stats for a `brawlers.abilities.Ability` (or None)."""
    cls = getattr(ability, "brawler_class", None) or "Damage Dealer"
    hp, dmg, reload_s, speed, pspeed = _ARCH.get(cls, _ARCH["Damage Dealer"])
    rng = getattr(ability, "range_tiles", None) or 7.0
    lob = bool(getattr(ability, "lobs_over_walls", False))
    # Short-range brawlers hit harder: that is the whole trade of the class.
    if rng < 4.5:
        dmg *= 1.45
    elif rng >= 9.0:
        dmg *= 0.95
    return Stats(name=getattr(ability, "id", "generic"), cls=cls, hp=hp, damage=dmg,
                 range_tiles=float(rng), reload_s=reload_s, speed=speed,
                 proj_speed=pspeed if not lob else 0.0, lob=lob,
                 splash=1.1 if lob else 0.0,
                 proj_radius=0.45 if rng < 4.5 else 0.30)


# --------------------------------------------------------------------------- #
# entities
# --------------------------------------------------------------------------- #
BODY_RADIUS = 0.42          # tiles: wall collision
HIT_RADIUS = 0.55           # tiles: what a projectile has to touch
PICKUP_RADIUS = 0.85
REVEAL_S = 1.2              # attacking from a bush reveals you this long
BUSH_SEE_TILES = 2.2        # anyone this close sees into your bush


@dataclass
class Brawler:
    idx: int
    stats: Stats
    x: float
    y: float
    hp: float = 0.0
    cubes: int = 0
    ammo: float = 3.0
    super_charge: float = 0.0
    alive: bool = True
    rank: Optional[int] = None
    kills: int = 0
    damage_dealt: float = 0.0
    cubes_collected: int = 0
    boxes_broken: int = 0
    last_shot_t: float = -99.0
    last_attack_t: float = -99.0
    last_hurt_t: float = -99.0
    last_hurt_by: Optional[int] = None
    revealed_until: float = -99.0
    death_cause: Optional[str] = None
    death_t: Optional[float] = None
    move: Tuple[float, float] = (0.0, 0.0)     # unit vector, set by controller
    facing: Tuple[float, float] = (1.0, 0.0)
    gas_time: float = 0.0
    shots: int = 0
    hits: int = 0
    box_hits: int = 0

    @property
    def max_hp(self) -> float:
        return self.stats.hp * (1.0 + 0.1 * self.cubes)

    @property
    def dmg_mult(self) -> float:
        return 1.0 + 0.1 * self.cubes


@dataclass
class Box:
    idx: int
    gx: int
    gy: int
    hp: float = 4500.0
    alive: bool = True

    @property
    def x(self) -> float:
        return self.gx + 0.5

    @property
    def y(self) -> float:
        return self.gy + 0.5


@dataclass
class Cube:
    x: float
    y: float
    alive: bool = True


@dataclass
class Projectile:
    owner: int
    x: float
    y: float
    dx: float
    dy: float
    speed: float
    remaining: float
    damage: float
    radius: float
    is_super: bool = False
    lob: bool = False
    tx: float = 0.0
    ty: float = 0.0
    land_t: float = 0.0
    splash: float = 0.0


@dataclass(frozen=True)
class ArenaConfig:
    n_players: int = 10
    n_boxes: int = 32
    box_hp: float = 4500.0
    gas_start_s: float = 35.0
    gas_end_s: float = 155.0
    gas_min_half: float = 3.5
    gas_dps: float = 1000.0
    gas_dps_growth: float = 12.0        # extra dps per second after gas starts
    regen_delay_s: float = 3.0
    regen_frac_per_s: float = 0.13
    max_seconds: float = 260.0


# --------------------------------------------------------------------------- #
class Arena:
    """One match. Controllers set `brawler.move` and call `fire()`; then `step()`."""

    def __init__(self, game_map, bush: np.ndarray, stats: List[Stats],
                 config: Optional[ArenaConfig] = None, seed: int = 0):
        self.cfg = config or ArenaConfig()
        self.rng = random.Random(seed)
        self.map = game_map
        self.name = game_map.name
        occ = np.asarray(game_map.occupancy, bool)
        inb = np.asarray(game_map.in_bounds, bool)
        self.gh, self.gw = occ.shape
        self.bush = np.asarray(bush, bool) & inb
        # Walls block movement and shots; bushes block neither.
        self.wall = (occ & ~self.bush) | ~inb
        self.in_bounds = inb
        self.box_at = np.full(occ.shape, -1, np.int32)
        self.t = 0.0
        self.cx, self.cy = self.gw / 2.0, self.gh / 2.0
        self.half0 = max(self.gw, self.gh) / 2.0 + 1.0
        self.projectiles: List[Projectile] = []
        self.cubes: List[Cube] = []
        self.boxes: List[Box] = []
        self.events: List[tuple] = []
        self.flow_cache: Dict[Tuple[int, int], np.ndarray] = {}

        # The final safe square must contain somewhere to stand. Many arenas
        # have a pond or a fort dead centre; a 7x7 final zone over it would
        # make every match end with everyone dying in the gas.
        self.min_half = self._min_half(self.cfg.gas_min_half, 40)

        spawns = self._spawn_points(len(stats))
        self.brawlers: List[Brawler] = []
        for i, (st, (x, y)) in enumerate(zip(stats, spawns)):
            b = Brawler(idx=i, stats=st, x=x, y=y, hp=st.hp, ammo=float(st.ammo))
            self.brawlers.append(b)
        self._place_boxes(spawns)

    # ------------------------------------------------------------------ #
    # setup
    # ------------------------------------------------------------------ #
    def _free_cells(self) -> List[Tuple[int, int]]:
        """Walkable cells in the arena's largest connected region.

        Layout extraction reads water and thin bridges as blocked, which can
        cut an island off from the middle. A spawn or a box there is an
        artefact of the extraction, not a property of the map -- a brawler
        spawned on it could never reach the safe zone.
        """
        if getattr(self, "_free_cache", None) is not None:
            return self._free_cache
        import cv2
        free = (~self.wall & self.in_bounds).astype(np.uint8)
        n, labels = cv2.connectedComponents(free, connectivity=8)
        if n > 2:
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0
            keep = labels == int(np.argmax(sizes))
        else:
            keep = free.astype(bool)
        ys, xs = np.nonzero(keep)
        self._free_cache = list(zip(xs.tolist(), ys.tolist()))
        return self._free_cache

    def _min_half(self, start: float, cells: int) -> float:
        free = set(self._free_cells())
        h = float(start)
        while h < self.half0:
            n = 0
            for gx, gy in free:
                if max(abs(gx + 0.5 - self.cx), abs(gy + 0.5 - self.cy)) <= h:
                    n += 1
            if n >= cells:
                return h
            h += 0.5
        return h

    def _spawn_points(self, n: int) -> List[Tuple[float, float]]:
        """Spread around the outer ring, the way Showdown spawns players."""
        cells = self._free_cells()
        ring = []
        for gx, gy in cells:
            # Chebyshev distance from the centre, as a fraction of the half size
            d = max(abs(gx + 0.5 - self.cx), abs(gy + 0.5 - self.cy)) / self.half0
            if 0.55 <= d <= 0.88 and self._clear_disc(gx + 0.5, gy + 0.5, 0.9):
                ring.append((gx + 0.5, gy + 0.5))
        if len(ring) < n:
            ring = [(gx + 0.5, gy + 0.5) for gx, gy in cells
                    if self._clear_disc(gx + 0.5, gy + 0.5, 0.6)]
        self.rng.shuffle(ring)
        chosen = [ring[0]]
        while len(chosen) < n:
            best, best_d = None, -1.0
            for c in ring[:600]:
                d = min((c[0] - p[0]) ** 2 + (c[1] - p[1]) ** 2 for p in chosen)
                if d > best_d:
                    best, best_d = c, d
            chosen.append(best)
        self.rng.shuffle(chosen)
        return chosen

    def _place_boxes(self, spawns) -> None:
        cells = [c for c in self._free_cells() if not self.bush[c[1], c[0]]]
        self.rng.shuffle(cells)
        placed: List[Tuple[int, int]] = []
        for gx, gy in cells:
            if len(placed) >= self.cfg.n_boxes:
                break
            x, y = gx + 0.5, gy + 0.5
            if any((x - sx) ** 2 + (y - sy) ** 2 < 3.0 ** 2 for sx, sy in spawns):
                continue
            if any((gx - px) ** 2 + (gy - py) ** 2 < 3.0 ** 2 for px, py in placed):
                continue
            # Never seal a corridor: keep boxes off cells with walls both sides.
            if (self.wall[gy, max(0, gx - 1)] and self.wall[gy, min(self.gw - 1, gx + 1)]) or \
               (self.wall[max(0, gy - 1), gx] and self.wall[min(self.gh - 1, gy + 1), gx]):
                continue
            placed.append((gx, gy))
        for i, (gx, gy) in enumerate(placed):
            self.boxes.append(Box(idx=i, gx=gx, gy=gy, hp=self.cfg.box_hp))
            self.box_at[gy, gx] = i

    # ------------------------------------------------------------------ #
    # geometry
    # ------------------------------------------------------------------ #
    def solid(self, gx: int, gy: int) -> bool:
        if gx < 0 or gy < 0 or gx >= self.gw or gy >= self.gh:
            return True
        return bool(self.wall[gy, gx]) or self.box_at[gy, gx] >= 0

    def _clear_disc(self, x: float, y: float, r: float) -> bool:
        for gy in range(int(math.floor(y - r)), int(math.floor(y + r)) + 1):
            for gx in range(int(math.floor(x - r)), int(math.floor(x + r)) + 1):
                if not self.solid(gx, gy):
                    continue
                nx = min(max(x, gx), gx + 1.0)
                ny = min(max(y, gy), gy + 1.0)
                if (x - nx) ** 2 + (y - ny) ** 2 < r * r:
                    return False
        return True

    def line_clear(self, x0, y0, x1, y1, step: float = 0.35) -> bool:
        """Does a straight shot from a to b avoid walls and boxes?

        The target's own cell is not an obstruction -- a box would otherwise
        block the line to itself, and nobody would ever shoot one.
        """
        d = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(d / step))
        end = (int(math.floor(x1)), int(math.floor(y1)))
        for i in range(1, n):
            f = i / n
            gx = int(math.floor(x0 + (x1 - x0) * f))
            gy = int(math.floor(y0 + (y1 - y0) * f))
            if (gx, gy) == end:
                break
            if self.solid(gx, gy):
                return False
        return True

    def gas_half(self, t: Optional[float] = None) -> float:
        t = self.t if t is None else t
        c = self.cfg
        if t <= c.gas_start_s:
            return self.half0
        f = min(1.0, (t - c.gas_start_s) / max(1e-6, c.gas_end_s - c.gas_start_s))
        return self.half0 + (self.min_half - self.half0) * f

    def in_gas(self, x: float, y: float, t: Optional[float] = None) -> bool:
        h = self.gas_half(t)
        return max(abs(x - self.cx), abs(y - self.cy)) > h

    def hidden(self, target: Brawler, viewer_x: float, viewer_y: float) -> bool:
        """Is `target` concealed from a viewer at (x, y)?"""
        gx, gy = int(target.x), int(target.y)
        if not (0 <= gx < self.gw and 0 <= gy < self.gh) or not self.bush[gy, gx]:
            return False
        if self.t < target.revealed_until:
            return False
        return math.hypot(target.x - viewer_x, target.y - viewer_y) > BUSH_SEE_TILES

    def alive(self) -> List[Brawler]:
        return [b for b in self.brawlers if b.alive]

    # ------------------------------------------------------------------ #
    # actions
    # ------------------------------------------------------------------ #
    def fire(self, b: Brawler, aim: Optional[Tuple[float, float]] = None,
             frac: float = 1.0, is_super: bool = False) -> bool:
        """Attack (or super). `aim` None = auto-aim, as a tap does in game."""
        if not b.alive:
            return False
        st = b.stats
        if is_super:
            if b.super_charge < 1.0:
                return False
        else:
            if b.ammo < 1.0 or self.t - b.last_shot_t < st.attack_lock_s:
                return False
        rng_t = st.range_tiles
        if aim is None:
            target = self._auto_target(b, rng_t)
            if target is None:
                dx, dy = b.facing
                dist = rng_t
            else:
                dx, dy = target[0] - b.x, target[1] - b.y
                dist = math.hypot(dx, dy)
                if dist > 1e-6:
                    dx, dy = dx / dist, dy / dist
                else:
                    dx, dy = b.facing
        else:
            dx, dy = aim
            n = math.hypot(dx, dy)
            if n < 1e-6:
                return False
            dx, dy = dx / n, dy / n
            dist = rng_t * max(0.15, min(1.0, frac)) if st.lob else rng_t
        dmg = st.damage * b.dmg_mult * (st.super_mult if is_super else 1.0)
        if st.lob:
            dist = min(dist, rng_t)
            self.projectiles.append(Projectile(
                owner=b.idx, x=b.x, y=b.y, dx=dx, dy=dy, speed=0.0, remaining=0.0,
                damage=dmg, radius=0.0, is_super=is_super, lob=True,
                tx=b.x + dx * dist, ty=b.y + dy * dist,
                land_t=self.t + st.lob_flight_s,
                splash=st.splash * (1.4 if is_super else 1.0)))
        else:
            self.projectiles.append(Projectile(
                owner=b.idx, x=b.x, y=b.y, dx=dx, dy=dy, speed=st.proj_speed,
                remaining=rng_t, damage=dmg,
                radius=st.proj_radius * (1.6 if is_super else 1.0), is_super=is_super))
        if is_super:
            b.super_charge = 0.0
        else:
            b.ammo -= 1.0
            b.last_shot_t = self.t
            b.shots += 1
        b.last_attack_t = self.t
        b.revealed_until = self.t + REVEAL_S
        b.facing = (dx, dy)
        return True

    def _auto_target(self, b: Brawler, rng_t: float) -> Optional[Tuple[float, float]]:
        """The game's auto-aim: nearest VISIBLE enemy in range, else nearest box."""
        best, best_d = None, rng_t
        for o in self.brawlers:
            if o is b or not o.alive or self.hidden(o, b.x, b.y):
                continue
            d = math.hypot(o.x - b.x, o.y - b.y)
            if d <= best_d:
                best, best_d = (o.x, o.y), d
        if best is not None:
            return best
        for bx in self.boxes:
            if not bx.alive:
                continue
            d = math.hypot(bx.x - b.x, bx.y - b.y)
            if d <= best_d:
                best, best_d = (bx.x, bx.y), d
        return best

    # ------------------------------------------------------------------ #
    # simulation
    # ------------------------------------------------------------------ #
    def step(self, dt: float) -> None:
        self.t += dt
        self.events = []
        for b in self.brawlers:
            if b.alive:
                self._move(b, dt)
        self._update_projectiles(dt)
        for b in self.brawlers:
            if not b.alive:
                continue
            st = b.stats
            if b.ammo < st.ammo:
                b.ammo = min(float(st.ammo), b.ammo + dt / st.reload_s)
            if self.in_gas(b.x, b.y):
                c = self.cfg
                dps = c.gas_dps + c.gas_dps_growth * max(0.0, self.t - c.gas_start_s)
                b.gas_time += dt
                self._damage(b, dps * dt, None, cause="gas")
            elif (self.t - b.last_hurt_t >= self.cfg.regen_delay_s
                  and self.t - b.last_attack_t >= self.cfg.regen_delay_s):
                b.hp = min(b.max_hp, b.hp + self.cfg.regen_frac_per_s * b.max_hp * dt)
        for b in self.brawlers:
            if not b.alive:
                continue
            for c in self.cubes:
                if c.alive and (c.x - b.x) ** 2 + (c.y - b.y) ** 2 <= PICKUP_RADIUS ** 2:
                    c.alive = False
                    b.cubes += 1
                    b.cubes_collected += 1
                    b.hp += 0.1 * b.stats.hp
        self.cubes = [c for c in self.cubes if c.alive]

    def _move(self, b: Brawler, dt: float) -> None:
        mx, my = b.move
        n = math.hypot(mx, my)
        if n < 1e-6:
            return
        mx, my = mx / n, my / n
        b.facing = (mx, my)
        dist = b.stats.speed * dt
        r = BODY_RADIUS
        nx, ny = b.x + mx * dist, b.y + my * dist
        if self._clear_disc(nx, ny, r):
            b.x, b.y = nx, ny
            return
        # Slide: the game deflects a blocked move along the wall rather than
        # stopping dead, which is what carries a brawler round a corner it
        # clipped. Try the move rotated progressively, at the speed of its
        # component along the original direction.
        for deg in (20, -20, 40, -40, 60, -60, 80, -80):
            a = math.radians(deg)
            ca, sa = math.cos(a), math.sin(a)
            rx, ry = mx * ca - my * sa, mx * sa + my * ca
            k = dist * ca
            nx, ny = b.x + rx * k, b.y + ry * k
            if self._clear_disc(nx, ny, r):
                b.x, b.y = nx, ny
                return

    def _update_projectiles(self, dt: float) -> None:
        keep = []
        for p in self.projectiles:
            if p.lob:
                if self.t >= p.land_t:
                    self._splash(p)
                else:
                    keep.append(p)
                continue
            travel = min(p.remaining, p.speed * dt)
            steps = max(1, int(math.ceil(travel / 0.3)))
            seg = travel / steps
            done = False
            for _ in range(steps):
                p.x += p.dx * seg
                p.y += p.dy * seg
                p.remaining -= seg
                gx, gy = int(math.floor(p.x)), int(math.floor(p.y))
                if gx < 0 or gy < 0 or gx >= self.gw or gy >= self.gh or self.wall[gy, gx]:
                    done = True
                    break
                bi = self.box_at[gy, gx]
                if bi >= 0:
                    self._hit_box(self.boxes[bi], p.damage, p.owner)
                    done = True
                    break
                hit = None
                for o in self.brawlers:
                    if o.idx == p.owner or not o.alive:
                        continue
                    if (o.x - p.x) ** 2 + (o.y - p.y) ** 2 <= (HIT_RADIUS + p.radius) ** 2:
                        hit = o
                        break
                if hit is not None:
                    self._hit_brawler(hit, p)
                    done = True
                    break
            if not done and p.remaining > 1e-6:
                keep.append(p)
        self.projectiles = keep

    def _splash(self, p: Projectile) -> None:
        for o in self.brawlers:
            if o.idx == p.owner or not o.alive:
                continue
            if (o.x - p.tx) ** 2 + (o.y - p.ty) ** 2 <= (p.splash + HIT_RADIUS) ** 2:
                self._hit_brawler(o, p)
        for bx in self.boxes:
            if bx.alive and (bx.x - p.tx) ** 2 + (bx.y - p.ty) ** 2 <= (p.splash + 0.5) ** 2:
                self._hit_box(bx, p.damage, p.owner)

    def _hit_brawler(self, o: Brawler, p: Projectile) -> None:
        owner = self.brawlers[p.owner]
        if not p.is_super:
            owner.super_charge = min(1.0, owner.super_charge + owner.stats.super_charge_per_hit)
            owner.hits += 1
        owner.damage_dealt += min(o.hp, p.damage)
        self._damage(o, p.damage, owner.idx, cause="enemy")

    def _hit_box(self, bx: Box, dmg: float, owner: int) -> None:
        if not bx.alive:
            return
        if 0 <= owner < len(self.brawlers):
            self.brawlers[owner].box_hits += 1
        bx.hp -= dmg
        if bx.hp <= 0:
            bx.alive = False
            self.box_at[bx.gy, bx.gx] = -1
            self.cubes.append(Cube(bx.x, bx.y))
            self.flow_cache.clear()           # a corridor may have opened
            if 0 <= owner < len(self.brawlers):
                self.brawlers[owner].boxes_broken += 1
            self.events.append(("box", bx.idx, owner))

    def _damage(self, o: Brawler, dmg: float, by: Optional[int], cause: str) -> None:
        if not o.alive or dmg <= 0:
            return
        o.hp -= dmg
        o.last_hurt_t = self.t
        if by is not None:
            o.last_hurt_by = by
        if o.hp <= 0:
            alive_before = sum(1 for b in self.brawlers if b.alive)
            o.alive = False
            o.hp = 0.0
            o.rank = alive_before
            o.death_t = self.t
            killer = by if by is not None else (
                o.last_hurt_by if cause == "gas" and self.t - o.last_hurt_t < 0.5 else None)
            o.death_cause = cause if by is None else "enemy"
            if by is not None:
                self.brawlers[by].kills += 1
            # Showdown: a defeated brawler drops its cubes (at least one).
            n = max(1, o.cubes)
            for k in range(n):
                a = 2 * math.pi * k / n + self.rng.uniform(-0.3, 0.3)
                r = 0.0 if n == 1 else 0.7
                cx, cy = o.x + math.cos(a) * r, o.y + math.sin(a) * r
                if self.solid(int(cx), int(cy)):
                    cx, cy = o.x, o.y
                self.cubes.append(Cube(cx, cy))
            self.events.append(("death", o.idx, killer, cause))
            alive = [b for b in self.brawlers if b.alive]
            if len(alive) == 1:
                alive[0].rank = 1

    def over(self) -> bool:
        return sum(1 for b in self.brawlers if b.alive) <= 1 or self.t >= self.cfg.max_seconds

    # ------------------------------------------------------------------ #
    # navigation support (bots)
    # ------------------------------------------------------------------ #
    def nearest_open(self, gx: int, gy: int, max_r: int = 12) -> Tuple[int, int]:
        """`(gx, gy)` if walkable, else the closest walkable cell to it.

        A goal inside a wall or a lake (the centre of Island Invasion is
        water) would otherwise seed a flow field that never reaches open
        ground, and every bot heading there walks into the shore and stays.
        """
        gx = min(max(int(gx), 0), self.gw - 1)
        gy = min(max(int(gy), 0), self.gh - 1)
        if not self.solid(gx, gy):
            return gx, gy
        for r in range(1, max_r + 1):
            best, bd = None, None
            for y in range(gy - r, gy + r + 1):
                for x in range(gx - r, gx + r + 1):
                    if max(abs(x - gx), abs(y - gy)) != r or self.solid(x, y):
                        continue
                    d = (x - gx) ** 2 + (y - gy) ** 2
                    if bd is None or d < bd:
                        best, bd = (x, y), d
            if best is not None:
                return best
        return gx, gy

    def flow_field(self, goal: Tuple[int, int]) -> np.ndarray:
        """Geodesic distance (8-connected, in steps) from every cell to `goal`."""
        goal = self.nearest_open(goal[0], goal[1])
        key = (int(goal[0]), int(goal[1]))
        f = self.flow_cache.get(key)
        if f is not None:
            return f
        import cv2
        passable = (~self.wall & (self.box_at < 0)).astype(np.uint8)
        gx, gy = key
        gx = min(max(gx, 0), self.gw - 1)
        gy = min(max(gy, 0), self.gh - 1)
        dist = np.full((self.gh, self.gw), 1e9, np.float32)
        frontier = np.zeros((self.gh, self.gw), np.uint8)
        frontier[gy, gx] = 1
        visited = frontier.copy()
        dist[gy, gx] = 0
        k = np.ones((3, 3), np.uint8)
        d = 0
        while True:
            d += 1
            nb = cv2.dilate(frontier, k) & passable & (1 - visited)
            if not nb.any():
                break
            dist[nb.astype(bool)] = d
            visited |= nb
            frontier = nb
        if len(self.flow_cache) > 400:
            self.flow_cache.clear()
        self.flow_cache[key] = dist
        return dist


# --------------------------------------------------------------------------- #
def playable(game_map, bush=None) -> bool:
    """Is this layout's extraction good enough to simulate on?

    A few renders segment badly -- Island Invasion's whole interior reads as
    water -- leaving only a thin ring walkable, on which every brawler dies in
    the gas whatever it does. Those say nothing about strategy and are
    skipped: the largest walkable region must cover a third of the arena.
    """
    import cv2
    bush = bush_mask(game_map) if bush is None else bush
    inb = np.asarray(game_map.in_bounds, bool)
    wall = (np.asarray(game_map.occupancy, bool) & ~bush) | ~inb
    free = (~wall & inb).astype(np.uint8)
    n, lab = cv2.connectedComponents(free, connectivity=8)
    if n <= 1:
        return False
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return sizes.max() / max(1, inb.sum()) >= 0.35 and wall[inb].mean() <= 0.6


def bush_mask(game_map) -> np.ndarray:
    """Bush cells of a reference layout (see perception.mapdb.bush_mask)."""
    from perception.mapdb import bush_mask as _bush_mask
    return _bush_mask(game_map)
