"""
sim/bots.py
===========

The nine opponents. Scripted, with a per-bot SKILL that sets how well they aim,
how far they lead, how fast they react and how aggressive they are -- so a
lobby has the spread of a real one rather than nine copies of the same player.

They see what a player sees (the same screen-shaped window, no bushes), know
the gas the way a player does (the cloud is drawn on screen; they anticipate
it a few seconds ahead), farm boxes, pick up cubes, fight when they think they
win and run when hurt. They are NOT the agent's code: a bot that shared the
agent's tactics would make every comparison circular.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .arena import Arena, Brawler

VIEW_X, VIEW_Y = 8.5, 3.8          # half-extent of a 20:9 screen, tiles


@dataclass(frozen=True)
class BotSkill:
    aim_noise_deg: float = 8.0
    lead: float = 0.5               # 0 = aim at where you are, 1 = perfect lead
    reaction_ticks: int = 2
    aggression: float = 1.0
    retreat_hp: float = 0.35
    gas_lookahead_s: float = 8.0

    @staticmethod
    def sample(rng: random.Random, lo: float = 0.25, hi: float = 1.0) -> "BotSkill":
        s = rng.uniform(lo, hi)
        return BotSkill(aim_noise_deg=16.0 - 12.0 * s,
                        lead=0.85 * s,
                        reaction_ticks=int(round(3 - 2 * s)),
                        aggression=rng.uniform(0.7, 1.3),
                        retreat_hp=rng.uniform(0.25, 0.45),
                        gas_lookahead_s=rng.uniform(5.0, 11.0))


class Bot:
    def __init__(self, arena: Arena, idx: int, skill: BotSkill, rng: random.Random):
        self.arena = arena
        self.idx = idx
        self.skill = skill
        self.rng = rng
        self._next_think = 0
        self._ticks = 0
        self._plan = ("wander", None, None)    # (mode, goal, target idx)
        self._wander_goal: Optional[Tuple[float, float]] = None
        self._wander_until = 0.0
        self._seen = {}                         # idx -> (x, y, t)
        self._strafe = 1.0
        self._strafe_until = 0.0
        self._box_focus = None
        self._box_since = 0.0
        self._box_ignored = {}

    @property
    def me(self) -> Brawler:
        return self.arena.brawlers[self.idx]

    # ------------------------------------------------------------------ #
    def act(self, dt: float) -> None:
        b = self.me
        if not b.alive:
            return
        a = self.arena
        self._ticks += 1
        if self._ticks >= self._next_think:
            self._plan = self._think()
            self._next_think = self._ticks + max(1, self.skill.reaction_ticks)
        mode, goal, tidx = self._plan

        if a.t >= self._strafe_until:
            self._strafe = -self._strafe
            self._strafe_until = a.t + self.rng.uniform(0.5, 1.3)

        b.move = self._steer(goal) if goal is not None else (0.0, 0.0)

        # Shoot: the enemy if he is in range and in the open, else a box.
        shot = False
        if tidx is not None:
            o = a.brawlers[tidx]
            if o.alive and not a.hidden(o, b.x, b.y):
                d = math.hypot(o.x - b.x, o.y - b.y)
                if d <= b.stats.range_tiles and (b.stats.lob or a.line_clear(b.x, b.y, o.x, o.y)):
                    aim, frac = self._aim_at(o, d)
                    if b.super_charge >= 1.0 and d <= 0.85 * b.stats.range_tiles:
                        a.fire(b, aim, frac, is_super=True)
                    a.fire(b, aim, frac)
                    shot = True
        if not shot and mode in ("box", "wander", "cube"):
            box = self._nearest_box()
            if box is not None:
                d = math.hypot(box.x - b.x, box.y - b.y)
                if d <= b.stats.range_tiles and (b.stats.lob or a.line_clear(b.x, b.y, box.x, box.y)):
                    a.fire(b, (box.x - b.x, box.y - b.y), d / b.stats.range_tiles)

    # ------------------------------------------------------------------ #
    def _visible_enemies(self):
        a, b = self.arena, self.me
        out = []
        for o in a.brawlers:
            if o is b or not o.alive:
                continue
            if abs(o.x - b.x) > VIEW_X or abs(o.y - b.y) > VIEW_Y:
                continue
            if a.hidden(o, b.x, b.y):
                continue
            out.append(o)
            self._seen[o.idx] = (o.x, o.y, a.t)
        return out

    def _nearest_box(self):
        a, b = self.arena, self.me
        best, bd = None, 11.0
        for bx in a.boxes:
            if not bx.alive or self._box_ignored.get(bx.idx, -1.0) > a.t:
                continue
            d = math.hypot(bx.x - b.x, bx.y - b.y)
            if d < bd:
                best, bd = bx, d
        return best

    def _think(self):
        a, b, sk = self.arena, self.me, self.skill
        enemies = self._visible_enemies()
        hpf = b.hp / max(1.0, b.max_hp)

        # Gas: get inside where the zone WILL be.
        half = a.gas_half(a.t + sk.gas_lookahead_s)
        cheb = max(abs(b.x - a.cx), abs(b.y - a.cy))
        if cheb > half - 1.2:
            target = None
            if enemies:
                target = min(enemies, key=lambda o: (o.x - b.x) ** 2 + (o.y - b.y) ** 2).idx
            gx, gy = a.nearest_open(int(a.cx + self.rng.uniform(-1, 1)),
                                    int(a.cy + self.rng.uniform(-1, 1)))
            return ("gas", (gx + 0.5, gy + 0.5), target)

        target = None
        if enemies:
            def val(o):
                d = math.hypot(o.x - b.x, o.y - b.y)
                return d + 4.0 * (o.hp / max(1.0, o.max_hp))
            target = min(enemies, key=val)

        if target is not None:
            d = math.hypot(target.x - b.x, target.y - b.y)
            mine = b.hp * b.dmg_mult * sk.aggression
            theirs = target.hp * target.dmg_mult
            if hpf < sk.retreat_hp and d < 9.0:
                ax, ay = b.x - target.x, b.y - target.y
                n = math.hypot(ax, ay) or 1.0
                gx = b.x + ax / n * 6.0 + (a.cx - b.x) * 0.15
                gy = b.y + ay / n * 6.0 + (a.cy - b.y) * 0.15
                return ("flee", self._clamp_zone(gx, gy), target.idx)
            if mine >= 0.7 * theirs or d <= 0.7 * b.stats.range_tiles:
                want = 0.75 * b.stats.range_tiles if b.stats.range_tiles > 4.5 else 1.0
                ang = math.atan2(b.y - target.y, b.x - target.x) + 0.6 * self._strafe
                gx = target.x + math.cos(ang) * want
                gy = target.y + math.sin(ang) * want
                return ("fight", (gx, gy), target.idx)

        # Cubes on the floor.
        best_c, bcd = None, 9.0
        for c in a.cubes:
            d = math.hypot(c.x - b.x, c.y - b.y)
            if d < bcd:
                best_c, bcd = c, d
        if best_c is not None:
            return ("cube", (best_c.x, best_c.y), target.idx if target else None)

        box = self._nearest_box()
        if box is not None and self._box_ignored.get(box.idx, -1.0) > a.t:
            box = None
        if box is not None:
            if self._box_focus != box.idx:
                self._box_focus, self._box_since = box.idx, a.t
            elif a.t - self._box_since > 9.0:
                # Spent too long on this one (unreachable, or walled in).
                self._box_ignored[box.idx] = a.t + 25.0
                self._box_focus = None
            d = math.hypot(box.x - b.x, box.y - b.y)
            clear = b.stats.lob or a.line_clear(b.x, b.y, box.x, box.y)
            if d <= b.stats.range_tiles * 0.95 and clear:
                return ("box", None, target.idx if target else None)
            # No line from here: walk up to it. Point blank always has one.
            want = 0.7 * b.stats.range_tiles if clear else 1.1
            return ("box", (box.x + (b.x - box.x) / max(d, 1e-6) * want,
                            box.y + (b.y - box.y) / max(d, 1e-6) * want),
                    target.idx if target else None)

        if (self._wander_goal is None or a.t >= self._wander_until or
                math.hypot(self._wander_goal[0] - b.x, self._wander_goal[1] - b.y) < 1.5):
            h = max(2.0, a.gas_half(a.t + 10.0) - 2.0)
            self._wander_goal = (a.cx + self.rng.uniform(-h, h) * 0.8,
                                 a.cy + self.rng.uniform(-h, h) * 0.8)
            self._wander_until = a.t + self.rng.uniform(3.0, 7.0)
        return ("wander", self._wander_goal, target.idx if target else None)

    def _clamp_zone(self, x, y):
        a = self.arena
        h = max(1.0, a.gas_half(a.t + self.skill.gas_lookahead_s) - 1.5)
        return (min(max(x, a.cx - h), a.cx + h), min(max(y, a.cy - h), a.cy + h))

    # ------------------------------------------------------------------ #
    def _aim_at(self, o: Brawler, d: float):
        b, sk = self.me, self.skill
        st = b.stats
        vx, vy = (0.0, 0.0)
        prev = getattr(self, "_aim_prev", {}).get(o.idx)
        now = self.arena.t
        if prev is not None and now - prev[2] > 1e-6:
            vx = (o.x - prev[0]) / (now - prev[2])
            vy = (o.y - prev[1]) / (now - prev[2])
        if not hasattr(self, "_aim_prev"):
            self._aim_prev = {}
        self._aim_prev[o.idx] = (o.x, o.y, now)
        t_hit = st.lob_flight_s if st.lob else d / max(st.proj_speed, 1e-6)
        tx = o.x + vx * t_hit * sk.lead
        ty = o.y + vy * t_hit * sk.lead
        ang = math.atan2(ty - b.y, tx - b.x) + math.radians(self.rng.gauss(0.0, sk.aim_noise_deg))
        dist = math.hypot(tx - b.x, ty - b.y)
        return (math.cos(ang), math.sin(ang)), dist / max(st.range_tiles, 1e-6)

    def _steer(self, goal) -> Tuple[float, float]:
        a, b = self.arena, self.me
        gx, gy = goal
        if a.solid(int(gx), int(gy)):
            ox, oy = a.nearest_open(int(gx), int(gy))
            gx, gy = ox + 0.5, oy + 0.5
        dx, dy = gx - b.x, gy - b.y
        d = math.hypot(dx, dy)
        if d < 0.3:
            return (0.0, 0.0)
        if a.line_clear(b.x, b.y, gx, gy, step=0.5) and a._clear_disc(
                b.x + dx / d * 0.6, b.y + dy / d * 0.6, 0.42):
            return (dx / d, dy / d)
        field = a.flow_field((int(gx), int(gy)))
        cx, cy = int(b.x), int(b.y)
        best, bv = None, field[min(max(cy, 0), a.gh - 1), min(max(cx, 0), a.gw - 1)]
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                if ox == 0 and oy == 0:
                    continue
                nx, ny = cx + ox, cy + oy
                if not (0 <= nx < a.gw and 0 <= ny < a.gh):
                    continue
                v = field[ny, nx]
                if ox and oy and (a.solid(cx + ox, cy) or a.solid(cx, cy + oy)):
                    continue
                if v < bv:
                    best, bv = (nx + 0.5, ny + 0.5), v
        if best is None:
            return (dx / d, dy / d)
        sx, sy = best[0] - b.x, best[1] - b.y
        n = math.hypot(sx, sy)
        return (sx / n, sy / n) if n > 1e-6 else (dx / d, dy / d)
