"""
rl/tactics.py
=============

A scripted Showdown strategy: decide WHERE to be, every tick, from everything
the perception stack knows.

WHY A SCRIPT
------------
The policy picks a (heading, distance) and the planner walks there. Until that
policy is trained -- hundreds of thousands of live steps, at ~36k an hour --
the heading is close to random, and random movement loses Showdown in a very
specific way: it walks into the gas, wanders into fights it cannot win, and
never goes near a box. None of those mistakes need learning to avoid. They are
the same kind of fact the planner already encodes for walls and gas: known
in advance, cheap to write down, expensive to rediscover from a reward.

So this module is the strategy a decent human plays by, as utilities:

    gas      the zone is closing on me          get inside it, early
    retreat  hurt, and a threat is near         break line of sight, regen
    heal     hurt, nobody near                  hold still (in a bush) and regen
    engage   a fight I win                      hold my range, strafe, shoot
    collect  a cube on the floor / a box        walk over it / stand in range
    explore  nothing known                      sweep toward the middle

Each tick every mode is scored, the best one wins (with hysteresis so the
agent does not flap between two near-equal options), and the winner produces
an ABSOLUTE destination for `WaypointPlanner.plan_goal`, which routes there
with the same clearance-aware A* and gas escape the policy path uses.

It is also a TEACHER. `goal_to_action` quantises any destination to the
policy's (heading, distance) action, so the same decisions are labels for
behaviour cloning (tools/behaviour_clone.py) and for pretraining in the
simulator (sim/).

WHAT IT KNOWS THAT THE OBSERVATION DOES NOT
-------------------------------------------
Enemy health and cube counts (rl/state.EntityObs), enemy velocity and
remembered positions in bushes (rl/enemy_tracker.py), remembered boxes, the
fused gas field and walls (rl/world_map.py), the arena centre once the
localiser has identified the map, and this brawler's real attack range
(rl/combat.py). The 60-number policy observation carries almost none of it.

EVERYTHING IS A TUNABLE
-----------------------
`TacticsConfig` holds every threshold. The defaults were chosen by running the
simulator (sim/evaluate.py), not by intuition alone; re-run it after changing
them. Per-brawler priors belong in brawlers/profiles.py as overrides.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from math import atan2, cos, hypot, pi, sin
from typing import List, Optional, Tuple

import numpy as np

from .combat import CombatHint

Point = Tuple[float, float]


@dataclass(frozen=True)
class TacticsConfig:
    # Calibration shared with rl/combat.py: tiles across the play rect width.
    tiles_across_screen: float = 17.0
    tick_seconds: float = 0.1

    # --- health thresholds (fraction of the largest HP seen this life) --- #
    retreat_hp: float = 0.42       # below this, a nearby threat means LEAVE
    reengage_hp: float = 0.72      # ...and stay away until back above this
    heal_hp: float = 0.62          # no threat around: stop and regen below this

    # --- engagement ------------------------------------------------------- #
    # Fight when our effective strength (hp x cube damage bonus) is at least
    # this multiple of the target's. Below 1 on purpose: the agent shoots
    # first more often than not, and unknown enemy HP is assumed full.
    engage_ratio: float = 0.85
    # Ranged brawlers hold this fraction of their own reach from the target.
    standoff_frac: float = 0.80
    # ...and close to this when the target is nearly dead (kill-secure).
    finish_frac: float = 0.55
    # Below this range (tiles) a brawler fights by closing, not by kiting.
    closer_range_tiles: float = 4.5
    # Assumed reach of an unknown enemy, tiles. Showdown's median is ~7.
    enemy_range_tiles: float = 7.5
    # Threats further than this are ignored for retreat decisions, tiles.
    threat_radius_tiles: float = 9.0
    # Strafing: offset the standoff point by +/- this angle, flipping every
    # `strafe_ticks` (+ jitter). Projectiles are aimed at where you WERE.
    strafe_deg: float = 38.0
    strafe_ticks: int = 7
    # More than this many enemies within threat radius is a third-party
    # waiting to happen; only a nearly dead target justifies staying.
    max_enemies_engaged: int = 1
    # COMMIT RATHER THAN RUN. A chasing enemy is as fast as you, so turning
    # your back on one already this close (tiles) just means being shot for
    # free. When the lone threat is within this distance and is not clearly
    # healthier than us (hp fraction no more than `commit_hp_margin` above
    # ours), fight it out instead of retreating. 0 disables.
    stand_and_fight_tiles: float = 0.0
    commit_hp_margin: float = 0.15

    # --- gas ---------------------------------------------------------------- #
    # Keep at least this far (tiles) from known gas. The cloud advances
    # ~1 tile per few seconds and the agent needs time to go round walls.
    # Inside `gas_urgent_tiles` the gas outranks everything but a fight for
    # life; out to `gas_margin_tiles` it competes with collecting.
    gas_margin_tiles: float = 5.0
    gas_urgent_tiles: float = 2.5
    # Goals for other modes this close to gas (tiles) are not worth it.
    gas_goal_margin_tiles: float = 1.5
    # Late game: hold near the zone centre when this few players remain.
    late_game_players: int = 4

    # --- collection ----------------------------------------------------------- #
    # Stand this fraction of the reach away from a box while breaking it.
    box_standoff_frac: float = 0.70
    # Ground cubes further than this (tiles) are not worth a detour.
    cube_radius_tiles: float = 12.0
    # A box/cube goal that has been blocked this many ticks is abandoned for a
    # while: routing at an unreachable box forever is a classic stall.
    give_up_ticks: int = 40
    blacklist_ticks: int = 300

    # --- exploration ----------------------------------------------------------- #
    explore_hold_ticks: int = 30
    explore_reach_frac: float = 0.45   # of the short side

    # --- mode selection ------------------------------------------------------ #
    hysteresis: float = 0.15
    seed: Optional[int] = None


@dataclass
class Goal:
    """One tick's decision for the planner and the gun."""
    point: Optional[Point]               # frame px; None = hold position
    mode: str = "explore"
    reason: str = ""
    avoid: List[Point] = field(default_factory=list)
    precise: bool = False
    # Let the planner pull a closed-in destination back toward open ground.
    # Off for gas/zone goals: walls between the player and the safe zone are
    # exactly when the pull-back collapsed the goal onto the player's own
    # position, and he stood still while the cloud arrived (measured in sim/).
    pull_in: bool = True
    hint: CombatHint = field(default_factory=CombatHint)
    target_id: Optional[int] = None

    def label(self) -> str:
        return f"{self.mode}: {self.reason}" if self.reason else self.mode


class TacticalBrain:
    """Utility-based movement for Solo Showdown. One instance per match."""

    MODES = ("gas", "retreat", "heal", "engage", "cube", "box", "zone", "explore")

    def __init__(self, config: Optional[TacticsConfig] = None, ability=None,
                 combat=None):
        """`combat` is the session's rl.combat.CombatPolicy, used for this
        brawler's real reach so standoff distances agree with when it fires."""
        self.config = config or TacticsConfig()
        self.ability = ability
        self.combat = combat
        self._rng = random.Random(self.config.seed)
        self.reset()

    def reset(self) -> None:
        self.tick = 0
        self.mode = "explore"
        self._max_hp: Optional[float] = None
        self._retreating = False
        self._strafe_sign = 1.0
        self._strafe_next = 0
        self._explore_heading: Optional[float] = None
        self._explore_until = 0
        self._zone_dir: Optional[Point] = None       # smoothed inward direction
        self._blacklist = {}                          # track id -> until tick
        self._goal_blocked_for = 0
        self._last_target: Optional[int] = None
        self.last_goal: Optional[Goal] = None
        self.last_scores = {}

    # ------------------------------------------------------------------ #
    # geometry
    # ------------------------------------------------------------------ #
    def _geometry(self, state, world):
        w, h = getattr(state, "frame_size", None) or (1280, 720)
        short = float(max(1, min(w, h)))
        span = float(world.rect[2]) if (world is not None and world.ready()
                                        and world.rect) else float(max(w, h))
        across = self.config.tiles_across_screen
        if self.combat is not None and getattr(self.combat, "_tiles_across", None):
            across = self.combat._tiles_across
        return (w, h), short, span / max(1e-6, across)

    def _reach_px(self, frame, span_px, ppt) -> float:
        if self.combat is not None:
            return float(self.combat.range_px(frame, span_px))
        tiles = getattr(self.ability, "gate_range_tiles", None) or 6.0
        return float(tiles) * ppt

    # ------------------------------------------------------------------ #
    def decide(self, state, planner=None, tracker=None) -> Goal:
        cfg = self.config
        self.tick += 1
        world = planner.world if planner is not None else None
        frame, short, ppt = self._geometry(state, world)
        span = frame[0]
        if world is not None and world.ready() and world.rect:
            span = world.rect[2]
        reach = self._reach_px(frame, span, ppt)

        player = getattr(state, "player_pos", None)
        if player is None:
            # No anchor: nothing positional is trustworthy. Keep walking the
            # way we were going so the planner's lost-mode steering has a
            # direction, and let the anchor come back.
            g = Goal(None if self.last_goal is None else self.last_goal.point,
                     mode="lost", reason="no anchor")
            self.last_goal = g
            return g
        px, py = float(player[0]), float(player[1])

        hp = getattr(state, "health", None)
        if hp is not None and hp > 0:
            self._max_hp = max(self._max_hp or hp, float(hp))
        hp_frac = (float(hp) / self._max_hp) if (hp is not None and self._max_hp) else 1.0
        if self._retreating and hp_frac >= cfg.reengage_hp:
            self._retreating = False

        tracks = tracker.enemies() if tracker is not None else []
        tick_now = tracker.tick if tracker is not None else 0
        enemies = self._enemy_views(state, tracker, px, py)
        threat_r = cfg.threat_radius_tiles * ppt
        threats = [e for e in enemies if e["dist"] <= threat_r]

        self._gas_dist = self._gas_distance_field(world)
        zone = self._zone_center(world, tracker, px, py)
        gas_d = self._gas_distance_tiles(planner, world, px, py, ppt)
        in_gas = bool(getattr(state, "in_gas", False))
        players_left = getattr(state, "players_left", None) or 10

        if self._strafe_next <= self.tick:
            self._strafe_sign = -self._strafe_sign
            self._strafe_next = self.tick + cfg.strafe_ticks + self._rng.randint(0, 4)

        # ---------------- score every mode ---------------- #
        cand = {}

        # GAS: inside it, or about to be.
        if in_gas:
            cand["gas"] = (10.0, self._gas_goal(world, px, py, zone, short, ppt),
                           "in the gas")
        elif gas_d is not None and gas_d < cfg.gas_urgent_tiles:
            cand["gas"] = (4.0 + (cfg.gas_urgent_tiles - gas_d),
                           self._gas_goal(world, px, py, zone, short, ppt),
                           f"gas {gas_d:.1f}t away")
        elif gas_d is not None and gas_d < cfg.gas_margin_tiles:
            cand["gas"] = (1.3 + 0.35 * (cfg.gas_margin_tiles - gas_d),
                           self._gas_goal(world, px, py, zone, short, ppt),
                           f"gas {gas_d:.1f}t away")

        # Fight assessment against the best visible target.
        target, fight_ok, why = self._pick_target(state, enemies, hp_frac, ppt, reach)

        # Cornered: running from a close, no-stronger enemy only gets you
        # shot in the back. Treat it as a fight (see stand_and_fight_tiles).
        if (target is not None and cfg.stand_and_fight_tiles > 0
                and len(threats) <= 1
                and target["dist"] <= cfg.stand_and_fight_tiles * ppt):
            t_frac = target["hp_frac"] if target["hp_frac"] is not None else 1.0
            if t_frac <= hp_frac + cfg.commit_hp_margin:
                fight_ok, why = True, f"cornered ({t_frac:.0%} vs {hp_frac:.0%})"
                self._retreating = False

        # RETREAT: hurt with threats near, or a fight we would lose.
        if threats and not fight_ok and (hp_frac < cfg.retreat_hp or self._retreating):
            self._retreating = True
            pt = self._retreat_point(world, planner, px, py, threats, zone, short, ppt)
            cand["retreat"] = (2.4 + (cfg.reengage_hp - hp_frac), pt,
                               f"hp {hp_frac:.0%}, {len(threats)} threat(s)")
        elif target is not None and not fight_ok and target["dist"] <= (
                cfg.enemy_range_tiles + 1.5) * ppt:
            pt = self._retreat_point(world, planner, px, py, threats or [target],
                                     zone, short, ppt)
            cand["retreat"] = (1.5, pt, f"avoid: {why}")

        # HEAL: hurt, nobody around.
        if not threats and hp_frac < cfg.heal_hp:
            cand["heal"] = (1.1 + (cfg.heal_hp - hp_frac),
                            self._heal_point(world, px, py, ppt), f"hp {hp_frac:.0%}")

        # ENGAGE.
        if target is not None and fight_ok:
            pt = self._engage_point(world, px, py, target, reach, ppt, short, state)
            u = 1.2 + 0.8 * (1.0 - (target["hp_frac"] if target["hp_frac"] is not None else 0.8))
            if target["hurt"]:
                u += 0.3
            u -= 0.03 * target["dist"] / ppt
            cand["engage"] = (u, pt, why)

        # COLLECT: a cube on the floor.
        cube = self._nearest_static(tracker, "cube", px, py,
                                    getattr(state, "ground_cube_positions", ()))
        if cube is not None:
            cpos, cid, cd = cube
            if cd <= cfg.cube_radius_tiles * ppt:
                u = 1.6 - 0.06 * cd / ppt
                if threats and hp_frac < 0.6:
                    u -= 0.6
                u -= self._gas_penalty(world, cpos, ppt)
                cand["cube"] = (u, cpos, f"cube {cd / ppt:.1f}t", cid)

        # COLLECT: a box to break.
        box = self._nearest_static(tracker, "box", px, py,
                                   getattr(state, "box_positions", ()))
        if box is not None:
            bpos, bid, bd = box
            u = 1.0 - 0.035 * bd / ppt
            if threats:
                u -= 0.35
            stand = self._box_point(world, px, py, bpos, reach)
            u -= self._gas_penalty(world, stand, ppt)
            cand["box"] = (u, stand, f"box {bd / ppt:.1f}t", bid)

        # ZONE: late game, hold the middle.
        if zone is not None and players_left <= cfg.late_game_players:
            zx, zy = zone
            dz = hypot(zx - px, zy - py) / ppt
            if dz > 3.0:
                cand["zone"] = (0.7 + 0.03 * dz,
                                self._toward_zone(world, px, py, zone, short, ppt),
                                f"centre {dz:.0f}t")

        # EXPLORE: always available.
        cand["explore"] = (0.3, self._explore_point(world, px, py, zone, short, ppt),
                           "sweep")

        # ---------------- choose with hysteresis ---------------- #
        def score(k):
            v = cand[k][0]
            return v + (cfg.hysteresis if k == self.mode else 0.0)

        best = max(cand, key=score)
        self.last_scores = {k: round(v[0], 2) for k, v in cand.items()}
        entry = cand[best]
        point, reason = entry[1], entry[2]
        tid = entry[3] if len(entry) > 3 else (target["id"] if target and best == "engage" else None)
        self.mode = best

        # Give up on goals that never get closer (unreachable box / cube).
        st = planner.status() if planner is not None else None
        if best in ("box", "cube") and st is not None and st.blocked:
            self._goal_blocked_for += 1
            if self._goal_blocked_for >= cfg.give_up_ticks and tid is not None:
                self._blacklist[tid] = self.tick + cfg.blacklist_ticks
                self._goal_blocked_for = 0
        else:
            self._goal_blocked_for = 0

        avoid = [e["pos"] for e in threats
                 if not (best == "engage" and target is not None and e["id"] == target["id"])]
        hint = CombatHint()
        if best == "heal" and not enemies:
            hint = CombatHint(hold_fire=True, allow_boxes=False)
        elif best == "retreat":
            hint = CombatHint(hold_fire=False, allow_boxes=False)

        g = Goal(point=point, mode=best, reason=reason, avoid=avoid,
                 precise=(best == "cube"), hint=hint, target_id=tid,
                 pull_in=best not in ("gas", "zone"))
        self.last_goal = g
        return g

    # ------------------------------------------------------------------ #
    # enemies
    # ------------------------------------------------------------------ #
    def _enemy_views(self, state, tracker, px, py) -> list:
        """Enemies as dicts: pos (screen px), dist, hp, hp_frac, cubes, hurt, id."""
        out = []
        if tracker is not None and tracker.enemies():
            for t in tracker.enemies():
                sx, sy = tracker.screen_pos(t)
                out.append({"pos": (sx, sy), "dist": hypot(sx - px, sy - py),
                            "hp": t.hp, "hp_frac": t.hp_frac(), "cubes": t.cubes,
                            "hurt": tracker.in_fight(t), "id": t.id,
                            "visible": t.age(tracker.tick) <= 6})
            return out
        for i, e in enumerate(state.enemy_list() if hasattr(state, "enemy_list") else []):
            ex, ey = float(e.pos[0]), float(e.pos[1])
            out.append({"pos": (ex, ey), "dist": hypot(ex - px, ey - py),
                        "hp": e.hp, "hp_frac": None, "cubes": e.cubes,
                        "hurt": False, "id": -1 - i, "visible": True})
        return out

    def _pick_target(self, state, enemies, hp_frac, ppt, reach):
        """Best enemy to fight and whether the fight is worth taking."""
        cfg = self.config
        visible = [e for e in enemies if e["visible"]]
        if not visible:
            return None, False, ""
        our_hp = float(getattr(state, "health", None) or self._max_hp or 4000.0)
        our_cubes = float(getattr(state, "cube_count", None) or 0)
        our_eff = our_hp * (1.0 + 0.1 * our_cubes)

        def strength(e):
            ehp = float(e["hp"]) if e["hp"] is not None else float(self._max_hp or our_hp)
            return ehp * (1.0 + 0.1 * float(e["cubes"] or 0))

        def value(e):
            hpf = e["hp_frac"] if e["hp_frac"] is not None else 0.8
            v = e["dist"] / max(reach, 1.0) + 0.6 * hpf
            if e["hurt"]:
                v -= 0.3
            if e["id"] == self._last_target:
                v -= 0.15                     # stick with a target
            return v

        target = min(visible, key=value)
        self._last_target = target["id"]
        ratio = our_eff / max(1.0, strength(target))
        crowd = sum(1 for e in enemies
                    if e["dist"] <= cfg.threat_radius_tiles * ppt and e["id"] != target["id"])
        nearly_dead = (target["hp_frac"] is not None and target["hp_frac"] < 0.3) or (
            target["hp"] is not None and target["hp"] < 0.25 * our_hp)
        ammo_empty = bool(getattr(state, "ammo_known", False)) and getattr(state, "ammo_count", 1) < 1

        if hp_frac < cfg.retreat_hp and not nearly_dead:
            return target, False, f"too hurt ({hp_frac:.0%})"
        if crowd >= cfg.max_enemies_engaged and not nearly_dead:
            return target, False, f"outnumbered ({crowd + 1})"
        if ratio < cfg.engage_ratio and not nearly_dead:
            return target, False, f"outmatched ({ratio:.2f})"
        if ammo_empty and not nearly_dead:
            return target, False, "reloading"
        return target, True, (f"target {target['dist'] / ppt:.1f}t "
                              f"({'low' if nearly_dead else f'x{ratio:.1f}'})")

    # ------------------------------------------------------------------ #
    # destinations
    # ------------------------------------------------------------------ #
    def _engage_point(self, world, px, py, target, reach, ppt, short, state) -> Point:
        """Hold range from the target, offset sideways to dodge."""
        cfg = self.config
        tx, ty = target["pos"]
        closer = reach < cfg.closer_range_tiles * ppt
        nearly_dead = target["hp_frac"] is not None and target["hp_frac"] < 0.3
        if closer:
            want = 0.45 * reach
        elif nearly_dead:
            want = cfg.finish_frac * reach
        else:
            want = cfg.standoff_frac * reach
        base = atan2(py - ty, px - tx)
        strafe = cfg.strafe_deg * pi / 180.0 * self._strafe_sign
        lobs = bool(getattr(self.ability, "lobs_over_walls", False))
        best, best_s = None, -1e9
        for k, extra in enumerate((0.0, 0.5, -0.5, 1.0, -1.0, 1.6, -1.6)):
            ang = base + strafe + extra
            cx, cy = tx + cos(ang) * want, ty + sin(ang) * want
            s = -abs(extra) * 0.4
            if world is not None and world.ready():
                if not self._standable(world, cx, cy):
                    continue
                s -= 2.0 * self._gas_at(world, cx, cy)
                if not lobs and self._walled(world, (cx, cy), (tx, ty)):
                    s -= 1.0                  # shooters need a clear line
                elif lobs and self._walled(world, (cx, cy), (tx, ty)):
                    s += 0.3                  # throwers love a wall in between
            s -= 0.2 * hypot(cx - px, cy - py) / max(short, 1.0)
            if s > best_s:
                best, best_s = (cx, cy), s
        return best if best is not None else (tx, ty)

    def _retreat_point(self, world, planner, px, py, threats, zone, short, ppt) -> Point:
        """Somewhere far from every threat, behind a wall, out of the gas."""
        best, best_s = None, -1e9
        tpos = [t["pos"] for t in threats]
        for i in range(16):
            ang = 2 * pi * i / 16
            for r in (0.30, 0.55):
                cx, cy = px + cos(ang) * r * short, py + sin(ang) * r * short
                if world is not None and world.ready() and not self._standable(world, cx, cy):
                    continue
                dmin = min(hypot(cx - tx, cy - ty) for tx, ty in tpos)
                s = dmin / ppt
                if world is not None and world.ready():
                    if any(self._walled(world, (cx, cy), t) for t in tpos[:2]):
                        s += 2.5
                    s += 1.5 * self._bush_at(world, cx, cy)
                    s -= 8.0 * self._gas_at(world, cx, cy, dilated=planner)
                    s -= 2.0 * self._edge_risk(world, cx, cy)
                if zone is not None:
                    zx, zy = zone
                    d0 = hypot(zx - px, zy - py)
                    d1 = hypot(zx - cx, zy - cy)
                    s += 0.8 * (d0 - d1) / max(short, 1.0)
                if s > best_s:
                    best, best_s = (cx, cy), s
        if best is None:
            tx, ty = tpos[0]
            ux, uy = _unit(px - tx, py - ty)
            best = (px + ux * 0.4 * short, py + uy * 0.4 * short)
        return best

    def _heal_point(self, world, px, py, ppt) -> Optional[Point]:
        """A nearby bush to regenerate in, or stay put."""
        if world is None or not world.ready():
            return None
        try:
            cx0, cy0 = world.to_cell((px, py))
            r = int(max(2, round(3.0 * ppt / world.cell_size_px())))
            gx0, gy0 = world.clamp_cell((cx0 - r, cy0 - r))
            gx1, gy1 = world.clamp_cell((cx0 + r, cy0 + r))
            bush = world.bush[gy0:gy1 + 1, gx0:gx1 + 1]
            if bush.size and float(bush.max()) > 0.5:
                ys, xs = np.nonzero(bush > 0.5)
                d = (xs + gx0 - cx0) ** 2 + (ys + gy0 - cy0) ** 2
                k = int(np.argmin(d))
                return world.to_pixels((xs[k] + gx0, ys[k] + gy0))
        except Exception:
            pass
        return None

    def _box_point(self, world, px, py, bpos, reach) -> Point:
        """Where to stand to shoot a box: in range, with a clear line."""
        cfg = self.config
        bx, by = bpos
        d = hypot(px - bx, py - by)
        want = cfg.box_standoff_frac * reach
        lobs = bool(getattr(self.ability, "lobs_over_walls", False))
        if d <= reach * 0.95 and (lobs or world is None or not world.ready()
                                  or not self._walled(world, (px, py), (bx, by))):
            return (px, py)                   # already in a firing position
        base = atan2(py - by, px - bx)
        for extra in (0.0, 0.6, -0.6, 1.2, -1.2, 2.0, -2.0, pi):
            ang = base + extra
            cx, cy = bx + cos(ang) * want, by + sin(ang) * want
            if world is None or not world.ready():
                return (cx, cy)
            if not self._standable(world, cx, cy):
                continue
            if lobs or not self._walled(world, (cx, cy), (bx, by)):
                return (cx, cy)
        return (bx, by)

    def _gas_distance_field(self, world):
        """Distance (cells) from every map cell to the nearest known gas, or
        None when no gas has been seen. Computed once per tick."""
        if world is None or not world.ready():
            return None
        try:
            gassy = world.gas > 0.3
            if not gassy.any():
                return None
            import cv2
            return cv2.distanceTransform((~gassy).astype(np.uint8), cv2.DIST_L2, 3)
        except Exception:
            return None

    def _gas_tiles_at(self, world, x, y, ppt) -> Optional[float]:
        d = getattr(self, "_gas_dist", None)
        if d is None or world is None:
            return None
        c = self._cell(world, x, y)
        if c is None:
            return None
        return float(d[c[1], c[0]]) * world.cell_size_px() / max(ppt, 1e-6)

    def _gas_penalty(self, world, pt, ppt) -> float:
        if pt is None:
            return 0.0
        d = self._gas_tiles_at(world, pt[0], pt[1], ppt)
        if d is None:
            return 0.0
        m = self.config.gas_goal_margin_tiles
        return 0.0 if d >= m + 1.5 else (1.5 if d < m else 0.6)

    def _gas_goal(self, world, px, py, zone, short, ppt) -> Point:
        """Where to go to get away from the gas: deep, reachable, inward.

        Candidates on rings around the player, scored by how far from known
        gas they are (capped -- far enough is far enough), how far the open
        run toward them reaches, and agreement with the zone direction. The
        previous "a fixed step toward the zone" collapsed whenever a wall
        stood in that direction.
        """
        if world is None or not world.ready():
            return self._toward_zone(world, px, py, zone, short, ppt)
        za = None
        if zone is not None:
            za = atan2(zone[1] - py, zone[0] - px)
        best, best_s = None, -1e9
        cap = self.config.gas_margin_tiles + 3.0
        for i in range(16):
            a = 2 * pi * i / 16
            run, _unseen, gassy = self._ray_run(world, px, py, a, 1.1 * short)
            for r in (0.45, 0.8, 1.1):
                d = r * short
                x, y = px + cos(a) * d, py + sin(a) * d
                if not self._standable(world, x, y):
                    continue
                gd = self._gas_tiles_at(world, x, y, ppt)
                gd = cap if gd is None else min(gd, cap)
                s = 1.0 * gd
                s += 1.2 * min(run, d) / max(d, 1.0)     # reachable in a line?
                s -= 3.0 * gassy if run < d else 0.0
                if za is not None:
                    s += 2.0 * cos(a - za)
                s -= 0.3 * r                               # nearer is quicker
                if s > best_s:
                    best, best_s = (x, y), s
        return best if best is not None else self._toward_zone(world, px, py, zone, short, ppt)

    def _toward_zone(self, world, px, py, zone, short, ppt) -> Point:
        """A step toward the safe-zone centre, or away from the nearest gas."""
        if zone is not None:
            zx, zy = zone
            d = hypot(zx - px, zy - py)
            if d < 1e-6:
                return (px, py)
            step = min(d, 0.55 * short)
            return (px + (zx - px) / d * step, py + (zy - py) / d * step)
        if world is not None and world.ready():
            away = self._away_from_gas(world, px, py)
            if away is not None:
                return (px + away[0] * 0.55 * short, py + away[1] * 0.55 * short)
        return (px, py)

    def _explore_point(self, world, px, py, zone, short, ppt) -> Point:
        cfg = self.config
        reach = cfg.explore_reach_frac * short
        stale = self._explore_heading is None or self.tick >= self._explore_until
        if not stale and self.last_goal is not None and self.last_goal.mode == "explore" \
                and self.last_goal.point is not None:
            gx, gy = self.last_goal.point
            if hypot(gx - px, gy - py) < 0.12 * short:
                stale = True                      # got there: pick the next leg
        if stale:
            self._explore_heading, run = self._pick_explore_heading(world, px, py, zone, short)
            self._explore_until = self.tick + cfg.explore_hold_ticks
            self._explore_reach = max(0.18 * short, min(reach, 0.9 * run))
        a = self._explore_heading
        r = getattr(self, "_explore_reach", reach)
        return (px + cos(a) * r, py + sin(a) * r)

    def _ray_run(self, world, px, py, a, max_px):
        """Free distance along heading `a` before the first blocked cell, and
        the fraction of that run over never-observed ground."""
        step = 0.5 * world.cell_size_px()
        n = int(max_px / max(step, 1e-6))
        occ = world.occupancy
        unexp = world.unexplored
        oob = world.out_of_bounds
        gas = world.gas
        ca, sa = cos(a), sin(a)
        run, unseen, gassy = 0.0, 0, 0.0
        for k in range(1, n + 1):
            x, y = px + ca * step * k, py + sa * step * k
            c = world.to_cell((x, y))
            if not world.in_bounds(c):
                break
            gx, gy = world.clamp_cell(c)
            if occ[gy, gx] or oob[gy, gx]:
                break
            run = step * k
            if unexp[gy, gx]:
                unseen += 1
            gassy = max(gassy, float(gas[gy, gx]))
        return run, (unseen / float(max(1, k))), gassy

    def _pick_explore_heading(self, world, px, py, zone, short):
        """Heading with the longest open run, toward unexplored ground and the
        zone, away from gas. Returns (heading, free run in px).

        Ray-marched rather than sampled at fixed radii: point samples hop over
        walls and, before the arena is identified, cannot tell unexplored
        arena from the void past its edge -- both read as unseen and free.
        A run that stops after a tile is a wall or the edge, whatever lies
        beyond it.
        """
        prev = self._explore_heading
        best, best_s, best_run = self._rng.uniform(-pi, pi), -1e9, 0.4 * short
        max_px = 1.0 * short
        for i in range(16):
            a = 2 * pi * i / 16
            s = self._rng.uniform(0.0, 0.25)
            run = max_px
            if world is not None and world.ready():
                run, unseen, gassy = self._ray_run(world, px, py, a, max_px)
                s += 1.6 * run / max_px
                s += 0.5 * unseen * (run / max_px)
                s -= 2.5 * gassy
                s -= 0.8 * self._edge_risk(world, px + cos(a) * run, py + sin(a) * run)
            if zone is not None:
                za = atan2(zone[1] - py, zone[0] - px)
                s += 0.6 * cos(a - za)
            if prev is not None:
                s += 0.25 * cos(a - prev)
            if s > best_s:
                best, best_s, best_run = a, s, run
        return best, best_run

    # ------------------------------------------------------------------ #
    # map queries (all fail soft)
    # ------------------------------------------------------------------ #
    def _nearest_static(self, tracker, kind, px, py, fallback):
        best = None
        if tracker is not None and tracker.tracks(kind):
            for t in tracker.tracks(kind):
                if self._blacklist.get(t.id, -1) > self.tick:
                    continue
                sx, sy = tracker.screen_pos(t)
                d = hypot(sx - px, sy - py)
                if best is None or d < best[2]:
                    best = ((sx, sy), t.id, d)
            return best
        for p in fallback or ():
            d = hypot(p[0] - px, p[1] - py)
            if best is None or d < best[2]:
                best = ((float(p[0]), float(p[1])), None, d)
        return best

    @staticmethod
    def _cell(world, x, y):
        c = world.to_cell((x, y))
        if not world.in_bounds(c):
            return None
        return world.clamp_cell(c)

    def _standable(self, world, x, y) -> bool:
        c = self._cell(world, x, y)
        if c is None:
            return True                       # off the map window: unknown
        gx, gy = c
        if world.out_of_bounds[gy, gx]:
            return False
        if world.occupancy[gy, gx]:
            return False
        return float(world.clearance()[gy, gx]) >= 0.6

    def _gas_at(self, world, x, y, dilated=None) -> float:
        c = self._cell(world, x, y)
        if c is None:
            return 0.0
        gx, gy = c
        if dilated is not None:
            try:
                return float(dilated._gas_field()[gy, gx])
            except Exception:
                pass
        return float(world.gas[gy, gx])

    def _bush_at(self, world, x, y) -> float:
        c = self._cell(world, x, y)
        if c is None:
            return 0.0
        return float(world.bush[c[1], c[0]])

    def _walled(self, world, a, b) -> bool:
        from .combat import _walled
        return _walled(world, a, b, 2)

    def _edge_risk(self, world, x, y) -> float:
        """0 in open ground, ->1 against the arena edge (where gas comes first)."""
        c = self._cell(world, x, y)
        if c is None:
            return 0.5
        gx, gy = c
        if world.have_reference:
            oob = world.out_of_bounds
            r = 3
            y0, y1 = max(0, gy - r), min(oob.shape[0], gy + r + 1)
            x0, x1 = max(0, gx - r), min(oob.shape[1], gx + r + 1)
            return float(oob[y0:y1, x0:x1].mean())
        return 0.0

    def _gas_distance_tiles(self, planner, world, px, py, ppt) -> Optional[float]:
        """Distance from the player to the nearest gas (dilated field), tiles."""
        if planner is None or world is None or not world.ready():
            return None
        try:
            field_ = world.gas
            gassy = field_ > 0.3
            if not gassy.any():
                return None
            import cv2
            free = (~gassy).astype(np.uint8)
            dist = cv2.distanceTransform(free, cv2.DIST_L2, 3)
            c = self._cell(world, px, py)
            if c is None:
                return None
            return float(dist[c[1], c[0]]) * world.cell_size_px() / max(ppt, 1e-6)
        except Exception:
            return None

    def _away_from_gas(self, world, px, py) -> Optional[Point]:
        """Unit vector from the gas mass near the player toward clean ground."""
        try:
            c = self._cell(world, px, py)
            if c is None:
                return None
            gx, gy = c
            r = 20
            y0, y1 = max(0, gy - r), min(world.gh, gy + r + 1)
            x0, x1 = max(0, gx - r), min(world.gw, gx + r + 1)
            g = world.gas[y0:y1, x0:x1]
            tot = float(g.sum())
            if tot < 1e-3:
                return None
            ys, xs = np.mgrid[y0:y1, x0:x1]
            mx = float((xs * g).sum() / tot)
            my = float((ys * g).sum() / tot)
            u = _unit(gx - mx, gy - my)
            return u if u != (0.0, 0.0) else None
        except Exception:
            return None

    def _zone_center(self, world, tracker, px, py) -> Optional[Point]:
        """Best estimate of where the safe ground is, as a frame-px point.

        1. The arena centre, once the localiser has identified the map --
           Showdown's cloud closes concentrically on it. Exact.
        2. Otherwise a point half a screen INWARD from the player, along the
           inward normal of the observed gas boundary near him. Showdown's
           cloud is a closing square, so locally its edge is a straight line
           (or a corner) and "away from the boundary, perpendicular to it" is
           the way in -- unlike the gas CENTROID (getGas.safe_vector), which
           for a ring points outward, and unlike "the deepest gas-free cell
           seen", which measured in sim/ flipped between opposite corners
           depending on which strip of floor happened to be on screen.
        The direction is smoothed in the tracker's world frame.
        """
        if world is None or not world.ready():
            return None
        try:
            cc = world.arena_center_cell()
        except Exception:
            cc = None
        if cc is not None:
            return world.to_pixels((cc[0] - 0.5, cc[1] - 0.5))
        d = self._inward_dir(world, px, py)
        if d is not None:
            if self._zone_dir is None:
                self._zone_dir = d
            else:
                a = 0.2
                zx, zy = self._zone_dir
                self._zone_dir = _unit(zx + a * (d[0] - zx), zy + a * (d[1] - zy))
        if self._zone_dir is None:
            return None
        w, h = (world.rect[2], world.rect[3]) if world.rect else (1280, 720)
        step = 0.6 * min(w, h)
        return (px + self._zone_dir[0] * step, py + self._zone_dir[1] * step)

    def _inward_dir(self, world, px, py) -> Optional[Point]:
        """Unit vector (frame px) from the gas boundary toward clean ground."""
        try:
            import cv2
            c = self._cell(world, px, py)
            if c is None:
                return None
            gx, gy = c
            r = 26
            y0, y1 = max(0, gy - r), min(world.gh, gy + r + 1)
            x0, x1 = max(0, gx - r), min(world.gw, gx + r + 1)
            g = np.ascontiguousarray(world.gas[y0:y1, x0:x1], dtype=np.float32)
            if float(g.max()) < 0.3:
                return None
            obs = ~world.gas_unobserved[y0:y1, x0:x1]
            gs = cv2.GaussianBlur(g, (0, 0), 2.0)
            dx = cv2.Sobel(gs, cv2.CV_32F, 1, 0, ksize=3)
            dy = cv2.Sobel(gs, cv2.CV_32F, 0, 1, ksize=3)
            band = obs & (gs > 0.08) & (gs < 0.92)
            if not band.any():
                return None
            ys, xs = np.mgrid[y0:y1, x0:x1]
            dist = np.hypot(xs - gx, ys - gy)
            wgt = band / (1.0 + dist / 8.0)
            vx = -float((wgt * dx).sum())
            vy = -float((wgt * dy).sum())
            cw, ch = world.cell
            return _unit(vx * cw, vy * ch) if (vx or vy) else None
        except Exception:
            return None


# --------------------------------------------------------------------------- #
def goal_to_action(goal_px, player_px, rect, n_headings: int = 16,
                   distances=(0.18, 0.34, 0.55)) -> Tuple[int, int]:
    """Quantise an absolute destination to the policy's (heading, distance).

    The inverse of `WaypointPlanner.target_for`, so the brain's decisions can
    be used as behaviour-cloning labels for the policy. A held position maps
    to the nearest tier in an arbitrary heading; callers that care can skip
    those samples.
    """
    if goal_px is None or player_px is None:
        return (0, 0)
    dx, dy = goal_px[0] - player_px[0], goal_px[1] - player_px[1]
    ang = atan2(dy, dx) % (2 * pi)
    heading = int(round(ang / (2 * pi / n_headings))) % n_headings
    short = max(1.0, min(rect[2], rect[3]))
    d = hypot(dx, dy) / short
    tier = int(np.argmin([abs(d - t) for t in distances]))
    return (heading, tier)


def _unit(x: float, y: float) -> Tuple[float, float]:
    length = hypot(x, y)
    return (0.0, 0.0) if length < 1e-6 else (x / length, y / length)
