"""
rl/state.py
===========

The seam between perception and learning.

`liveLoop.LivePerception.tick()` already produces a rich, temporally-smoothed
"live state" dict every frame. This module defines the typed :class:`GameState`
the reward engine and observation encoder consume, and :func:`adapt_live_state`
which maps that live dict onto it.

Live dict (produced by perception/liveLoop.py) -> GameState field:

    live["hp"]                          -> health
    live["hud_cubes"]                   -> cube_count
    live["ammo"]                        -> ammo_count
    live["anchor"] (x, y, r)            -> player_pos (x, y)
    live["enemies"][i]["center"]        -> enemy_positions
    live["boxes"] / live["cubes"]       -> n_boxes / n_ground_cubes
    live["boxes"][i]["center"]          -> box_positions
    live["cubes"][i]["center"]          -> ground_cube_positions
    live["in_gas"]                      -> in_gas
    live["game_state"]["brawlers_left"] -> players_left
    live["game_state"]["state"]/"rank"  -> is_alive / match_over / won

Two reward inputs have no perception source yet and default safe:
    super_charge   (blue super ring)   -> None  [NEEDS EXTRACTOR]
    kills_this_tick(kill attribution)  -> 0     [NEEDS EXTRACTOR]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

Point = Tuple[int, int]


@dataclass
class GameState:
    """A normalized snapshot of one game tick."""

    tick: int = 0

    # --- from the perception layer (liveLoop) ---
    health: Optional[int] = None
    cube_count: Optional[int] = None
    ammo_count: int = 0
    # Whether ammo_count came from an actual read this tick. On real footage the
    # ammo bar is only located on a minority of frames, and a failed read also
    # reports 0 — so "ammo_count == 0" alone must never be taken to mean the
    # clip is empty. Anything gating on empty must check this first.
    ammo_known: bool = False
    player_pos: Optional[Point] = None
    # How player_pos was obtained this tick: "digits" (the reliable digit-first
    # path), "ring" / "ring_unverified" (the weak fallback), "none". Diagnostic
    # only -- nothing learns from it -- but without it a WRONG anchor and a
    # MISSING anchor are indistinguishable in the logs, and they are different
    # bugs. See perception/getAnchor.AnchorResult.
    anchor_source: str = "none"
    anchor_fresh: bool = False
    enemy_positions: List[Point] = field(default_factory=list)
    n_boxes: int = 0
    n_ground_cubes: int = 0
    box_positions: List[Point] = field(default_factory=list)
    ground_cube_positions: List[Point] = field(default_factory=list)
    in_gas: bool = False
    gas_sides: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    gas_safe: Tuple[float, float] = (0.0, 0.0)
    players_left: Optional[int] = None

    # Resolution the pixel coordinates above are expressed in. Carried on the
    # state so anything consuming it (notably the reward engine's distance
    # shaping) can normalise without having to be told the capture size
    # separately. The env overwrites this from the first real frame.
    frame_size: Tuple[int, int] = (1280, 720)

    # --- episode / terminal flags ---
    is_alive: bool = True
    match_over: bool = False
    won: bool = False
    final_rank: Optional[int] = None

    # --- signals still needing a detector (safe defaults) ---
    super_charge: Optional[float] = None   # NEEDS EXTRACTOR (super ring 0..1)
    kills_this_tick: int = 0               # NEEDS EXTRACTOR (kill attribution)

    def nearest_enemy(self) -> Optional[Point]:
        if not self.enemy_positions or self.player_pos is None:
            return None
        px, py = self.player_pos
        return min(self.enemy_positions,
                   key=lambda e: (e[0] - px) ** 2 + (e[1] - py) ** 2)


def adapt_live_state(live: dict, tick: int = 0,
                     super_charge: Optional[float] = None,
                     kills_this_tick: int = 0,
                     frame_size: Optional[Tuple[int, int]] = None) -> GameState:
    """Map one `LivePerception.tick()` dict to a :class:`GameState`.

    `super_charge` / `kills_this_tick` can be injected once detectors exist.
    """
    gs = live.get("game_state") or {}
    screen = gs.get("state", "unknown")
    rank = gs.get("rank")

    match_over = screen == "match_end"
    won = bool(match_over and rank == 1)
    # Dead if the mid-match death/spectate screen is up ("Defeated" + Exit,
    # detected by getGameState), or any non-first final placement.
    died = screen == "defeated" or bool(
        match_over and rank is not None and rank >= 2)

    anchor = live.get("anchor")
    player_pos = (int(anchor[0]), int(anchor[1])) if anchor else None

    def _centers(items) -> List[Point]:
        out: List[Point] = []
        for it in items or []:
            c = it.get("center") if isinstance(it, dict) else it
            if c is not None:
                out.append((int(c[0]), int(c[1])))
        return out

    enemy_positions = _centers(live.get("enemies"))
    box_positions = _centers(live.get("boxes"))
    ground_cube_positions = _centers(live.get("cubes"))

    # super charge comes from perception (getSuper via liveLoop); an explicit
    # arg overrides it (e.g. a custom detector wired into the env).
    sc = super_charge if super_charge is not None else live.get("super_charge")

    return GameState(
        tick=tick,
        health=live.get("hp"),
        cube_count=live.get("hud_cubes"),
        ammo_count=int(live.get("ammo") or 0),
        ammo_known=bool(live.get("ammo_known")),
        player_pos=player_pos,
        anchor_source=str(live.get("anchor_source") or "none"),
        anchor_fresh=bool(live.get("anchor_fresh")),
        enemy_positions=enemy_positions,
        n_boxes=len(live.get("boxes") or []),
        n_ground_cubes=len(live.get("cubes") or []),
        box_positions=box_positions,
        ground_cube_positions=ground_cube_positions,
        in_gas=bool(live.get("in_gas")),
        gas_sides=tuple(live.get("gas_sides") or (0.0, 0.0, 0.0, 0.0)),
        gas_safe=tuple(live.get("gas_safe") or (0.0, 0.0)),
        players_left=gs.get("brawlers_left"),
        is_alive=not died,
        match_over=match_over,
        won=won,
        final_rank=rank,
        super_charge=sc,
        kills_this_tick=kills_this_tick,
        frame_size=frame_size or (1280, 720),
    )


# The explicit gap list, surfaced in the README and asserted in tests.
# Implemented since: super_charge (perception/getSuper.py), kills_this_tick
# (rl/kills.KillAttributor, heuristic), mid_match_death (getGameState
# "defeated" screen, calibrated from media/fixtures/defeated.png), and gas direction
# (getGas.gas_info). What remains:
MISSING_EXTRACTORS = {
    "kill_banner": "Exact kill attribution from the on-screen defeat banner (optional; "
                   "rl/kills.py already gives a heuristic attribution without it).",
}
