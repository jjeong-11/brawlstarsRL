"""
brawlers/abilities.py
=====================

Per-brawler combat facts, and the two decisions the combat script makes from
them: **how far to trust a shot** and **whether to aim it**.

WHY AIM MODE IS DERIVED, NOT LISTED
-----------------------------------
A hand-written list of 106 aim modes is 106 opinions that rot independently.
A rule over two published facts — the brawler's class and its attack range — is
one opinion that can be argued with, tested, and re-applied automatically when
the game rebalances. So `abilities.json` stores only measurements, and this
module derives the mode.

The rule, and the reason for each clause:

    Artillery  -> MANUAL.  The attack arcs over walls, and that is the entire
                  point of the class. Auto-aim will not fire at a target it has
                  no line of sight to, so an auto-aiming thrower throws away
                  the one thing it is for.

    Marksman   -> MANUAL.  One slow, single, high-damage projectile. It has to
                  be led, and auto-aim does not lead — it fires at where the
                  enemy IS. At marksman ranges the travel time is long enough
                  that "where they are" and "where they will be" are different
                  places.

    range >= LONG_RANGE_TILES -> MANUAL, whatever the class. Travel time scales
                  with range, so the lead problem above is really a range
                  problem; the class is a proxy. This clause catches the
                  long-range brawlers that are not marksmen (8-Bit, Byron, R-T,
                  Rico, Leon).

    otherwise  -> AUTO.  Close in, the lead error is smaller than the target,
                  and auto-aim's target selection — nearest enemy in range — is
                  exactly what you would pick anyway. Aiming manually here
                  would be strictly worse: same result, more ways to be wrong.

WHY RANGE GATES THE SHOT
------------------------
`rl/env._gate_weapons` already blocks shots that provably cannot do anything —
no visible target, confirmed-empty clip, uncharged super. "The nearest enemy is
14 tiles away and I am Edgar" belongs in that same category and was missing:
the agent paid `attack_cost` to fire into space. Gating on real range removes
those samples at zero learning cost, which is the same argument the README
makes for handling gas in the planner rather than the reward.

CONFIDENCE IS PART OF THE DATA
------------------------------
Most published ranges are BANDS, not numbers, so `range_tiles` is often a band
midpoint. That is fine for a gate (which needs "roughly how far") and not fine
for tuning (which needs the real value). Each row carries its `confidence` so
callers can tell the difference, and `range_max` is what the gate actually uses
— erring toward letting a marginal shot through rather than blocking a real one.

`tools/sync_brawler_abilities.py` replaces all of it with exact values from the
game's own data files.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional

_HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_ABILITIES = _HERE / "abilities.json"

# At or beyond this, travel time makes auto-aim's lack of lead expensive enough
# that manual aim wins regardless of class. Set at the floor of the published
# top range band; every brawler above it is a dedicated long-range design.
LONG_RANGE_TILES = 9.5

# Classes whose attack must be aimed by hand, and why (kept as data so the
# reason can be shown in the trace rather than inferred from the code).
MANUAL_AIM_CLASSES = {
    "Artillery": "arcs over walls — auto-aim will not fire without line of sight",
    "Marksman": "slow single projectile — has to be led, and auto-aim does not lead",
}

AUTO = "auto"
MANUAL = "manual"


@dataclass(frozen=True)
class Ability:
    """One brawler's combat facts, plus the aim decision derived from them."""

    id: str
    name: str
    brawler_class: str = "Unknown"
    range_tiles: Optional[float] = None
    range_min: Optional[float] = None
    range_max: Optional[float] = None
    lobs_over_walls: bool = False
    confidence: str = "unknown"
    sources: tuple = ()

    # --- the derived decisions ------------------------------------------- #
    @property
    def aim(self) -> str:
        return MANUAL if self.aim_reason else AUTO

    @property
    def aim_reason(self) -> str:
        """Why this brawler aims manually, or "" if it auto-aims.

        Returned as prose rather than a bool because the trace panel shows it,
        and "manual" on its own is exactly the kind of unexplained flag that
        nobody can tell is wrong.
        """
        why = MANUAL_AIM_CLASSES.get(self.brawler_class)
        if why:
            return why
        if self.range_tiles is not None and self.range_tiles >= LONG_RANGE_TILES:
            return (f"{self.range_tiles:g} tiles — at this range the projectile's "
                    f"travel time makes auto-aim's lack of lead costly")
        return ""

    @property
    def gate_range_tiles(self) -> Optional[float]:
        """The distance the attack gate uses. None = do not gate.

        `range_max`, not `range_tiles`: where only a band is published, the true
        value is somewhere inside it, and blocking a shot that would have landed
        is a worse error than allowing one that misses. The miss costs
        `attack_cost` (0.05); the block costs a kill.
        """
        return self.range_max if self.range_max is not None else self.range_tiles

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "class": self.brawler_class,
            "range_tiles": self.range_tiles,
            "range_min": self.range_min, "range_max": self.range_max,
            "lobs_over_walls": self.lobs_over_walls,
            "confidence": self.confidence,
            "aim": self.aim, "aim_reason": self.aim_reason,
            "sources": list(self.sources),
        }


class AbilityBook:
    """All brawlers' combat facts. Cheap to build; reload() re-reads the file."""

    def __init__(self, path: Optional[pathlib.Path] = None) -> None:
        self.path = pathlib.Path(path or DEFAULT_ABILITIES)
        self._by_id: Dict[str, Ability] = {}
        self.meta: dict = {}
        self.reload()

    def reload(self) -> None:
        doc = json.loads(self.path.read_text(encoding="utf-8"))
        self.meta = {k: v for k, v in doc.items() if k.startswith("_")}
        self._by_id = {}
        for e in doc.get("brawlers", []):
            a = Ability(
                id=str(e["id"]), name=str(e.get("name", e["id"])),
                brawler_class=str(e.get("class", "Unknown")),
                range_tiles=e.get("range_tiles"),
                range_min=e.get("range_min"), range_max=e.get("range_max"),
                lobs_over_walls=bool(e.get("lobs_over_walls", False)),
                confidence=str(e.get("confidence", "unknown")),
                sources=tuple(e.get("sources", ())),
            )
            self._by_id[a.id] = a

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, brawler_id: object) -> bool:
        return brawler_id in self._by_id

    def all(self) -> List[Ability]:
        return list(self._by_id.values())

    def get(self, brawler_id: Optional[str]) -> Optional[Ability]:
        """This brawler's abilities, or None.

        None is a legitimate answer, not an error: a session may run without a
        brawler selected, and every consumer must already handle "no per-brawler
        information" by falling back to the shared behaviour.
        """
        return self._by_id.get(brawler_id) if brawler_id else None

    def manual_aimers(self) -> List[Ability]:
        return [a for a in self.all() if a.aim == MANUAL]

    def coverage(self) -> Dict[str, int]:
        from collections import Counter
        return dict(Counter(a.confidence for a in self.all()))


_DEFAULT: Optional[AbilityBook] = None


def default_abilities() -> AbilityBook:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = AbilityBook()
    return _DEFAULT


if __name__ == "__main__":       # python -m brawlers.abilities
    book = default_abilities()
    manual = book.manual_aimers()
    print(f"{len(book)} brawlers from {book.path.name}")
    print(f"confidence: {book.coverage()}")
    print(f"\nMANUAL AIM ({len(manual)}):")
    for a in sorted(manual, key=lambda x: -(x.range_tiles or 0)):
        r = f"{a.range_tiles:>5.2f}" if a.range_tiles is not None else "    ?"
        print(f"  {a.id:<18} {r}t {a.brawler_class:<14} {a.aim_reason}")
    print(f"\nAUTO AIM ({len(book) - len(manual)}): "
          f"{', '.join(sorted(a.id for a in book.all() if a.aim == AUTO))}")
