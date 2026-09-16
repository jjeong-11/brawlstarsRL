#!/usr/bin/env python3
"""
Render docs/BRAWLERS.md from the roster + abilities data.

    python tools/dump_brawler_reference.py

The findings are STORED in brawlers/roster.json and brawlers/abilities.json —
those are what the code reads. This produces the human-readable view of them,
so there is exactly one place the numbers live and the document cannot drift
from the behaviour it describes. Re-run it after a sync.
"""
from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from brawlers import default_abilities, default_registry          # noqa: E402
from brawlers.abilities import LONG_RANGE_TILES, MANUAL_AIM_CLASSES  # noqa: E402

OUT = _ROOT / "docs" / "BRAWLERS.md"

CONF_MARK = {"gamefile": "exact", "measured": "exact*", "bucket": "band",
             "unknown": "—"}


def main() -> None:
    reg, book = default_registry(), default_abilities()
    manual = book.manual_aimers()
    cov = book.coverage()

    L = []
    add = L.append
    add("# Brawler reference\n")
    add("**Generated — do not edit.** `python tools/dump_brawler_reference.py`\n")
    add("Source of truth: [`brawlers/roster.json`](../brawlers/roster.json) "
        "(names, rarity, multi-body) and "
        "[`brawlers/abilities.json`](../brawlers/abilities.json) "
        "(class, range, thrower flag). The code reads those; this file is the "
        "readable view of them, so it cannot drift from what the bot does.\n")

    add("## How aim mode is decided\n")
    add("Aim mode is **derived, never listed**. A hand-written list of "
        f"{len(book)} aim modes is {len(book)} opinions that rot independently; "
        "a rule over two published facts is one opinion you can argue with and "
        "re-apply automatically after a rebalance. See "
        "[`brawlers/abilities.py`](../brawlers/abilities.py).\n")
    add("| Condition | Mode | Why |")
    add("| --- | --- | --- |")
    for cls, why in MANUAL_AIM_CLASSES.items():
        add(f"| class = {cls} | **manual** | {why} |")
    add(f"| range ≥ {LONG_RANGE_TILES} tiles | **manual** | travel time makes "
        f"auto-aim's lack of lead costly, whatever the class |")
    add("| otherwise | **auto** | lead error is smaller than the target, and "
        "auto-aim already picks the nearest enemy in range |")
    add("")
    add(f"That gives **{len(manual)} manual**, "
        f"**{len(book) - len(manual)} auto**.\n")

    add("## Confidence\n")
    add("Most published ranges are *bands*, not numbers, so `range_tiles` is "
        "often a band midpoint. The attack gate uses `range_max` — letting a "
        "marginal shot through costs `attack_cost` (0.05); blocking a real one "
        "costs a kill.\n")
    add("| Level | Meaning | Rows |")
    add("| --- | --- | --- |")
    for k, v in (book.meta.get("_confidence") or {}).items():
        add(f"| `{k}` | {v} | {cov.get(k, 0)} |")
    add("")
    add("Run `python tools/sync_brawler_abilities.py --write` to replace every "
        "estimate with the exact value from the game's own "
        "`characters.csv` + `skills.csv`.\n")

    add("## All brawlers\n")
    add("`aim` is derived. `range` shows the tile value and how it is known: "
        "`exact` from the game files, `exact*` stated by a cross-checked "
        "source, `band` a published range band's midpoint.\n")
    add("| Brawler | Rarity | Class | Range | | Aim | Notes |")
    add("| --- | --- | --- | ---: | --- | --- | --- |")
    for b in reg.all():
        a = book.get(b.id)
        if a is None:
            add(f"| {b.name} | {b.rarity} | — | — | — | — | no ability data |")
            continue
        rng = f"{a.range_tiles:g}" if a.range_tiles is not None else "?"
        if a.confidence == "bucket" and a.range_min is not None:
            rng = f"{a.range_tiles:g} <sub>{a.range_min:g}–{a.range_max:g}</sub>"
        notes = []
        if a.lobs_over_walls:
            notes.append("arcs over walls")
        if b.multi_body:
            notes.append("multi-body — perception unreliable")
        aim = "**manual**" if a.aim == "manual" else "auto"
        add(f"| {b.name} | {b.rarity} | {a.brawler_class} | {rng} | "
            f"{CONF_MARK.get(a.confidence, a.confidence)} | {aim} | "
            f"{', '.join(notes)} |")
    add("")

    add("## Manual aimers, by range\n")
    add("| Brawler | Range | Class | Reason |")
    add("| --- | ---: | --- | --- |")
    for a in sorted(manual, key=lambda x: -(x.range_tiles or 0)):
        r = f"{a.range_tiles:g}" if a.range_tiles is not None else "?"
        add(f"| {a.name} | {r} | {a.brawler_class} | {a.aim_reason} |")
    add("")

    add("## Sources\n")
    for s in book.meta.get("_sources", []):
        add(f"- {s}")
    add(f"\nFetched {book.meta.get('_fetched', '?')}. The Artillery set was "
        "cross-checked against three independent sources before being written; "
        "they agree on exactly eight brawlers.\n")
    add("Not affiliated with, endorsed or sponsored by Supercell.\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT.relative_to(_ROOT)}  "
          f"({len(book)} brawlers, {len(manual)} manual-aim, confidence {cov})")


if __name__ == "__main__":
    main()
