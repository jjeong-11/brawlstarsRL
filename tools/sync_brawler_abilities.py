#!/usr/bin/env python3
"""
Replace the published range estimates in brawlers/abilities.json with the exact
values from the game's own data files.

    python tools/sync_brawler_abilities.py            # dry run: show the diff
    python tools/sync_brawler_abilities.py --write    # apply it

WHY THIS BEATS EVERY WIKI
-------------------------
`abilities.json` ships seeded from published range TIERS — Brawl Planet groups
brawlers into bands ("8.0 - 9.4 tiles") rather than printing a number, so most
rows carry a band midpoint and `confidence: "bucket"`. That is honest and it is
good enough to gate a shot, but it is not a measurement.

The exact numbers exist. BrawlAPI mirrors the game's own CSVs, and the range
lives on the SKILL a character points at:

    csv_logic/characters   Name, WeaponSkill, UltimateSkill, ...
    csv_logic/skills       Name, CastingRange, ...   <- the real number

Joining those two gives ground truth, straight from the files the game itself
reads, with no wiki in the loop. Run this after a balance patch and every
`confidence` becomes "gamefile".

WHAT IT WILL NOT TOUCH
----------------------
`class` and `lobs_over_walls` are left alone. Range is a number the game states;
"is this a thrower" is a classification, cross-checked against three independent
sources when the file was seeded, and not worth re-deriving from a CSV column
whose meaning could change under us. If a sync ever disagrees about range by a
lot for an Artillery brawler, that is worth a look by hand.

Units: the CSVs express range in hundredths of a tile (Supercell's internal
grid), so 1000 -> 10.0 tiles. Sanity-checked below, because a units change is
exactly the kind of silent break that would poison every gate at once.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import urllib.request

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

ABILITIES = _ROOT / "brawlers" / "abilities.json"
CHARACTERS = "https://api.brawlapi.com/game/csv_logic/characters"
SKILLS = "https://api.brawlapi.com/game/csv_logic/skills"

# Column names have moved before. Try each in order and say which one hit.
RANGE_FIELDS = ("CastingRange", "AttackRange", "Range")
# Plausible tile ranges for a Brawl Stars attack. Anything outside means the
# units changed or we joined the wrong column — refuse rather than write it.
MIN_TILES, MAX_TILES = 1.0, 16.0


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "brawlstars-rl/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def pick_range(skill_row: dict):
    for f in RANGE_FIELDS:
        v = skill_row.get(f)
        if isinstance(v, (int, float)) and v > 0:
            return float(v), f
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true",
                    help="apply the update (default: show the diff and stop)")
    ap.add_argument("--tolerance", type=float, default=1.5, metavar="TILES",
                    help="flag rows whose exact value differs from the seeded "
                         "estimate by more than this (default 1.5)")
    args = ap.parse_args()

    doc = json.loads(ABILITIES.read_text(encoding="utf-8"))
    rows = {e["id"]: e for e in doc["brawlers"]}

    print(f"fetching {CHARACTERS} …")
    try:
        characters = fetch_json(CHARACTERS)
        print(f"fetching {SKILLS} …")
        skills = fetch_json(SKILLS)
    except Exception as e:
        print(f"could not reach the game data: {e}\n"
              f"abilities.json is unchanged; the seeded estimates still work.")
        return

    # characters.csv is keyed by internal name ("ShotgunGirl"), and `ItemName`
    # carries the public one ("shelly") -- which is what our roster ids derive
    # from. Match on that, then fall back to a normalised display name.
    def norm(s):
        return "".join(c for c in str(s).lower() if c.isalnum())

    by_norm = {norm(i): i for i in rows}
    by_norm.update({norm(e["name"]): i for i, e in rows.items()})

    updated, unmatched, rejected, big_moves = {}, [], [], []
    field_used = set()
    for internal, row in characters.items():
        if not isinstance(row, dict):
            continue
        weapon = row.get("WeaponSkill")
        item = row.get("ItemName") or ""
        bid = by_norm.get(norm(item)) or by_norm.get(norm(internal))
        if bid is None or not weapon:
            continue
        skill = skills.get(weapon)
        if not isinstance(skill, dict):
            continue
        raw, field = pick_range(skill)
        if raw is None:
            continue
        tiles = round(raw / 100.0, 2)
        if not (MIN_TILES <= tiles <= MAX_TILES):
            rejected.append((bid, raw, tiles))
            continue
        field_used.add(field)
        old = rows[bid].get("range_tiles")
        if old is not None and abs(tiles - old) > args.tolerance:
            big_moves.append((bid, old, tiles))
        updated[bid] = tiles

    unmatched = [i for i in rows if i not in updated]

    print(f"\nmatched {len(updated)}/{len(rows)} brawlers"
          f"  (range column: {sorted(field_used) or 'none'})")
    if rejected:
        print(f"\nREJECTED {len(rejected)} rows outside {MIN_TILES}-{MAX_TILES} "
              f"tiles — the units almost certainly changed:")
        for bid, raw, tiles in rejected[:10]:
            print(f"  {bid:<18} raw {raw} -> {tiles} tiles")
        print("  Not writing those. Check RANGE_FIELDS / the /100 scaling.")
    if unmatched:
        print(f"\nunmatched (keeping the seeded estimate): {unmatched}")
    if big_moves:
        print(f"\nmoved more than {args.tolerance} tiles from the estimate — "
              f"worth an eyeball, this is where a bad join shows up:")
        for bid, old, new in sorted(big_moves, key=lambda t: -abs(t[2] - t[1])):
            print(f"  {bid:<18} {old:>5.2f} -> {new:>5.2f}")

    if not updated:
        print("\nnothing to write.")
        return
    if not args.write:
        print(f"\ndry run — {len(updated)} rows would become exact. "
              f"Re-run with --write to apply.")
        return

    for bid, tiles in updated.items():
        e = rows[bid]
        e["range_tiles"] = e["range_min"] = e["range_max"] = tiles
        e["confidence"] = "gamefile"
        e.pop("range_tier", None)
        srcs = [s for s in e.get("sources", []) if "brawlplanet" not in s
                or e["class"] == "Artillery"]
        if CHARACTERS not in srcs:
            srcs.append(CHARACTERS)
        e["sources"] = srcs
    doc["_fetched"] = __import__("datetime").date.today().isoformat()
    ABILITIES.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    print(f"\nwrote {ABILITIES.relative_to(_ROOT)} — {len(updated)} rows now exact")
    print("Regenerate the reference table:  python tools/dump_brawler_reference.py")


if __name__ == "__main__":
    main()
