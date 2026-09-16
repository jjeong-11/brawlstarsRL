#!/usr/bin/env python3
"""
Refresh brawlers/roster.json (and optionally cache portraits) from BrawlAPI.

    python tools/sync_brawlers.py              # roster only, dry run first
    python tools/sync_brawlers.py --write      # actually rewrite roster.json
    python tools/sync_brawlers.py --write --icons   # + cache the portraits

WHY THIS EXISTS
---------------
The roster is not stable data. Supercell ships new brawlers every season, and
rarity tiers get reshuffled wholesale — the Chromatic rarity existed for three
years and then vanished in the January 2024 progression overhaul, with its
brawlers redistributed into Epic, Mythic and Legendary. A hand-written roster is
wrong the moment either happens, and wrong quietly: the UI still renders, it just
groups brawlers under a tier the game no longer has.

So the roster is generated, not authored. Run this after a season update.

WHAT IS PRESERVED
-----------------
`multi_body` is the one field BrawlAPI cannot tell us — it is our own judgement
about whether a brawler's kit puts a second health-barred body on screen, which
matters because `getEnemies` counts those as extra players. Existing flags are
carried across, and brawlers new since the last sync are reported so you can
decide. Everything else is overwritten from the API.

SOURCES
-------
Roster   https://api.brawlapi.com/v1/brawlers   (free, no auth, CORS, cached)
Icons    https://cdn.brawlify.com/brawlers/borderless/<id>.png

Both are Brawlify projects. The CDN is MIT-licensed with no traffic limit and is
explicitly meant to be linked or fetched programmatically. Assets are Supercell's
— use is subject to the Supercell Fan Content Policy, and neither Brawlify nor
this project is affiliated with or endorsed by Supercell.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import urllib.request

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

API = "https://api.brawlapi.com/v1/brawlers"
ROSTER = _ROOT / "brawlers" / "roster.json"
ICON_DIR = _ROOT / "webui" / "static" / "icons"

# Order the UI groups by. Anything the API returns that is not listed here is
# appended alphabetically rather than dropped — a new tier must not silently
# disappear from the picker, which is exactly how the stale Chromatic data
# survived unnoticed.
RARITY_ORDER = ["Starting Brawler", "Rare", "Super Rare", "Epic", "Mythic",
                "Legendary", "Ultra Legendary"]


def slug(name: str) -> str:
    """Roster id: lowercase, ascii, underscore-separated. Stable across syncs
    because it is derived from the name, and `models/<id>/` is keyed on it —
    a slug change would orphan that brawler's trained weights."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower().replace("&", "and")).strip("_")


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "brawlstars-rl/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true",
                    help="rewrite roster.json (default: show the diff and stop)")
    ap.add_argument("--icons", action="store_true",
                    help="also download portraits into webui/static/icons/. "
                         "Optional: the UI falls back to the CDN per-image, so "
                         "this only buys offline use and a faster first paint.")
    ap.add_argument("--force", action="store_true",
                    help="re-download icons that are already cached")
    args = ap.parse_args()

    print(f"fetching {API} …")
    try:
        payload = fetch_json(API)
    except Exception as e:
        print(f"could not reach the roster API: {e}")
        return
    entries = payload.get("list") or []
    if not entries:
        print("the API returned no brawlers — refusing to overwrite the roster")
        return

    old = {}
    if ROSTER.exists():
        doc = json.loads(ROSTER.read_text(encoding="utf-8"))
        old = {b["id"]: b for b in doc.get("brawlers", [])}

    out, unreleased = [], []
    for e in entries:
        name = (e.get("name") or "").strip()
        scid = e.get("id")
        if not name or scid is None:
            continue
        if not e.get("released", True):
            unreleased.append(name)
            continue
        bid = slug(name)
        out.append({
            "id": bid,
            "name": name.upper(),
            "rarity": ((e.get("rarity") or {}).get("name") or "Unknown").strip(),
            "scid": int(scid),
            # Our judgement, not the API's — carried across so a sync never
            # silently clears a flag someone set deliberately.
            "multi_body": bool(old.get(bid, {}).get("multi_body", False)),
        })
    out.sort(key=lambda b: b["scid"])

    rarities = sorted({b["rarity"] for b in out})
    order = [r for r in RARITY_ORDER if r in rarities]
    order += [r for r in rarities if r not in RARITY_ORDER]

    # --- report ---------------------------------------------------------- #
    new_ids = [b["id"] for b in out if b["id"] not in old]
    gone = [i for i in old if i not in {b["id"] for b in out}]
    moved = [(b["id"], old[b["id"]]["rarity"], b["rarity"]) for b in out
             if b["id"] in old and old[b["id"]]["rarity"] != b["rarity"]]

    print(f"\n{len(out)} released brawlers"
          f"{f' (+{len(unreleased)} unreleased skipped: {unreleased})' if unreleased else ''}")
    for r in order:
        print(f"  {r:<18} {sum(1 for b in out if b['rarity'] == r):>3}")
    if r_unknown := [r for r in rarities if r not in RARITY_ORDER]:
        print(f"\n  NOTE: rarity not in RARITY_ORDER, appended last: {r_unknown}")
    print(f"\nnew        : {new_ids or '—'}")
    print(f"removed    : {gone or '—'}")
    print(f"re-rarified: {[f'{i}: {a} -> {b}' for i, a, b in moved] or '—'}")
    if new_ids:
        print("\n  Set `multi_body` by hand for any new brawler whose kit puts a\n"
              "  second health-barred body on screen (summon, clone, split,\n"
              "  mind-control) — the API cannot tell us that.")

    if not args.write:
        print("\ndry run — nothing written. Re-run with --write to apply.")
    else:
        doc = {
            "_comment": (
                "Brawl Stars roster. Every entry maps to models/<id>/ for its own "
                "policy weights. GENERATED by tools/sync_brawlers.py — edit by "
                "hand only for the multi_body flag, which a sync preserves."
            ),
            "_source": API,
            "_fetched": __import__("datetime").date.today().isoformat(),
            "icon_url_template":
                "https://cdn.brawlify.com/brawlers/borderless/{scid}.png",
            "rarity_order": order,
            "brawlers": out,
        }
        ROSTER.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")
        print(f"\nwrote {ROSTER.relative_to(_ROOT)}")

    # --- icons ------------------------------------------------------------ #
    if args.icons:
        ICON_DIR.mkdir(parents=True, exist_ok=True)
        got = skipped = failed = 0
        for b in out:
            dest = ICON_DIR / f"{b['scid']}.png"
            if dest.exists() and not args.force:
                skipped += 1
                continue
            url = f"https://cdn.brawlify.com/brawlers/borderless/{b['scid']}.png"
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "brawlstars-rl/1.0"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    data = r.read()
                if not data.startswith(b"\x89PNG"):
                    raise ValueError("not a PNG")
                dest.write_bytes(data)
                got += 1
                print(f"  {b['id']:<18} {len(data):>7} B")
            except Exception as e:
                failed += 1
                print(f"  {b['id']:<18} FAILED {e}")
        print(f"\nicons: {got} downloaded, {skipped} already cached, {failed} failed"
              f"  -> {ICON_DIR.relative_to(_ROOT)}")
        if failed:
            print("Failures are not fatal: the UI falls back to the CDN URL for "
                  "any portrait it cannot find on disk.")


if __name__ == "__main__":
    main()
