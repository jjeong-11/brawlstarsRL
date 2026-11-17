#!/usr/bin/env python3
"""
Fetch map metadata, map renders and the event rotation from BrawlAPI / the
Brawlify CDN, and report which Showdown maps the localiser cannot recognise.

    python tools/fetch_game_data.py maps                 # datasets/brawlify/maps.json
    python tools/fetch_game_data.py maps --renders       # + Showdown map renders
    python tools/fetch_game_data.py events               # current + upcoming rotation
    python tools/fetch_game_data.py coverage             # which Showdown maps lack a layout

WHY
---
`perception/mapdb.py` turns the layout renders in `media/showdownmaps/` into
occupancy grids, and `perception/localize.py` uses them to know the whole map,
border included. A map with no render there is played blind (live terrain
only). Showdown's pool changes every season, so the gap has to be checked,
not remembered. `coverage` compares the active Showdown maps (and, with
`--all`, every Showdown map BrawlAPI knows) against the local renders.

Brawlify's renders are a different drawing from the ones in
`media/showdownmaps/` (no alpha mask around the playfield, different scale),
so they are saved under `datasets/brawlify/maps/` for inspection and are NOT
dropped into the localiser's directory automatically. To add one, look at
`perception.mapdb.inspect_overlay(<png>)` first, then copy it in as
`<Name>-Map.png` and rebuild with `python -m perception.mapdb --build`.

SOURCES AND TERMS
-----------------
api.brawlapi.com (free, no key) and cdn.brawlify.com (MIT repo; "no traffic
limit and no IP blocking"; you must follow Supercell's Terms of Service and
Fan Content Policy). Map art is Supercell's: `datasets/` is git-ignored and
nothing fetched here should be committed or redistributed.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time
import urllib.request
from typing import Dict, Iterable, List

_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = _ROOT / "datasets" / "brawlify"
LOCAL_MAPS = _ROOT / "media" / "showdownmaps"
API = "https://api.brawlapi.com/v1"
UA = {"User-Agent": "brawlstars-rl/1.0"}
SHOWDOWN = ("Solo Showdown", "Duo Showdown", "Trio Showdown", "Showdown")


def _get(url: str, binary: bool = False):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read()
    return data if binary else json.loads(data)


def _list(payload) -> list:
    """BrawlAPI wraps lists as {"list": [...]} (sometimes "items")."""
    if isinstance(payload, list):
        return payload
    return payload.get("list") or payload.get("items") or []


def slug(name: str) -> str:
    """'Rockwall Brawl' / 'Rockwall_Brawl-Map.png' -> 'rockwallbrawl'."""
    name = re.sub(r"-Map(\.png)?$", "", name, flags=re.I)
    return re.sub(r"[^a-z0-9]", "", name.lower())


def mode_name(m: dict) -> str:
    gm = m.get("gameMode") or {}
    return gm.get("name") if isinstance(gm, dict) else str(gm or "")


def showdown_maps(maps: Iterable[dict], include_disabled: bool = False) -> List[dict]:
    out = []
    for m in maps:
        if mode_name(m) not in SHOWDOWN:
            continue
        if not include_disabled and m.get("disabled"):
            continue
        out.append(m)
    return out


def local_layouts(directory=LOCAL_MAPS) -> Dict[str, str]:
    return {slug(p.name): p.name for p in pathlib.Path(directory).glob("*.png")}


def coverage_report(maps: Iterable[dict], local: Dict[str, str]) -> dict:
    have, missing = [], []
    seen = set()
    for m in maps:
        s = slug(m.get("name", ""))
        if not s or s in seen:
            continue
        seen.add(s)
        (have if s in local else missing).append(m.get("name"))
    return {"have": sorted(have), "missing": sorted(missing),
            "share": (len(have) / max(1, len(have) + len(missing)))}


def active_showdown(events_payload: dict) -> List[dict]:
    out = []
    for key in ("active", "upcoming"):
        for ev in events_payload.get(key, []) or []:
            m = ev.get("map") or {}
            if mode_name(m) in SHOWDOWN or (ev.get("slot") or {}).get("name", "").endswith("Showdown"):
                out.append(m)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("maps")
    m.add_argument("--renders", action="store_true", help="download Showdown map renders")
    m.add_argument("--all-modes", action="store_true", help="renders for every mode, not just Showdown")
    sub.add_parser("events")
    c = sub.add_parser("coverage")
    c.add_argument("--all", action="store_true", help="every Showdown map, not just the rotation")
    c.add_argument("--maps-json", help="use a saved maps.json instead of fetching")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    if args.cmd == "maps":
        maps = _list(_get(f"{API}/maps"))
        (OUT / "maps.json").write_text(json.dumps(maps, indent=1))
        print(f"{len(maps)} maps -> {OUT / 'maps.json'}")
        if args.renders:
            todo = maps if args.all_modes else showdown_maps(maps, include_disabled=True)
            (OUT / "maps").mkdir(exist_ok=True)
            got = 0
            for mp in todo:
                url = mp.get("imageUrl")
                dest = OUT / "maps" / f"{mp.get('id')}_{slug(mp.get('name', ''))}.png"
                if not url or dest.exists():
                    continue
                try:
                    dest.write_bytes(_get(url, binary=True))
                    got += 1
                except Exception as e:
                    print(f"  {mp.get('name')}: {e}")
                time.sleep(0.1)
            print(f"{got} renders -> {OUT / 'maps'}")
        return 0

    if args.cmd == "events":
        ev = _get(f"{API}/events")
        (OUT / "events.json").write_text(json.dumps(ev, indent=1))
        for key in ("active", "upcoming"):
            for e in ev.get(key, []) or []:
                mp = e.get("map") or {}
                print(f"{key:9s} {mode_name(mp):16s} {mp.get('name')}")
        return 0

    if args.cmd == "coverage":
        if args.all:
            maps = (json.loads(pathlib.Path(args.maps_json).read_text()) if args.maps_json
                    else _list(_get(f"{API}/maps")))
            maps = showdown_maps(maps, include_disabled=True)
        else:
            maps = active_showdown(_get(f"{API}/events"))
        rep = coverage_report(maps, local_layouts())
        print(f"layouts for {len(rep['have'])}/{len(rep['have']) + len(rep['missing'])} "
              f"Showdown maps ({rep['share']:.0%})")
        for name in rep["missing"]:
            print(f"  missing: {name}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
