#!/usr/bin/env python3
"""
Build a footage manifest from YouTube METADATA: which videos exist, how long
they are, which mode they show, and their licence. Nothing is downloaded.

    export YOUTUBE_API_KEY=...               # Google Cloud, YouTube Data API v3
    python tools/list_footage.py --out datasets/footage_manifest.csv
    python tools/list_footage.py --modes soloShowdown --min-minutes 3 --max-per-channel 300

WHY A MANIFEST AND NOT A DOWNLOADER
-----------------------------------
Phase 2 of docs/PLAN_pro_imitation.md needs hours of first-person play. The
videos belong to their creators, and YouTube's Terms of Service do not allow
downloading them outside YouTube's own features. So the route is: find the
right videos here, then get them with permission -- ask the creator for raw
recordings (unedited, no overlays, is better for us anyway), or record your
own games -- and feed the local files to `tools/fit_from_footage.py`. The
`rights` column starts as "ask" and is for you to fill in.

The manifest is sorted so the most useful candidates come first: seed
priority, then guessed mode, then length (long unedited sessions beat edits).

QUOTA
-----
channels.list, playlistItems.list and videos.list cost 1 unit per call (up to
50 items each); search.list (100 units) is never used. Listing 18 channels
with 300 videos each is ~250 units of the default 10,000/day.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import re
import sys
import urllib.parse
import urllib.request
from typing import Callable, Dict, Iterable, List, Optional

_ROOT = pathlib.Path(__file__).resolve().parent.parent
SEEDS = _ROOT / "tools" / "footage_seeds.json"
API = "https://www.googleapis.com/youtube/v3"

# Title keywords -> mode. Order matters: the first match wins, so the specific
# showdown variants come before the generic word.
MODE_WORDS = [
    ("duoShowdown", r"duo\s*showdown|showdown\s*duo|duo\s*sd\b"),
    ("trioShowdown", r"trio\s*showdown"),
    ("soloShowdown", r"solo\s*showdown|showdown|supervivencia|überleben|sobreviv|\bsd\b"),
    ("duels", r"duel"),
    ("ranked", r"ranked|power\s*league|liga|legendary|masters|\belo\b|clasificatori"),
    ("gemGrab", r"gem\s*grab|atrapagemas|gemas"),
    ("brawlBall", r"brawl\s*ball|balón|fútbol"),
    ("knockout", r"knockout|noqueo"),
    ("heist", r"heist|atraco"),
    ("hotZone", r"hot\s*zone|zona\s*restringida"),
    ("bounty", r"bounty|caza\s*estelar"),
    ("wipeout", r"wipeout"),
]
LIVE_WORDS = r"\blive\b|stream|directo|en\s*vivo|ライブ|配信"


def iso_duration(s: str) -> int:
    """'PT1H2M3S' -> 3723 seconds; '' or 'P0D' -> 0."""
    m = re.fullmatch(r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?", s or "")
    if not m:
        return 0
    d, h, mi, se = (int(x) if x else 0 for x in m.groups())
    return d * 86400 + h * 3600 + mi * 60 + se


def guess_mode(title: str, description: str = "", default: str = "") -> str:
    text = f"{title}\n{description[:300]}".lower()
    for mode, pat in MODE_WORDS:
        if re.search(pat, text):
            return mode
    return default


class YouTube:
    def __init__(self, key: str, opener: Optional[Callable] = None):
        self._key = key
        self._open = opener or self._urlopen
        self.units = 0

    @staticmethod
    def _urlopen(url):
        req = urllib.request.Request(url, headers={"User-Agent": "brawlstars-rl/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    def call(self, endpoint: str, **params) -> dict:
        params["key"] = self._key
        self.units += 1
        return self._open(f"{API}/{endpoint}?{urllib.parse.urlencode(params)}")

    def uploads_playlist(self, handle: str = None, channel_id: str = None) -> Optional[dict]:
        q = {"forHandle": handle} if handle else {"id": channel_id}
        items = self.call("channels", part="snippet,contentDetails", **q).get("items") or []
        if not items:
            return None
        it = items[0]
        return {"channel_id": it["id"], "title": it["snippet"]["title"],
                "uploads": it["contentDetails"]["relatedPlaylists"]["uploads"]}

    def playlist_video_ids(self, playlist_id: str, limit: int) -> List[str]:
        ids, token = [], None
        while len(ids) < limit:
            q = dict(part="contentDetails", playlistId=playlist_id, maxResults=50)
            if token:
                q["pageToken"] = token
            page = self.call("playlistItems", **q)
            ids += [i["contentDetails"]["videoId"] for i in page.get("items", [])]
            token = page.get("nextPageToken")
            if not token:
                break
        return ids[:limit]

    def videos(self, ids: Iterable[str]) -> List[dict]:
        ids = list(dict.fromkeys(ids))
        out = []
        for i in range(0, len(ids), 50):
            page = self.call("videos", part="snippet,contentDetails,status",
                             id=",".join(ids[i:i + 50]))
            out += page.get("items", [])
        return out


def to_row(v: dict, seed: dict) -> dict:
    sn, cd, st = v.get("snippet", {}), v.get("contentDetails", {}), v.get("status", {})
    title = sn.get("title", "")
    modes = seed.get("modes") or []
    default = modes[0] if len(modes) == 1 and modes[0] != "all" else ""
    return {
        "priority": seed.get("priority", 3),
        "mode": guess_mode(title, sn.get("description", ""), default),
        "minutes": round(iso_duration(cd.get("duration", "")) / 60.0, 1),
        "live_archive": bool(re.search(LIVE_WORDS, title.lower())),
        "license": st.get("license", ""),          # "youtube" or "creativeCommon"
        "channel": sn.get("channelTitle", seed.get("name", "")),
        "title": title,
        "published": sn.get("publishedAt", "")[:10],
        "url": f"https://www.youtube.com/watch?v={v.get('id')}",
        "video_id": v.get("id"),
        "rights": "creative-commons: check terms" if st.get("license") == "creativeCommon" else "ask",
    }


def keep(row: dict, modes: Optional[set], min_minutes: float) -> bool:
    if row["minutes"] < min_minutes:
        return False
    return not modes or row["mode"] in modes


def sort_key(row: dict):
    return (row["priority"], row["mode"] != "soloShowdown", not row["live_archive"], -row["minutes"])


def build(yt: YouTube, seeds: dict, max_per_channel: int, log: Callable = print) -> List[dict]:
    rows: Dict[str, dict] = {}

    def add(videos, seed):
        for v in videos:
            r = to_row(v, seed)
            if r["video_id"] not in rows or r["priority"] < rows[r["video_id"]]["priority"]:
                rows[r["video_id"]] = r

    for ch in seeds.get("channels", []):
        info = yt.uploads_playlist(ch.get("handle"), ch.get("id"))
        if info is None:
            log(f"  channel not found: {ch.get('handle') or ch.get('id')} ({ch.get('name')})")
            continue
        ids = yt.playlist_video_ids(info["uploads"], max_per_channel)
        add(yt.videos(ids), {**ch, "name": info["title"]})
        log(f"  {info['title']}: {len(ids)} videos")
    for pl in seeds.get("playlists", []):
        try:
            ids = yt.playlist_video_ids(pl["id"], max_per_channel)
        except Exception as e:                       # deleted / private playlist
            log(f"  playlist {pl['id']}: {e}")
            continue
        add(yt.videos(ids), pl)
        log(f"  playlist {pl.get('name')}: {len(ids)} videos")
    singles = seeds.get("videos", [])
    by_id = {s["id"]: s for s in singles}
    for v in yt.videos(by_id):
        add([v], by_id[v["id"]])
    return list(rows.values())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--seeds", default=str(SEEDS))
    ap.add_argument("--out", default=str(_ROOT / "datasets" / "footage_manifest.csv"))
    ap.add_argument("--modes", default="", help="comma list, e.g. soloShowdown,duoShowdown")
    ap.add_argument("--min-minutes", type=float, default=2.0)
    ap.add_argument("--max-per-channel", type=int, default=200)
    args = ap.parse_args(argv)

    key = os.environ.get("YOUTUBE_API_KEY")
    if not key:
        print("Set YOUTUBE_API_KEY (Google Cloud console, YouTube Data API v3).", file=sys.stderr)
        return 2
    yt = YouTube(key)
    seeds = json.loads(pathlib.Path(args.seeds).read_text())
    rows = build(yt, seeds, args.max_per_channel)
    modes = {m for m in args.modes.split(",") if m} or None
    rows = sorted((r for r in rows if keep(r, modes, args.min_minutes)), key=sort_key)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(to_row({}, {}).keys()))
        w.writeheader()
        w.writerows(rows)
    hours: Dict[str, float] = {}
    for r in rows:
        hours[r["mode"] or "?"] = hours.get(r["mode"] or "?", 0.0) + r["minutes"] / 60.0
    print(f"{len(rows)} videos -> {out}  ({yt.units} quota units)")
    for m, h in sorted(hours.items(), key=lambda kv: -kv[1]):
        print(f"  {m or '?':14s} {h:7.1f} h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
