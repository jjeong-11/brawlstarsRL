#!/usr/bin/env python3
"""
Crawl Solo Showdown battle records from the official Brawl Stars API.

    export BRAWLSTARS_API_KEY=...          # from developer.brawlstars.com
    python tools/crawl_battlelogs.py crawl --seed-rankings global --max-requests 2000
    python tools/crawl_battlelogs.py stats --min-trophies 800
    python tools/crawl_battlelogs.py export --out showdown.csv

WHAT IT GIVES YOU
-----------------
The API's battle log is the only bulk source of real Solo Showdown outcomes:
for each battle, the ten players, their brawlers, power and trophies, the map
and -- for the player whose log it is -- their placement and trophy change.
There are no positions, damage or kills, so this is not imitation data. It is:

  * a base rate: how strong players place per brawler, map and trophy band,
    which is what the bot's mean place should be compared against;
  * a selector: which maps and brawlers are worth recording footage for;
  * a calibration target for the simulator's opponents (how often a
    1000-trophy lobby ends with the top four all on high-HP brawlers, ...).

FULL PLACEMENTS FROM ONE-SIDED LOGS
-----------------------------------
Each log records only its owner's placement. But the other nine players have
logs of their own that contain the SAME battle. Battles are keyed by
(battle time, map, sorted player tags), so crawling the participants fills in
their placements on the stored battle, and the more of the lobby is crawled
the closer each battle gets to a complete 1-10 ranking.

LIMITS AND RULES
----------------
  * Keys are bound to the IP address they were created for; a request from any
    other address gets 403. Create the key on the machine that runs this.
  * No rate limit is published. The client spaces requests (`--rps`, default
    5/s), backs off on 429 and 5xx, and stops at `--max-requests` /
    `--max-seconds`. A battle log holds roughly the last 25 battles, so
    re-polling a player sooner than `--recrawl-hours` mostly returns duplicates.
  * Use is governed by Supercell's Terms of Service and the developer portal
    terms; read them before a large crawl. The data is about other players:
    keep the database private and publish only aggregates (`stats`).

The key is read from the environment and never printed or written to disk.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pathlib
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, Dict, Iterable, List, Optional, Tuple

_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_DB = _ROOT / "datasets" / "battlelogs.sqlite"
API_BASE = "https://api.brawlstars.com/v1"
SHOWDOWN_MODES = ("soloShowdown",)


# --------------------------------------------------------------------------- #
# Parsing (pure; tested offline)
# --------------------------------------------------------------------------- #
def normalize_tag(tag: str) -> str:
    """'#2pp', '2PP', '%232PP' -> '#2PP'. Tags use O and 0 interchangeably in
    user input but the API only knows 0."""
    t = urllib.parse.unquote(str(tag)).strip().upper().lstrip("#").replace("O", "0")
    return "#" + t


def _mode(record: dict) -> str:
    return (record.get("battle") or {}).get("mode") or (record.get("event") or {}).get("mode") or ""


def _players(battle: dict) -> List[Tuple[dict, int]]:
    """Every participant with a team index (solo modes: each player is a team)."""
    if battle.get("players"):
        return [(p, i) for i, p in enumerate(battle["players"])]
    out = []
    for ti, team in enumerate(battle.get("teams") or []):
        out.extend((p, ti) for p in team)
    return out


def battle_key(record: dict) -> str:
    """Same battle, same key, whichever participant's log it came from."""
    battle = record.get("battle") or {}
    tags = sorted(normalize_tag(p.get("tag", "")) for p, _ in _players(battle))
    event = record.get("event") or {}
    raw = "|".join([record.get("battleTime", ""), str(event.get("map") or event.get("id") or ""),
                    _mode(record)] + tags)
    return hashlib.sha1(raw.encode()).hexdigest()[:20]


def parse_battle(record: dict, owner_tag: str) -> Optional[Tuple[dict, List[dict]]]:
    """One battle-log entry -> (battle row, participant rows), or None when the
    entry has no players (e.g. some special events). Only the owner's row gets
    `rank` / `trophy_change` / `result`; in duo modes the owner's teammate
    shares the rank."""
    battle = record.get("battle") or {}
    players = _players(battle)
    if not players:
        return None
    owner = normalize_tag(owner_tag)
    event = record.get("event") or {}
    key = battle_key(record)
    row = {
        "id": key,
        "battle_time": record.get("battleTime", ""),
        "mode": _mode(record),
        "map": event.get("map"),
        "event_id": event.get("id"),
        "type": battle.get("type"),
        "duration": battle.get("duration"),
        "n_players": len(players),
    }
    owner_team = next((ti for p, ti in players if normalize_tag(p.get("tag", "")) == owner), None)
    parts = []
    for p, ti in players:
        tag = normalize_tag(p.get("tag", ""))
        brawler = p.get("brawler") or {}
        if not brawler and p.get("brawlers"):          # duels: one entry per brawler used
            brawler = p["brawlers"][0]
        is_owner_side = owner_team is not None and ti == owner_team and row["mode"] != "soloShowdown"
        known = tag == owner or is_owner_side
        parts.append({
            "battle_id": key, "tag": tag, "name": p.get("name"), "team": ti,
            "brawler_id": brawler.get("id"), "brawler": brawler.get("name"),
            "power": brawler.get("power"), "trophies": brawler.get("trophies"),
            "rank": battle.get("rank") if known else None,
            "result": battle.get("result") if known else None,
            "trophy_change": battle.get("trophyChange") if tag == owner else None,
        })
    return row, parts


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #
SCHEMA = """
CREATE TABLE IF NOT EXISTS battles (
  id TEXT PRIMARY KEY, battle_time TEXT, mode TEXT, map TEXT, event_id INTEGER,
  type TEXT, duration INTEGER, n_players INTEGER);
CREATE TABLE IF NOT EXISTS participants (
  battle_id TEXT, tag TEXT, name TEXT, team INTEGER, brawler_id INTEGER, brawler TEXT,
  power INTEGER, trophies INTEGER, rank INTEGER, result TEXT, trophy_change INTEGER,
  PRIMARY KEY (battle_id, tag));
CREATE TABLE IF NOT EXISTS players (
  tag TEXT PRIMARY KEY, priority REAL DEFAULT 0, last_crawled REAL, status TEXT);
CREATE INDEX IF NOT EXISTS idx_battles_mode ON battles(mode, map);
CREATE INDEX IF NOT EXISTS idx_players_todo ON players(last_crawled, priority);
"""


class Store:
    def __init__(self, path):
        path = pathlib.Path(path)
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.executescript(SCHEMA)

    def add(self, row: dict, parts: List[dict]) -> bool:
        """Insert a battle; merge what this log knows (placements) into an
        existing copy. Returns True when the battle is new."""
        cur = self.db.execute("INSERT OR IGNORE INTO battles VALUES (:id,:battle_time,:mode,:map,"
                              ":event_id,:type,:duration,:n_players)", row)
        new = cur.rowcount == 1
        for p in parts:
            self.db.execute("INSERT OR IGNORE INTO participants VALUES (:battle_id,:tag,:name,:team,"
                            ":brawler_id,:brawler,:power,:trophies,:rank,:result,:trophy_change)", p)
            if p["rank"] is not None or p["result"] is not None or p["trophy_change"] is not None:
                self.db.execute(
                    "UPDATE participants SET rank=COALESCE(rank,:rank), result=COALESCE(result,:result),"
                    " trophy_change=COALESCE(trophy_change,:trophy_change)"
                    " WHERE battle_id=:battle_id AND tag=:tag", p)
        return new

    def enqueue(self, tag: str, priority: float = 0.0) -> None:
        self.db.execute("INSERT INTO players(tag, priority) VALUES (?, ?) ON CONFLICT(tag) DO UPDATE"
                        " SET priority=MAX(priority, excluded.priority)", (normalize_tag(tag), priority))

    def next_players(self, n: int, recrawl_s: float, now: float) -> List[str]:
        rows = self.db.execute(
            "SELECT tag FROM players WHERE (status IS NULL OR status != 'gone') AND"
            " (last_crawled IS NULL OR last_crawled < ?) ORDER BY last_crawled IS NOT NULL,"
            " priority DESC LIMIT ?", (now - recrawl_s, n)).fetchall()
        return [r[0] for r in rows]

    def mark(self, tag: str, now: float, status: str = "ok") -> None:
        self.db.execute("UPDATE players SET last_crawled=?, status=? WHERE tag=?", (now, status, tag))

    def commit(self) -> None:
        self.db.commit()


# --------------------------------------------------------------------------- #
# HTTP client
# --------------------------------------------------------------------------- #
class ApiError(RuntimeError):
    def __init__(self, status: int, msg: str):
        super().__init__(f"HTTP {status}: {msg}")
        self.status = status


class Client:
    """Spaced requests, retry on 429 / 5xx with exponential backoff.

    `opener(url, headers) -> (status, body_bytes)` is injectable for tests."""

    def __init__(self, key: str, base: str = API_BASE, rps: float = 5.0,
                 opener: Optional[Callable] = None, sleep: Callable = time.sleep,
                 clock: Callable = time.monotonic, max_retries: int = 5):
        self._key = key
        self.base = base.rstrip("/")
        self.min_gap = 1.0 / max(rps, 1e-3)
        self._open = opener or self._urlopen
        self._sleep = sleep
        self._clock = clock
        self._last = -1e9
        self.max_retries = max_retries
        self.requests = 0

    @staticmethod
    def _urlopen(url, headers):
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def get(self, path: str) -> dict:
        url = self.base + path
        headers = {"Authorization": f"Bearer {self._key}", "Accept": "application/json",
                   "User-Agent": "brawlstars-rl-research/1.0"}
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            wait = self._last + self.min_gap - self._clock()
            if wait > 0:
                self._sleep(wait)
            self._last = self._clock()
            self.requests += 1
            status, body = self._open(url, headers)
            if status == 200:
                return json.loads(body)
            if status in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                self._sleep(delay)
                delay = min(delay * 2, 120.0)
                continue
            try:
                msg = json.loads(body).get("message") or json.loads(body).get("reason") or ""
            except Exception:
                msg = body[:200].decode("utf-8", "replace") if body else ""
            raise ApiError(status, msg)
        raise ApiError(429, "gave up after retries")


def _tag_path(tag: str) -> str:
    return urllib.parse.quote(normalize_tag(tag), safe="")


# --------------------------------------------------------------------------- #
# Crawl
# --------------------------------------------------------------------------- #
def seed(store: Store, client: Client, rankings: Iterable[str], brawler_top: int) -> int:
    n = 0
    for country in rankings:
        for p in client.get(f"/rankings/{country}/players").get("items", []):
            store.enqueue(p["tag"], float(p.get("trophies", 0)) / 100.0)
            n += 1
        if brawler_top > 0:
            for b in client.get("/brawlers").get("items", []):
                items = client.get(f"/rankings/{country}/brawlers/{b['id']}").get("items", [])
                for p in items[:brawler_top]:
                    store.enqueue(p["tag"], float(p.get("trophies", 0)))
                    n += 1
    store.commit()
    return n


def crawl(store: Store, client: Client, modes: Optional[Tuple[str, ...]], max_requests: int,
          max_seconds: float, recrawl_hours: float, expand: bool = True,
          log: Callable = print, now: Callable = time.time) -> Dict[str, int]:
    t0 = now()
    stats = {"players": 0, "battles_new": 0, "battles_seen": 0, "errors": 0}
    while client.requests < max_requests and now() - t0 < max_seconds:
        batch = store.next_players(50, recrawl_hours * 3600.0, now())
        if not batch:
            log("frontier empty")
            break
        for tag in batch:
            if client.requests >= max_requests or now() - t0 >= max_seconds:
                break
            try:
                items = client.get(f"/players/{_tag_path(tag)}/battlelog").get("items", [])
            except ApiError as e:
                if e.status == 403:
                    raise SystemExit("403 from the API: the key is missing, wrong, or bound to "
                                     "another IP address. Create a key for this machine's IP.")
                store.mark(tag, now(), "gone" if e.status == 404 else "error")
                stats["errors"] += 1
                continue
            stats["players"] += 1
            for rec in items:
                if modes and _mode(rec) not in modes:
                    continue
                parsed = parse_battle(rec, tag)
                if parsed is None:
                    continue
                row, parts = parsed
                stats["battles_seen"] += 1
                stats["battles_new"] += store.add(row, parts)
                if expand:
                    for p in parts:
                        if p["tag"] != tag:
                            # Brawler trophies stand in for skill: crawl strong lobbies first.
                            store.enqueue(p["tag"], float(p["trophies"] or 0))
            store.mark(tag, now())
        store.commit()
        log(f"{client.requests} requests, {stats['players']} players, "
            f"{stats['battles_new']} new battles ({stats['battles_seen']} seen)")
    store.commit()
    return stats


# --------------------------------------------------------------------------- #
# Aggregates
# --------------------------------------------------------------------------- #
def placement_stats(store: Store, mode: str = "soloShowdown", by: str = "brawler",
                    min_trophies: int = 0, max_trophies: int = 10 ** 9,
                    min_n: int = 20) -> List[dict]:
    """Mean placement, win and top-4 rate per brawler (or map), over rows with a
    known placement. Each (battle, player) counts once."""
    col = {"brawler": "p.brawler", "map": "b.map", "brawler_map": "p.brawler || ' @ ' || b.map"}[by]
    rows = store.db.execute(
        f"SELECT {col} AS k, COUNT(*), AVG(p.rank), AVG(p.rank = 1), AVG(p.rank <= 4),"
        " AVG(p.trophies) FROM participants p JOIN battles b ON b.id = p.battle_id"
        " WHERE b.mode = ? AND p.rank IS NOT NULL AND p.trophies BETWEEN ? AND ?"
        " GROUP BY k HAVING COUNT(*) >= ? ORDER BY AVG(p.rank)",
        (mode, min_trophies, max_trophies, min_n)).fetchall()
    return [{"key": k, "n": n, "mean_rank": r, "win_rate": w, "top4_rate": t4, "avg_trophies": tr}
            for k, n, r, w, t4, tr in rows]


def coverage(store: Store, mode: str = "soloShowdown") -> dict:
    q = store.db.execute
    n_b = q("SELECT COUNT(*) FROM battles WHERE mode=?", (mode,)).fetchone()[0]
    known = q("SELECT COUNT(*) FROM participants p JOIN battles b ON b.id=p.battle_id"
              " WHERE b.mode=? AND p.rank IS NOT NULL", (mode,)).fetchone()[0]
    full = q("SELECT COUNT(*) FROM (SELECT b.id FROM battles b JOIN participants p ON b.id=p.battle_id"
             " WHERE b.mode=? GROUP BY b.id HAVING SUM(p.rank IS NOT NULL) = COUNT(*))",
             (mode,)).fetchone()[0]
    return {"battles": n_b, "known_placements": known, "fully_ranked_battles": full,
            "players_crawled": q("SELECT COUNT(*) FROM players WHERE last_crawled IS NOT NULL").fetchone()[0],
            "frontier": q("SELECT COUNT(*) FROM players WHERE last_crawled IS NULL").fetchone()[0]}


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=str(DEFAULT_DB))
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("crawl", help="seed and/or crawl battle logs")
    c.add_argument("--tags", nargs="*", default=[], help="player tags to start from")
    c.add_argument("--seed-rankings", nargs="*", default=[],
                   help="'global' and/or 2-letter country codes: seed from the top players")
    c.add_argument("--seed-brawler-top", type=int, default=0,
                   help="also seed the top N players of every brawler (one request per brawler)")
    c.add_argument("--modes", default=",".join(SHOWDOWN_MODES),
                   help="comma list of battle modes to keep; 'all' keeps every mode")
    c.add_argument("--max-requests", type=int, default=2000)
    c.add_argument("--max-seconds", type=float, default=1800)
    c.add_argument("--recrawl-hours", type=float, default=6.0)
    c.add_argument("--rps", type=float, default=5.0)
    c.add_argument("--no-expand", action="store_true", help="do not queue the other participants")
    c.add_argument("--base", default=os.environ.get("BRAWLSTARS_API_BASE", API_BASE),
                   help="API base URL (a proxy that forwards to the official API, if your IP changes)")

    s = sub.add_parser("stats", help="placement aggregates (safe to share)")
    s.add_argument("--by", choices=["brawler", "map", "brawler_map"], default="brawler")
    s.add_argument("--mode", default="soloShowdown")
    s.add_argument("--min-trophies", type=int, default=0)
    s.add_argument("--max-trophies", type=int, default=10 ** 9)
    s.add_argument("--min-n", type=int, default=20)
    s.add_argument("--json", action="store_true")

    e = sub.add_parser("export", help="participant rows as CSV (contains player tags: keep private)")
    e.add_argument("--out", required=True)
    e.add_argument("--mode", default="soloShowdown")

    args = ap.parse_args(argv)
    store = Store(args.db)

    if args.cmd == "crawl":
        key = os.environ.get("BRAWLSTARS_API_KEY")
        if not key:
            print("Set BRAWLSTARS_API_KEY (developer.brawlstars.com; the key is bound to this "
                  "machine's public IP).", file=sys.stderr)
            return 2
        client = Client(key, base=args.base, rps=args.rps)
        for t in args.tags:
            store.enqueue(t, 1e9)
        if args.seed_rankings:
            print(f"seeded {seed(store, client, args.seed_rankings, args.seed_brawler_top)} players")
        modes = None if args.modes == "all" else tuple(m for m in args.modes.split(",") if m)
        st = crawl(store, client, modes, args.max_requests, args.max_seconds,
                   args.recrawl_hours, expand=not args.no_expand)
        print(json.dumps({**st, **coverage(store, (modes or SHOWDOWN_MODES)[0])}, indent=2))
        return 0

    if args.cmd == "stats":
        rows = placement_stats(store, args.mode, args.by, args.min_trophies, args.max_trophies,
                               args.min_n)
        if args.json:
            print(json.dumps({"coverage": coverage(store, args.mode), "rows": rows}, indent=2))
        else:
            print(json.dumps(coverage(store, args.mode)))
            print(f"{'key':32s} {'n':>6s} {'place':>6s} {'win':>6s} {'top4':>6s} {'troph':>6s}")
            for r in rows:
                print(f"{str(r['key'])[:32]:32s} {r['n']:6d} {r['mean_rank']:6.2f} "
                      f"{r['win_rate']:6.1%} {r['top4_rate']:6.1%} {r['avg_trophies'] or 0:6.0f}")
        return 0

    if args.cmd == "export":
        cur = store.db.execute(
            "SELECT b.battle_time, b.map, b.mode, p.* FROM participants p JOIN battles b"
            " ON b.id = p.battle_id WHERE b.mode = ? ORDER BY b.battle_time", (args.mode,))
        cols = [d[0] for d in cur.description]
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerows(cur)
        print(f"wrote {args.out}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
