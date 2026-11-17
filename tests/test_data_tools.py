"""Offline tests for the data-acquisition tools (no network: fake openers)."""
import json

from tools import crawl_battlelogs as cb
from tools import fetch_game_data as fg
from tools import list_footage as lf


def _sd(owner_rank, tags, time="20261117T101500.000Z", map_name="Skull Creek", owner=None):
    players = [{"tag": t, "name": t[1:], "brawler": {"id": 16000000 + i, "name": f"B{i}",
                                                    "power": 11, "trophies": 900 + i}}
               for i, t in enumerate(tags)]
    return {"battleTime": time, "event": {"id": 15000001, "mode": "soloShowdown", "map": map_name},
            "battle": {"mode": "soloShowdown", "type": "ranked", "rank": owner_rank,
                       "trophyChange": 8, "duration": 120, "players": players}}


TAGS = ["#2PP", "#8QQ", "#9LL", "#YYY", "#GGG", "#RRR", "#JJJ", "#CCC", "#UUU", "#VVV"]


def test_tag_normalisation_and_battle_key_is_owner_independent():
    assert cb.normalize_tag("%232pp") == "#2PP" == cb.normalize_tag("2PP")
    a = _sd(3, TAGS)
    b = _sd(7, list(reversed(TAGS)))
    assert cb.battle_key(a) == cb.battle_key(b)
    assert cb.battle_key(a) != cb.battle_key(_sd(3, TAGS, map_name="Other"))


def test_placements_merge_across_participants_logs():
    st = cb.Store(":memory:")
    for i, owner in enumerate(TAGS):
        row, parts = cb.parse_battle(_sd(i + 1, TAGS), owner)
        assert sum(p["rank"] is not None for p in parts) == 1
        assert st.add(row, parts) == (i == 0)
    cov = cb.coverage(st)
    assert cov["battles"] == 1 and cov["known_placements"] == 10 and cov["fully_ranked_battles"] == 1
    rows = cb.placement_stats(st, by="brawler", min_n=1)
    assert {r["key"]: r["mean_rank"] for r in rows}["B0"] == 1
    assert len(rows) == 10


def test_duo_rank_is_shared_with_teammate_only():
    rec = {"battleTime": "t", "event": {"map": "M"},
           "battle": {"mode": "duoShowdown", "rank": 2,
                      "teams": [[{"tag": "#A", "brawler": {"name": "X"}}, {"tag": "#B", "brawler": {"name": "Y"}}],
                                [{"tag": "#C", "brawler": {"name": "Z"}}, {"tag": "#D", "brawler": {"name": "W"}}]]}}
    _, parts = cb.parse_battle(rec, "#B")
    assert {p["tag"]: p["rank"] for p in parts} == {"#A": 2, "#B": 2, "#C": None, "#D": None}


def test_client_backs_off_on_429_and_crawl_expands_frontier():
    calls, sleeps = [], []
    owner_log = {"items": [_sd(4, TAGS), {"battleTime": "x", "event": {"mode": "gemGrab"},
                                          "battle": {"mode": "gemGrab", "teams": [[{"tag": "#Z"}]]}}]}

    def opener(url, headers):
        calls.append(url)
        assert headers["Authorization"] == "Bearer k"
        if len(calls) == 1:
            return 429, b'{"reason":"throttled"}'
        if "%232PP" in url:
            return 200, json.dumps(owner_log).encode()
        return 404, b'{"reason":"notFound"}'

    t = [0.0]
    client = cb.Client("k", rps=1000, opener=opener, sleep=lambda s: sleeps.append(s),
                       clock=lambda: t[0])
    st = cb.Store(":memory:")
    st.enqueue("#2PP", 1e9)
    stats = cb.crawl(st, client, ("soloShowdown",), max_requests=2, max_seconds=60,
                     recrawl_hours=6, log=lambda *_: None, now=lambda: 0.0)
    assert sleeps and sleeps[0] >= 2.0                      # backed off once
    assert stats["battles_new"] == 1 and stats["players"] == 1
    # the nine other participants were queued, strongest brawler trophies first
    nxt = st.next_players(3, 3600, now=0.0)
    assert nxt[0] == "#VVV" and "#2PP" not in nxt


def test_game_data_coverage_matches_local_layout_names():
    maps = [{"name": "Skull Creek", "gameMode": {"name": "Solo Showdown"}},
            {"name": "Rockwall Brawl", "gameMode": {"name": "Duo Showdown"}},
            {"name": "Hard Rock Mine", "gameMode": {"name": "Gem Grab"}},
            {"name": "Nowhere New", "gameMode": {"name": "Solo Showdown"}, "disabled": True}]
    sd = fg.showdown_maps(maps)
    assert [m["name"] for m in sd] == ["Skull Creek", "Rockwall Brawl"]
    local = fg.local_layouts()
    assert len(local) >= 60 and fg.slug("Rockwall_Brawl-Map.png") == "rockwallbrawl"
    rep = fg.coverage_report(fg.showdown_maps(maps, include_disabled=True), local)
    assert "Nowhere New" in rep["missing"]
    events = {"active": [{"map": {"name": "Skull Creek", "gameMode": {"name": "Solo Showdown"}}},
                         {"map": {"name": "Hard Rock Mine", "gameMode": {"name": "Gem Grab"}}}]}
    assert [m["name"] for m in fg.active_showdown(events)] == ["Skull Creek"]


def test_footage_manifest_parsing_and_ranking():
    assert lf.iso_duration("PT1H2M3S") == 3723 and lf.iso_duration("PT45S") == 45
    assert lf.iso_duration("P0D") == 0 and lf.iso_duration("bogus") == 0
    assert lf.guess_mode("RANK 35 Leon in Duo Showdown") == "duoShowdown"
    assert lf.guess_mode("ASI es SUPERVIVENCIA en 66.000 COPAS") == "soloShowdown"
    assert lf.guess_mode("Dominating Ranked with Control Brawlers!") == "ranked"
    assert lf.guess_mode("random vlog", default="soloShowdown") == "soloShowdown"

    def opener(url):
        if "/channels?" in url:
            return {"items": [{"id": "UC1", "snippet": {"title": "Chan"},
                               "contentDetails": {"relatedPlaylists": {"uploads": "UU1"}}}]}
        if "/playlistItems?" in url:
            return {"items": [{"contentDetails": {"videoId": v}} for v in ("a", "b", "c")]}
        if "/videos?" in url:
            meta = {"a": ("Solo Showdown push LIVE", "PT2H"), "b": ("Gem Grab tips", "PT8M"),
                    "c": ("short", "PT30S"), "z": ("Rank 35 in showdown", "PT12M")}
            ids = url.split("id=")[1].split("&")[0].replace("%2C", ",").split(",")
            return {"items": [{"id": i, "snippet": {"title": meta[i][0], "channelTitle": "Chan"},
                               "contentDetails": {"duration": meta[i][1]},
                               "status": {"license": "youtube"}} for i in ids]}
        raise AssertionError(url)

    yt = lf.YouTube("k", opener=opener)
    seeds = {"channels": [{"handle": "@x", "priority": 2}], "videos": [{"id": "z", "priority": 1}]}
    rows = lf.build(yt, seeds, max_per_channel=10, log=lambda *_: None)
    rows = sorted((r for r in rows if lf.keep(r, None, 2.0)), key=lf.sort_key)
    assert [r["video_id"] for r in rows] == ["z", "a", "b"]
    assert rows[1]["live_archive"] and rows[1]["minutes"] == 120.0 and rows[0]["rights"] == "ask"
    assert yt.units == 4
