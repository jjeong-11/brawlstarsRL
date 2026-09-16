"""The per-brawler weights layout, and the observe-mode safety property.

Two things are worth a test here and the rest is plumbing:

1. **Which weights resolve.** A brawler-specific policy and the shared base
   policy produce visually identical sessions, so a silent fallback would be
   invisible until someone concluded Piper's tuned weights were bad when they
   had never been loaded. `slot().source` is the only thing that distinguishes
   them, so it is tested directly.

2. **Observe mode cannot send input.** The claim is structural — no executor
   object is built, so there is nothing to send with — and structural claims
   are exactly the ones that quietly stop being true during a refactor.
"""

import json
import pathlib
import re

import pytest

from brawlers import BrawlerRegistry, apply_overrides, profile_for
from brawlers.profiles import BrawlerProfile


# --- roster ---------------------------------------------------------------- #
def test_roster_loads_and_ids_are_unique():
    reg = BrawlerRegistry()
    ids = [b.id for b in reg.all()]
    scids = [b.scid for b in reg.all()]
    assert len(ids) > 50, "roster looks truncated"
    assert len(ids) == len(set(ids)), "duplicate brawler ids"
    assert len(scids) == len(set(scids)), "duplicate Supercell ids"
    assert "shelly" in reg and "el_primo" in reg
    print(f"roster: {len(ids)} brawlers, "
          f"{len({b.rarity for b in reg.all()})} rarities")


def test_chromatic_rarity_is_gone():
    """Chromatic was retired in the January 2024 progression overhaul and its
    brawlers redistributed into Epic / Mythic / Legendary.

    Worth a test rather than a comment because a stale rarity fails *silently*:
    the picker still renders, it just groups brawlers under a tier the game no
    longer has, and nothing about the running session looks wrong.
    """
    reg = BrawlerRegistry()
    rarities = {b.rarity for b in reg.all()}
    assert "Chromatic" not in rarities, f"stale roster — run tools/sync_brawlers.py"
    # Spot-check three that moved, one into each destination tier.
    assert reg.get("gale").rarity == "Epic"
    assert reg.get("buzz").rarity == "Mythic"
    assert reg.get("surge").rarity == "Legendary"
    print(f"rarities: {sorted(rarities)}")


def test_every_rarity_is_in_the_display_order():
    """A rarity missing from rarity_order would drop those brawlers out of the
    grouped view entirely — which is how the stale Chromatic data survived."""
    reg = BrawlerRegistry()
    ordered = {r for r, _ in reg.grouped()}
    assert {b.rarity for b in reg.all()} == ordered


def test_unknown_brawler_names_the_file_to_edit():
    reg = BrawlerRegistry()
    with pytest.raises(KeyError) as e:
        reg.get("definitely_not_a_brawler")
    assert "roster.json" in str(e.value)


def test_grouped_covers_every_brawler_exactly_once():
    reg = BrawlerRegistry()
    seen = [b.id for _, group in reg.grouped() for b in group]
    assert sorted(seen) == sorted(b.id for b in reg.all())


# --- weights resolution ---------------------------------------------------- #
def _registry(tmp_path, base: bool):
    """A registry over an empty models/ dir, with or without a base policy."""
    base_path = tmp_path / "base_policy.zip"
    if base:
        base_path.write_bytes(b"not really a zip, never loaded in this test")
    return BrawlerRegistry(models_root=tmp_path / "models", base_policy=base_path)


def test_falls_back_to_base_when_untrained(tmp_path):
    slot = _registry(tmp_path, base=True).slot("piper")
    assert slot.source == "base"
    assert not slot.is_brawler_specific
    # The label must not read as though these are Piper's weights.
    assert "not yet tuned" in slot.label
    print(f"untrained -> {slot.label}")


def test_prefers_brawler_weights_when_present(tmp_path):
    reg = _registry(tmp_path, base=True)
    p = reg.slot("piper").policy_path
    p.parent.mkdir(parents=True)
    p.write_bytes(b"piper weights")
    slot = reg.slot("piper")           # no cache to invalidate: re-checks disk
    assert slot.source == "brawler" and slot.is_brawler_specific
    assert slot.resolved == p
    assert reg.trained() == ["piper"]
    print(f"trained   -> {slot.label}")


def test_no_base_and_no_weights_means_random_headings(tmp_path):
    slot = _registry(tmp_path, base=False).slot("piper")
    assert slot.source == "none" and slot.resolved is None
    assert "random headings" in slot.label


def test_base_policy_accepted_with_or_without_zip_suffix(tmp_path):
    """rl/train.py saves via SB3, which appends .zip itself, so callers hold
    the extensionless path. Both spellings must resolve to the same file."""
    (tmp_path / "base_policy.zip").write_bytes(b"x")
    bare = BrawlerRegistry(models_root=tmp_path / "m",
                           base_policy=tmp_path / "base_policy")
    dotted = BrawlerRegistry(models_root=tmp_path / "m",
                             base_policy=tmp_path / "base_policy.zip")
    assert bare.slot("colt").resolved == dotted.slot("colt").resolved


def test_prepare_creates_the_training_layout(tmp_path):
    reg = _registry(tmp_path, base=True)
    slot = reg.prepare("mortis")
    assert slot.checkpoint_dir.is_dir()
    meta = json.loads(slot.meta_path.read_text())
    assert meta["brawler"] == "mortis" and meta["trained"] is False
    # prepare() must not fabricate weights — the slot still resolves to base.
    assert reg.slot("mortis").source == "base"


def test_every_brawler_gets_a_distinct_directory(tmp_path):
    reg = _registry(tmp_path, base=True)
    dirs = {reg.slot(b.id).dir for b in reg.all()}
    assert len(dirs) == len(reg)


# --- portraits ------------------------------------------------------------- #
def test_every_brawler_has_a_portrait_url():
    reg = BrawlerRegistry()
    urls = [reg.icon_url(b.id) for b in reg.all()]
    assert all(u and u.startswith("https://") for u in urls)
    assert len(set(urls)) == len(reg), "two brawlers share a portrait URL"


def test_local_icon_is_preferred_when_cached(tmp_path):
    """The UI serves /api/icon/<id> either way, so a cached install and a
    CDN-only install render from the same markup."""
    reg = BrawlerRegistry(models_root=tmp_path / "m", icon_dir=tmp_path / "icons")
    assert reg.local_icon("shelly") is None          # nothing cached
    (tmp_path / "icons").mkdir()
    (tmp_path / "icons" / f"{reg.get('shelly').scid}.png").write_bytes(b"\x89PNG")
    assert reg.local_icon("shelly") is not None
    assert reg.local_icon("colt") is None            # only shelly was cached


# --- profiles -------------------------------------------------------------- #
def test_profiles_are_empty_so_behaviour_is_unchanged():
    """Every profile is a no-op today. If one stops being empty, whoever filled
    it in should have measured it — this test is the prompt to say so here."""
    from brawlers.profiles import PROFILES
    assert all(p.is_default for p in PROFILES.values())
    assert profile_for("anything_at_all").is_default


def test_apply_overrides_ignores_unknown_keys_instead_of_raising():
    """A profile naming a knob a later refactor renamed must not take a live
    session down mid-match."""
    from dataclasses import dataclass

    @dataclass
    class Cfg:
        a: int = 1
        b: float = 2.0

    out = apply_overrides(Cfg(), {"a": 9, "vanished_knob": 3})
    assert out.a == 9 and out.b == 2.0
    assert apply_overrides(Cfg(), {}) == Cfg()


def test_profile_overrides_flow_through_to_a_config():
    prof = BrawlerProfile(id="x", reward_overrides={"gamma": 0.99})
    from rl.rewards import RewardConfig
    cfg = apply_overrides(RewardConfig(), prof.reward_overrides)
    assert cfg.gamma == 0.99


# --- session config / observe-mode safety ---------------------------------- #
def test_session_config_rejects_nonsense():
    from webui.runner import SessionConfig
    SessionConfig().validate()                                  # default is fine
    with pytest.raises(ValueError):
        SessionConfig(mode="halfway").validate()
    with pytest.raises(ValueError):
        SessionConfig(fps=0).validate()
    with pytest.raises(ValueError):
        SessionConfig(source="video", video_path=None).validate()


def test_observe_mode_builds_no_executor():
    """The safety property, asserted directly.

    Observe mode is not 'an executor that has been told not to fire' — it is no
    executor at all, so there is no object a future edit could accidentally
    call. `_build_executor` is the single place the two modes diverge.
    """
    from webui.runner import SessionConfig, SessionRunner
    runner = SessionRunner()
    assert runner._build_executor(SessionConfig(mode="observe")) is None


def test_runner_starts_idle_and_refuses_unknown_brawlers():
    from webui.runner import SessionConfig, SessionRunner
    runner = SessionRunner()
    s = runner.status()
    assert s["status"] == "idle" and s["running"] is False and s["tick"] == 0
    with pytest.raises(KeyError):
        runner.start(SessionConfig(brawler="not_a_brawler"))
    assert runner.status()["status"] == "idle", "a rejected start must not arm the runner"


# --- web layer ------------------------------------------------------------- #
@pytest.fixture()
def client():
    flask = pytest.importorskip("flask", reason="web UI is optional: pip install flask")
    from webui.app import create_app
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def test_index_inlines_the_whole_roster(client):
    """Selection is a local lookup, so the page must ship the roster with it.

    This is the fix for the picker: it used to fetch /api/brawlers/<id> on every
    change, so one failed request left it looking dead. Now the only thing that
    can fail is the page load itself, which is visible.
    """
    body = client.get("/").get_data(as_text=True)
    assert "Press Start" in body

    m = re.search(r'<script id="rosterData" type="application/json">(.*?)</script>',
                  body, re.S)
    assert m, "roster payload is not inlined in the page"
    payload = json.loads(m.group(1))
    reg = BrawlerRegistry()
    assert len(payload["brawlers"]) == len(reg)
    ids = {b["id"] for b in payload["brawlers"]}
    assert ids == {b.id for b in reg.all()}
    one = next(b for b in payload["brawlers"] if b["id"] == "shelly")
    assert one["icon"] == "/api/icon/shelly"
    assert one["model"]["label"]
    print(f"inlined {len(payload['brawlers'])} brawlers, "
          f"{len(m.group(1))} chars")


def test_inlined_roster_cannot_break_out_of_the_script_tag(client):
    """A literal </ inside the JSON would end the <script> early and take the
    rest of the page with it. Escaped as <\\/, which is still valid JSON."""
    body = client.get("/").get_data(as_text=True)
    block = re.search(r'<script id="rosterData".*?>(.*?)</script>', body, re.S).group(1)
    assert "</" not in block


def test_api_brawlers_reports_the_weights_source(client):
    data = client.get("/api/brawlers").get_json()
    assert len(data["brawlers"]) == len(BrawlerRegistry())
    one = next(b for b in data["brawlers"] if b["id"] == "shelly")
    assert one["model"]["source"] in ("brawler", "base", "none")
    assert "label" in one["model"]


def test_icon_route_redirects_to_the_cdn_when_nothing_is_cached(client):
    r = client.get("/api/icon/shelly")
    # 302 to the CDN without a local cache; 200 with one. Both are correct —
    # what must not happen is a 404 leaving a hole in the grid.
    assert r.status_code in (200, 302)
    if r.status_code == 302:
        assert "cdn.brawlify.com" in r.headers["Location"]


def test_icon_route_404s_on_an_unknown_brawler(client):
    assert client.get("/api/icon/not_a_brawler").status_code == 404


def test_status_and_stop_are_safe_before_any_session(client):
    assert client.get("/api/status").get_json()["running"] is False
    assert client.post("/api/stop").status_code == 200


def test_start_rejects_an_unknown_brawler(client):
    r = client.post("/api/start", json={"brawler": "nope"})
    assert r.status_code == 400 and "error" in r.get_json()


def test_frame_endpoint_is_empty_before_any_session(client):
    assert client.get("/api/frame.jpg").status_code == 204
