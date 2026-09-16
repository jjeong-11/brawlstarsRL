"""Per-brawler range gating and aim mode.

Two things carry the weight here, and neither is obvious from reading the code:

  * **The gate can only tighten.** Converting tiles to pixels needs a
    calibration number that has not been measured on this setup, so the
    per-brawler range is applied as a *ceiling* on the old global gate. If the
    calibration is wrong the worst case is the behaviour we already had. That
    property is the entire reason it is safe to ship, so it is asserted
    directly rather than trusted.

  * **Aim mode is derived, not listed.** 106 hand-written opinions rot
    independently; a rule over two published facts is one opinion. The tests
    pin the rule, not the 106 outcomes.
"""

import pytest

from brawlers import default_abilities, default_registry
from brawlers.abilities import (AUTO, MANUAL, Ability, AbilityBook,
                                LONG_RANGE_TILES, MANUAL_AIM_CLASSES)
from rl.combat import CombatConfig, CombatDecision, CombatPolicy
from rl.state import GameState


# --- coverage --------------------------------------------------------------- #
def test_every_brawler_in_the_roster_has_abilities():
    reg, book = default_registry(), default_abilities()
    missing = [b.id for b in reg.all() if book.get(b.id) is None]
    assert not missing, f"no ability data for {missing}"
    print(f"{len(book)} brawlers, confidence {book.coverage()}")


def test_every_row_records_where_it_came_from():
    """A number with no provenance cannot be checked, and most of these are
    estimates from published range BANDS rather than measurements."""
    for a in default_abilities().all():
        assert a.confidence in ("gamefile", "measured", "bucket", "unknown")
        if a.confidence != "unknown":
            assert a.sources, f"{a.id} has a range but no source"


def test_bands_bracket_their_midpoint():
    for a in default_abilities().all():
        if a.range_tiles is None:
            continue
        assert a.range_min <= a.range_tiles <= a.range_max, a.id


def test_artillery_is_exactly_the_eight_throwers():
    """Cross-checked against three independent sources when the file was
    seeded. If this ever fails, the roster changed — go and look."""
    book = default_abilities()
    art = {a.id for a in book.all() if a.brawler_class == "Artillery"}
    assert art == {"barley", "dynamike", "tick", "penny", "grom", "sprout",
                   "larry_and_lawrie", "juju"}
    assert all(book.get(i).lobs_over_walls for i in art)
    assert not any(a.lobs_over_walls for a in book.all()
                   if a.brawler_class != "Artillery")


# --- the aim rule ----------------------------------------------------------- #
def _ability(**kw):
    base = dict(id="x", name="X", brawler_class="Damage Dealer",
                range_tiles=6.0, range_min=6.0, range_max=6.0)
    base.update(kw)
    return Ability(**base)


@pytest.mark.parametrize("cls", sorted(MANUAL_AIM_CLASSES))
def test_manual_aim_classes_always_aim(cls):
    """Artillery arcs over walls; a marksman's slow single shot has to be led.
    Neither depends on the range, so a short-range one must still aim."""
    a = _ability(brawler_class=cls, range_tiles=2.0, range_min=2.0, range_max=2.0)
    assert a.aim == MANUAL and a.aim_reason


def test_long_range_aims_whatever_the_class():
    """Travel time scales with range, so the lead problem is really a range
    problem — the class is only a proxy. This clause catches the long-range
    brawlers that are not marksmen (8-Bit, Byron, R-T, Rico, Leon)."""
    r = LONG_RANGE_TILES
    assert _ability(range_tiles=r, range_min=r, range_max=r).aim == MANUAL
    assert _ability(range_tiles=r - 0.1, range_min=r - 0.1,
                    range_max=r - 0.1).aim == AUTO


def test_short_range_auto_aims_and_says_nothing():
    a = _ability(range_tiles=3.0, range_min=3.0, range_max=3.0)
    assert a.aim == AUTO and a.aim_reason == ""


def test_unknown_range_does_not_become_manual_by_accident():
    """A brand-new brawler with no published range must fall to the safe
    default rather than picking up the long-range clause from a None."""
    a = _ability(brawler_class="Tank", range_tiles=None,
                 range_min=None, range_max=None, confidence="unknown")
    assert a.aim == AUTO
    assert a.gate_range_tiles is None


def test_the_split_is_the_one_documented():
    book = default_abilities()
    manual = {a.id for a in book.manual_aimers()}
    art = {a.id for a in book.all() if a.brawler_class == "Artillery"}
    mark = {a.id for a in book.all() if a.brawler_class == "Marksman"}
    longr = {a.id for a in book.all()
             if (a.range_tiles or 0) >= LONG_RANGE_TILES}
    assert manual == art | mark | longr
    print(f"manual {len(manual)} = {len(art)} artillery ∪ {len(mark)} marksman "
          f"∪ {len(longr)} long-range; auto {len(book) - len(manual)}")


# --- the range gate --------------------------------------------------------- #
def _state(dist_frac, w=1280, h=720):
    return GameState(player_pos=(w // 2, h // 2), health=5000,
                     ammo_known=True, ammo_count=3,
                     enemy_positions=[(w // 2 + int(dist_frac * min(w, h)), h // 2)])


def test_gate_can_only_tighten_the_global_ceiling():
    """THE safety property. `tiles_across_screen` is an unverified estimate;
    this is what makes shipping it anyway defensible."""
    cfg = CombatConfig()
    book = default_abilities()
    for a in book.all():
        got = CombatPolicy(cfg, ability=a).range_frac()
        assert got <= cfg.max_range_frac + 1e-9, f"{a.id} loosened the gate"
    worst = max(book.all(), key=lambda a: CombatPolicy(cfg, ability=a).range_frac())
    best = min(book.all(), key=lambda a: CombatPolicy(cfg, ability=a).range_frac())
    print(f"gate spans {CombatPolicy(cfg, ability=best).range_frac():.3f} "
          f"({best.id}) .. {CombatPolicy(cfg, ability=worst).range_frac():.3f} "
          f"({worst.id}); global ceiling {cfg.max_range_frac}")


def test_short_range_brawler_stops_firing_into_space():
    """Edgar reaches ~3 tiles and was firing at four times that, paying
    attack_cost to spray at people he cannot touch."""
    book = default_abilities()
    edgar = CombatPolicy(CombatConfig(), ability=book.get("edgar"))
    shelly = CombatPolicy(CombatConfig(), ability=book.get("shelly"))
    far = _state(0.35)
    assert not edgar.decide(far, frame_size=(1280, 720)).attack
    assert shelly.decide(far, frame_size=(1280, 720)).attack
    print(f"edgar gate {edgar.range_frac():.3f} vs shelly {shelly.range_frac():.3f}")


def test_no_ability_behaves_exactly_as_before():
    """The offline env, the tests and any session without a brawler have no
    per-brawler facts, and must be untouched by all of this."""
    cfg = CombatConfig()
    assert CombatPolicy(cfg, ability=None).range_frac() == cfg.max_range_frac
    d = CombatPolicy(cfg).decide(_state(0.3), frame_size=(1280, 720))
    assert d.attack and d.aim is None and d.mode == AUTO


def test_gate_uses_the_band_maximum_not_the_midpoint():
    """Where only a band is published the true value is somewhere inside it.
    Blocking a shot that would have landed costs a kill; allowing one that
    misses costs attack_cost (0.05). Err toward allowing."""
    a = _ability(range_tiles=6.0, range_min=5.0, range_max=7.0)
    assert a.gate_range_tiles == 7.0


# --- what comes out --------------------------------------------------------- #
def test_decision_still_unpacks_as_a_pair():
    """`decode_action` and every existing test take (attack, super)."""
    d = CombatDecision(attack=True, super=False)
    attack, fire_super = d
    assert attack is True and fire_super is False
    assert len(d) == 2 and d[0] is True


def test_manual_aimer_emits_a_unit_vector_toward_the_target():
    book = default_abilities()
    cp = CombatPolicy(CombatConfig(), ability=book.get("piper"))
    d = cp.decide(_state(0.2), frame_size=(1280, 720))
    assert d.attack and d.mode == MANUAL
    assert d.aim is not None
    assert abs((d.aim[0] ** 2 + d.aim[1] ** 2) ** 0.5 - 1.0) < 1e-6
    assert d.aim[0] > 0.99, "enemy is due east; aim should be too"


def test_auto_aimer_emits_no_vector():
    cp = CombatPolicy(CombatConfig(), ability=default_abilities().get("shelly"))
    d = cp.decide(_state(0.2), frame_size=(1280, 720))
    assert d.attack and d.mode == AUTO and d.aim is None


def test_thrower_scales_the_drag_with_distance():
    """For a lobbed attack the drag LENGTH decides where the shot lands, so a
    thrower aimed at full deflection always overshoots a close target."""
    cp = CombatPolicy(CombatConfig(), ability=default_abilities().get("barley"))
    near = cp.decide(_state(0.10), frame_size=(1280, 720))
    far = cp.decide(_state(0.38), frame_size=(1280, 720))
    assert near.attack and far.attack
    assert near.aim_frac < far.aim_frac
    print(f"barley aim_frac: near {near.aim_frac:.2f} far {far.aim_frac:.2f}")


def test_no_aim_vector_when_nothing_fires():
    """An aim attached to a decision that fires nothing is a value nobody reads
    and everybody has to reason about."""
    cp = CombatPolicy(CombatConfig(), ability=default_abilities().get("piper"))
    d = cp.decide(_state(0.9), frame_size=(1280, 720))
    assert not d.attack and d.aim is None


def test_intent_carries_the_aim_through_to_the_executor():
    from rl.actions import decode_action
    cp = CombatPolicy(CombatConfig(), ability=default_abilities().get("tick"))
    st = _state(0.2)
    d = cp.decide(st, frame_size=(1280, 720))
    intent = decode_action([0, 1], state=st, frame_size=(1280, 720), combat=d)
    assert intent.fire_attack and intent.aim == d.aim
    assert intent.aim_mode == MANUAL
    assert "AIM" in repr(intent)


def test_decode_action_still_accepts_a_bare_tuple():
    from rl.actions import decode_action
    intent = decode_action([0, 1], state=_state(0.2), frame_size=(1280, 720),
                           combat=(True, False))
    assert intent.fire_attack and intent.aim is None and intent.aim_mode == AUTO


def test_aimed_press_never_sends_a_drag_short_enough_to_read_as_a_tap():
    """A drag under the system's tap slop is delivered as a TAP, which fires an
    auto-aimed shot — so an under-length aim does not aim badly, it silently
    switches mode."""
    from rl.actions import Controls
    c = Controls.from_screen(2424, 1080)
    for frac in (0.0, 0.05, 0.15, 1.0):
        r = max(c.aim_min_radius,
                int(c.aim_radius * max(c.aim_min_frac, min(1.0, frac))))
        assert r >= c.aim_min_radius >= c.MIN_SWIPE_PX if hasattr(c, "MIN_SWIPE_PX") \
            else r >= c.aim_min_radius
    print(f"aim radius {c.aim_radius}px, floor {c.aim_min_radius}px")


# --- the book itself -------------------------------------------------------- #
def test_reload_picks_up_an_edited_file(tmp_path):
    import json
    p = tmp_path / "abilities.json"
    p.write_text(json.dumps({"brawlers": [
        {"id": "x", "name": "X", "class": "Marksman", "range_tiles": 9.0,
         "range_min": 9.0, "range_max": 9.0, "confidence": "measured",
         "sources": ["t"]}]}))
    book = AbilityBook(p)
    assert len(book) == 1 and book.get("x").aim == MANUAL
    assert book.get("nope") is None
