#!/usr/bin/env python3
"""
Reward-engine tests: drive a synthetic Showdown match through RewardCalculator
and assert every reward/punishment fires with the right sign and size, plus the
OCR-noise guards and the perception->GameState adapter.

    python scripts/test_rewards.py        (no cv2 / gym / SB3 needed)
"""
import pathlib
import sys


from rl.state import GameState, adapt_live_state, MISSING_EXTRACTORS  # noqa: E402
from rl.rewards import RewardCalculator, RewardConfig  # noqa: E402


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


def make_calc():
    c = RewardCalculator(RewardConfig())
    c.reset()
    return c


def test_baseline_first_tick_zero():
    c = make_calc()
    r = c.compute(GameState(tick=0, health=3000, cube_count=0, players_left=10, super_charge=0.0))
    assert r.total == 0.0
    print("ok  first tick = 0")


def test_cube_pickup():
    c = make_calc(); c.compute(GameState(health=3000, cube_count=0))
    r = c.compute(GameState(health=3000, cube_count=2))
    assert approx(r.breakdown["cube_pickup"], 2.0), r.breakdown
    print("ok  cube pickup ->", r.breakdown)


def test_super_charge():
    c = make_calc(); c.compute(GameState(health=3000, super_charge=0.0))
    r = c.compute(GameState(health=3000, super_charge=0.5))
    assert approx(r.breakdown["super_charge"], 2.0), r.breakdown  # 0.5 * 4.0
    print("ok  super charge ->", r.breakdown)


def test_kill_and_placement():
    c = make_calc(); c.compute(GameState(health=3000, players_left=10))
    r = c.compute(GameState(health=3000, players_left=9, kills_this_tick=1))
    assert approx(r.breakdown["kill"], 5.0) and approx(r.breakdown["placement"], 3.0), r.breakdown
    print("ok  kill+placement ->", r.breakdown)


def test_survive_gated_by_gas():
    c = make_calc(); c.compute(GameState(health=3000, is_alive=True))
    safe = c.compute(GameState(health=3000, is_alive=True, in_gas=False))
    # Read the magnitude from the config rather than repeating it here: these
    # numbers are meant to be TUNED, and a test that hard-codes them turns
    # every tuning change into a spurious failure. What is being asserted is
    # the gating behaviour (paid when safe, withheld in gas), not the value.
    assert approx(safe.breakdown["survive"], RewardConfig().survive_tick), safe.breakdown
    gas = c.compute(GameState(health=3000, is_alive=True, in_gas=True))
    assert "survive" not in gas.breakdown and gas.breakdown["gas"] < 0, gas.breakdown
    print("ok  survive gating ->", safe.breakdown, "/", gas.breakdown)


def test_damage_scaled():
    c = make_calc(); c.compute(GameState(health=4000, is_alive=True))
    r = c.compute(GameState(health=3000, is_alive=True))
    assert approx(r.breakdown["damage_taken"], -0.75), r.breakdown  # -3*(1000/4000)
    print("ok  damage scaled ->", r.breakdown)


def test_death_terminal_once():
    c = make_calc(); c.compute(GameState(health=500, is_alive=True))
    r = c.compute(GameState(health=0, is_alive=False, match_over=True))
    assert approx(r.breakdown["death"], -30.0) and "damage_taken" not in r.breakdown, r.breakdown
    r2 = c.compute(GameState(health=0, is_alive=False, match_over=True))
    assert "death" not in r2.breakdown
    print("ok  death terminal ->", r.breakdown)


def test_win_terminal():
    c = make_calc(); c.compute(GameState(health=3000, players_left=2, is_alive=True))
    r = c.compute(GameState(health=3000, players_left=1, is_alive=True, won=True, match_over=True))
    assert approx(r.breakdown["win"], 50.0) and r.total > 0, r.breakdown
    print("ok  win terminal ->", r.breakdown)


def test_none_health_carry_forward():
    c = make_calc(); c.compute(GameState(health=3000))
    r = c.compute(GameState(health=None))
    assert "damage_taken" not in r.breakdown
    r2 = c.compute(GameState(health=2500))
    assert r2.breakdown["damage_taken"] < 0
    print("ok  None health carry-forward ->", r2.breakdown)


def test_cube_spike_rejected():
    c = make_calc(); c.compute(GameState(cube_count=1))
    r = c.compute(GameState(cube_count=999))
    assert "cube_pickup" not in r.breakdown
    print("ok  cube OCR spike rejected")


def test_players_left_monotonic():
    c = make_calc(); c.compute(GameState(players_left=5, is_alive=True))
    r = c.compute(GameState(players_left=8, is_alive=True))
    assert "placement" not in r.breakdown
    print("ok  players_left monotonic guard")


def test_adapter_maps_live_dict():
    # win case
    live = {"game_state": {"state": "match_end", "brawlers_left": None, "rank": 1},
            "anchor": None, "hp": None, "ammo": 0, "hud_cubes": None,
            "enemies": [], "boxes": [], "cubes": [], "in_gas": False}
    gs = adapt_live_state(live, tick=3)
    assert gs.match_over and gs.won and gs.is_alive, gs
    # death case (rank >= 2)
    live["game_state"]["rank"] = 5
    gs2 = adapt_live_state(live)
    assert gs2.match_over and not gs2.won and not gs2.is_alive, gs2
    # in-match mapping
    live3 = {"game_state": {"state": "in_match", "brawlers_left": 6, "rank": None},
             "anchor": (100, 200, 30), "hp": 2500, "ammo": 3, "hud_cubes": 4,
             "enemies": [{"center": (150, 220), "radius": 20}], "boxes": [{}], "cubes": [{}, {}],
             "in_gas": True}
    gs3 = adapt_live_state(live3, tick=9)
    assert (gs3.health, gs3.cube_count, gs3.players_left, gs3.in_gas, gs3.n_ground_cubes) == (2500, 4, 6, True, 2), gs3
    assert gs3.player_pos == (100, 200) and gs3.enemy_positions == [(150, 220)], gs3
    print("ok  adapter maps live dict (win/death/in-match)")


def test_full_match_win_beats_death():
    w = make_calc()
    for s in [
        GameState(health=3000, cube_count=0, players_left=10, super_charge=0.0, is_alive=True),
        GameState(health=3000, cube_count=2, players_left=10, super_charge=0.3, is_alive=True),
        GameState(health=2600, cube_count=3, players_left=8, super_charge=0.7, is_alive=True),
        GameState(health=2600, cube_count=5, players_left=5, super_charge=1.0, is_alive=True, kills_this_tick=1),
        GameState(health=2600, cube_count=6, players_left=1, is_alive=True, won=True, match_over=True),
    ]:
        w.compute(s)
    d = make_calc()
    for s in [
        GameState(health=3000, players_left=10, is_alive=True),
        GameState(health=1500, players_left=10, is_alive=True, in_gas=True),
        GameState(health=0, players_left=10, is_alive=False, match_over=True),
    ]:
        d.compute(s)
    print(f"\n  win  return = {w.episode_return:+.2f}  {dict(w.episode_breakdown)}")
    print(f"  loss return = {d.episode_return:+.2f}  {dict(d.episode_breakdown)}")
    assert w.episode_return > 0 > d.episode_return and w.episode_return > d.episode_return
    print("ok  win >> death over a full match")


def test_allow_positive_gate():
    # A tick that earns a cube (+) and survive (+) but also takes damage (-):
    # with allow_positive=False only the punishment survives.
    c = make_calc(); c.compute(GameState(health=4000, cube_count=0, is_alive=True))
    r = c.compute(GameState(health=3000, cube_count=1, is_alive=True), allow_positive=False)
    assert "cube_pickup" not in r.breakdown and "survive" not in r.breakdown, r.breakdown
    assert r.breakdown["damage_taken"] < 0 and r.total < 0, r.breakdown
    # sanity: same transition WITH positives allowed does include them
    c2 = make_calc(); c2.compute(GameState(health=4000, cube_count=0, is_alive=True))
    r2 = c2.compute(GameState(health=3000, cube_count=1, is_alive=True))
    assert "cube_pickup" in r2.breakdown and "survive" in r2.breakdown, r2.breakdown
    print("ok  allow_positive gate ->", r.breakdown)


def test_missing_extractors_declared():
    # super_charge (getSuper), kills_this_tick (KillAttributor), and
    # mid_match_death (getGameState "defeated" screen) are now done.
    assert "mid_match_death" not in MISSING_EXTRACTORS
    assert "super_charge" not in MISSING_EXTRACTORS
    assert "kills_this_tick" not in MISSING_EXTRACTORS
    print("ok  missing-extractor list:", list(MISSING_EXTRACTORS))


def test_defeated_screen_means_dead():
    # The mid-match death/spectate screen must read as dead + terminal-worthy.
    live = {"game_state": {"state": "defeated", "brawlers_left": None, "rank": None},
            "hp": 900, "ammo": 1, "hud_cubes": 3, "anchor": None,
            "enemies": [], "boxes": [], "cubes": [], "in_gas": False}
    gs = adapt_live_state(live)
    assert not gs.is_alive and not gs.match_over and not gs.won, gs
    print("ok  defeated screen -> is_alive=False")


def test_kill_attributor():
    from rl.kills import KillAttributor, KillConfig
    ka = KillAttributor(KillConfig())
    ka.reset()

    def live(state="in_match", left=8, sc=0.0, enemies=1, anchor=(600, 400, 30)):
        return {"game_state": {"state": state, "brawlers_left": left, "rank": None},
                "super_charge": sc, "anchor": anchor,
                "enemies": [{"center": (620, 410), "radius": 20}] * enemies}

    # Baseline (no drop) -> no kill.
    assert ka.update(live(left=8, sc=0.1)) == 0
    # Firing is now observed directly rather than inferred from the super charge
    # rising -- the charge caps at 1.0, so the inference silently stopped working
    # at full super. See rl/kills.py.
    ka.note_fired()
    ka.update(live(left=8, sc=0.3))
    credited = ka.update(live(left=7, sc=0.3, enemies=1))  # someone died
    assert credited == 1, credited

    # A death with NO recent shot -> not our kill (placement only).
    ka2 = KillAttributor()
    ka2.reset()
    ka2.update(live(left=6, sc=0.0, enemies=0))
    for _ in range(15):                                  # let the windows lapse
        ka2.update(live(left=6, sc=0.0, enemies=0))
    assert ka2.update(live(left=5, sc=0.0, enemies=0)) == 0
    print("ok  kill attributor credits own kills, ignores distant deaths")


def run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Running {len(tests)} tests\n" + "-" * 52)
    for t in tests:
        t()
    print("-" * 52 + f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    run_all()


# --- kill attribution -------------------------------------------------------
def _live(left, enemies=(), super_charge=0.0, anchor=(960, 540, 40),
          state="in_match", frame_size=(1920, 1080)):
    return {
        "game_state": {"state": state, "brawlers_left": left, "rank": None},
        "enemies": [{"center": e} for e in enemies],
        "anchor": anchor,
        "super_charge": super_charge,
        "frame_size": frame_size,
    }


def test_kill_credited_when_engaged():
    from rl.kills import KillAttributor
    k = KillAttributor()
    near = (960 + 200, 540)
    k.update(_live(5, [near]))
    k.note_fired()
    k.update(_live(5, [near]))
    k.note_fired()
    assert k.update(_live(4, [near])) == 1, "did not credit a kill while engaged"


def test_kill_not_credited_when_never_shot():
    """A death while the agent has not fired is someone else's kill."""
    from rl.kills import KillAttributor
    k = KillAttributor()
    near = (960 + 200, 540)
    for _ in range(5):
        k.update(_live(5, [near]))
    assert k.update(_live(4, [near])) == 0, "credited a kill without firing"


def test_kill_not_credited_for_distant_death():
    from rl.kills import KillAttributor
    k = KillAttributor()
    far = (960 + 900, 540 + 400)          # well beyond auto-aim range
    k.note_fired()
    k.update(_live(5, [far]))
    assert k.update(_live(4, [far])) == 0, "credited a kill on an out-of-range enemy"


def test_kill_credit_survives_a_full_super():
    """THE REGRESSION THIS CLASS WAS REWRITTEN FOR.

    The old version inferred "dealt damage" from the super charge RISING. The
    charge caps at 1.0, so once full it stops rising and kills stopped being
    credited -- exactly when the agent is strongest. Firing is now observed
    directly, so a pinned super must not suppress credit.
    """
    from rl.kills import KillAttributor
    k = KillAttributor()
    near = (960 + 200, 540)
    for _ in range(6):                     # super pinned at full, never rises
        k.note_fired()
        k.update(_live(5, [near], super_charge=1.0))
    assert k.update(_live(4, [near], super_charge=1.0)) == 1, (
        "a full super suppressed kill credit")


def test_kill_attributor_resets_between_matches():
    from rl.kills import KillAttributor
    k = KillAttributor()
    near = (960 + 200, 540)
    k.note_fired()
    k.update(_live(3, [near]))
    k.update(_live(3, [near], state="match_end"))
    assert k._prev_left is None
    assert k.update(_live(9, [near])) == 0
