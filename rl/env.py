"""
rl/env.py
=========

The Gymnasium environment that ties the whole project together. One `step()` is
one game tick:

    action (policy)
      -> ActionExecutor.apply()                 actuate the device      [actions.py]
      -> source.grab()                          grab the next frame     [liveLoop sources]
      -> LivePerception.tick(frame) -> live dict perception stack        [perception/]
      -> adapt_live_state(live) -> GameState     normalize               [rl/state.py]
      -> RewardCalculator.compute(state)         scalar reward           [rl/rewards.py]
      -> encode_observation(state)               vector for the network
      -> (obs, reward, terminated, truncated, info)

Frame sources are injected via `source_factory` (a zero-arg callable returning a
fresh source each reset):

    offline / dry-run :  lambda: VideoSource("media/testvideos/test_game1.mp4")
    live play         :  lambda: ScreenSource(region)     + AdbExecutor

so you can develop and test the whole loop against recorded footage, then flip to
live capture without touching this file.
"""

from __future__ import annotations

import sys
import time
import pathlib
from typing import Callable, Optional

import numpy as np

# Make the sibling `perception` package importable even if env.py is imported
# directly (entry-point scripts also put the repo root on the path).
_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM = True
except Exception:  # allow import/use without gymnasium installed
    gym = object  # type: ignore
    spaces = None  # type: ignore
    _GYM = False

from perception.liveLoop import (  # noqa: E402
    LivePerception, VideoSource, ScreenSource, AdbScreencapSource)
from perception.getGameState import get_game_state  # noqa: E402
from perception.getTerrain import find_terrain, ProfileSelector  # noqa: E402
from perception.getGas import gas_info  # noqa: E402

from .actions import make_action_space, LoggingExecutor, ActionExecutor
from .camera_tracker import CameraTracker
from .path_planner import WaypointPlanner, PlannerStatus
from .rewards import RewardCalculator, RewardConfig
from .state import GameState, adapt_live_state
from .kills import KillAttributor

N_ENEMY_SLOTS = 3          # nearest N enemies encoded individually
N_BOX_SLOTS = 2            # nearest N power-cube boxes encoded individually
N_CUBE_SLOTS = 2           # nearest N ground cubes encoded individually

_ENTITY_DIM = (N_ENEMY_SLOTS + N_BOX_SLOTS + N_CUBE_SLOTS) * 4
_MOTION_DIM = 3            # velocity dx, dy, speed
_PREV_ACTION_DIM = 5       # cos/sin heading, distance tier, attacked, supered
_PLANNER_DIM = 6           # active, progress, waypoint dx/dy/dist, blocked
OBS_DIM = 18 + _ENTITY_DIM + _MOTION_DIM + _PREV_ACTION_DIM + _PLANNER_DIM  # = 60

# Screen pixels per step that count as "full speed" when normalising velocity.
# A brawler crosses roughly a quarter of the short side per second, so at a
# ~0.1s tick this is comfortably above the real maximum and keeps the feature
# inside [0, 1] without clipping normal movement flat.
_VEL_SCALE = 60.0

# Super charge at which the button is actually usable. Mirrors
# perception.getSuper.READY_CHARGE; tapping below this does nothing at all.
_SUPER_READY = 0.9

# How many steps a gas grid may be reused before it is recomputed.
#
# gas_info's mask is the single most expensive thing this env does (~3ms, more
# than the entire perception tick, which the stage scheduler keeps at ~1ms).
# But gas expands over SECONDS, so a grid a few steps old is still accurate --
# the only thing that goes stale quickly is its alignment, because the world
# scrolls underneath. That part is corrected exactly, by rolling the cached
# grid by the camera delta we already measure for waypoint latching.
_GAS_REFRESH_STEPS = 3

# The tick length PathPlannerConfig's tick counts were sized against.
_REFERENCE_TICK = 0.1


def _planner_config_for(tick_seconds: float):
    """Planner config with its tick counts rescaled for this loop rate.

    `commit_ticks` and `stuck_ticks` are counted in DECISIONS but chosen for
    wall-clock durations: (8, 16, 26) decisions is 0.8-2.6s of travel at the
    reference 0.1s tick, and 5 motionless decisions is half a second.

    Change --tick-seconds and those meanings silently change with it. Halving
    the tick to double the data rate would also halve every commitment, undoing
    the temporal abstraction the whole planner exists to provide -- a real trap,
    because the loop would look twice as productive while the agent quietly went
    back to twitching. Scaling here keeps commitments fixed in SECONDS, so the
    tick rate is safe to tune.
    """
    from dataclasses import replace
    from .path_planner import PathPlannerConfig

    base = PathPlannerConfig()
    if not tick_seconds or tick_seconds <= 0:
        return base                       # offline / unpaced: leave as authored
    scale = _REFERENCE_TICK / float(tick_seconds)
    if abs(scale - 1.0) < 0.05:
        return base
    return replace(
        base,
        commit_ticks=tuple(max(1, int(round(t * scale))) for t in base.commit_ticks),
        stuck_ticks=max(2, int(round(base.stuck_ticks * scale))),
    )


def _nearest_slots(positions, px, py, w, h, n_slots):
    """Encode the nearest `n_slots` positions as [present, dx, dy, dist] each.

    Nearest-first; dx/dy centered on 0.5. Empty slots are
    [0.0, 0.5, 0.5, 1.0] (absent, centered, max distance).
    """
    by_dist = sorted(
        positions,
        key=lambda p: (p[0] - px) ** 2 + (p[1] - py) ** 2,
    )[:n_slots]
    out = []
    for i in range(n_slots):
        if i < len(by_dist):
            x, y = by_dist[i]
            dx, dy = (x - px) / w, (y - py) / h
            out += [
                1.0,
                float(np.clip(dx * 0.5 + 0.5, 0.0, 1.0)),
                float(np.clip(dy * 0.5 + 0.5, 0.0, 1.0)),
                min(1.0, (dx ** 2 + dy ** 2) ** 0.5),
            ]
        else:
            out += [0.0, 0.5, 0.5, 1.0]
    return out


def encode_observation(state: GameState, max_health: float,
                       frame_size=(1280, 720), velocity=(0.0, 0.0),
                       prev_action=None, planner_status: PlannerStatus = None) -> np.ndarray:
    """Flatten a GameState into a fixed-length, ~[0,1]-scaled vector.

    Layout:
        0  health fraction          9-12 gas coverage left/right/above/below
        1  cubes                    13   gas safe-direction dx (0.5 = none)
        2  ammo                     14   gas safe-direction dy
        3  super charge             15   n enemies
        4  players left             16   ground cubes
        5  in gas                   17   n boxes
        6  alive                    18+  N_ENEMY_SLOTS x [present, dx, dy, dist]
        7  player x                      then N_BOX_SLOTS x 4 (nearest boxes)
        8  player y                      then N_CUBE_SLOTS x 4 (nearest ground
                                         cubes); all nearest-first, empty slot
                                         = [0, 0.5, 0.5, 1]

    Then the three motion blocks. These exist because everything above is a
    single-frame SNAPSHOT: it says where things are, never that the agent is
    already moving. A policy that cannot perceive its own motion has no way to
    prefer continuing a smooth run over reversing into a jitter, which is a
    large part of why v3's movement never settled.

        velocity     : world-frame dx, dy (0.5-centred) and speed
        prev action  : cos/sin of the last heading, distance tier, attack, super
                       — lets the network notice it is repeating or thrashing
        planner      : active, progress, waypoint dx/dy/distance, blocked
                       — CRITICAL: while a commitment is active the movement
                       heads are ignored, and this is how the agent can tell
                       which steps its movement choice actually mattered on
    """
    w, h = frame_size
    px, py = state.player_pos or (w / 2, h / 2)

    hp = state.health or 0
    gl, gr, gt, gb = state.gas_sides
    sdx, sdy = state.gas_safe

    obs = [
        np.clip(hp / max(1.0, max_health), 0.0, 1.0),        # 0 health fraction
        np.clip((state.cube_count or 0) / 20.0, 0.0, 1.0),   # 1 cubes
        np.clip(state.ammo_count / 6.0, 0.0, 1.0),           # 2 ammo
        np.clip(state.super_charge or 0.0, 0.0, 1.0),        # 3 super charge
        np.clip((state.players_left or 10) / 10.0, 0.0, 1.0),# 4 players left
        1.0 if state.in_gas else 0.0,                        # 5 in gas
        1.0 if state.is_alive else 0.0,                      # 6 alive
        np.clip(px / w, 0.0, 1.0),                           # 7 player x
        np.clip(py / h, 0.0, 1.0),                           # 8 player y
        np.clip(gl, 0.0, 1.0),                               # 9 gas left of player
        np.clip(gr, 0.0, 1.0),                               # 10 gas right
        np.clip(gt, 0.0, 1.0),                               # 11 gas above
        np.clip(gb, 0.0, 1.0),                               # 12 gas below
        np.clip(sdx * 0.5 + 0.5, 0.0, 1.0),                  # 13 gas safe dx
        np.clip(sdy * 0.5 + 0.5, 0.0, 1.0),                  # 14 gas safe dy
        np.clip(len(state.enemy_positions) / 9.0, 0.0, 1.0), # 15 n enemies
        np.clip(state.n_ground_cubes / 10.0, 0.0, 1.0),      # 16 ground cubes
        np.clip(state.n_boxes / 10.0, 0.0, 1.0),             # 17 n boxes
    ]

    # Nearest-K slots, each as [present, dx, dy, dist] relative to the player
    obs += _nearest_slots(state.enemy_positions, px, py, w, h, N_ENEMY_SLOTS)
    obs += _nearest_slots(state.box_positions, px, py, w, h, N_BOX_SLOTS)
    obs += _nearest_slots(state.ground_cube_positions, px, py, w, h, N_CUBE_SLOTS)

    # --- motion ---
    vx, vy = velocity
    obs += [
        float(np.clip(vx / _VEL_SCALE * 0.5 + 0.5, 0.0, 1.0)),
        float(np.clip(vy / _VEL_SCALE * 0.5 + 0.5, 0.0, 1.0)),
        float(np.clip((vx ** 2 + vy ** 2) ** 0.5 / _VEL_SCALE, 0.0, 1.0)),
    ]

    # --- previous action ---
    if prev_action is None:
        # "No previous action" must not look like a real one. Heading is encoded
        # on the unit circle, so its neutral value is the centre (0.5, 0.5).
        obs += [0.5, 0.5, 0.0, 0.0, 0.0]
    else:
        from .actions import N_HEADINGS, N_DISTANCES
        heading, dist, atk, sup = (int(prev_action[0]), int(prev_action[1]),
                                   int(prev_action[2]), int(prev_action[3]))
        angle = 2.0 * np.pi * (heading % N_HEADINGS) / N_HEADINGS
        obs += [
            float(np.cos(angle) * 0.5 + 0.5),
            float(np.sin(angle) * 0.5 + 0.5),
            float(dist / max(1, N_DISTANCES - 1)),
            1.0 if atk else 0.0,
            1.0 if sup else 0.0,
        ]

    # --- planner commitment ---
    s = planner_status or PlannerStatus()
    obs += [
        1.0 if s.active else 0.0,
        float(np.clip(s.progress, 0.0, 1.0)),
        float(np.clip(s.waypoint_dx * 0.5 + 0.5, 0.0, 1.0)),
        float(np.clip(s.waypoint_dy * 0.5 + 0.5, 0.0, 1.0)),
        float(np.clip(s.distance, 0.0, 1.0)),
        1.0 if s.blocked else 0.0,
    ]

    return np.array(obs, dtype=np.float32)


class BrawlStarsEnv(gym.Env if _GYM else object):
    """Gymnasium env for Brawl Stars Solo Showdown.

    Parameters
    ----------
    source_factory : Callable[[], source] | None
        Returns a fresh frame source (``.grab()`` -> BGR frame or None, ``.close()``)
        each reset. None = no capture (smoke test with empty states).
    executor : ActionExecutor
        Sends actions to the device. Defaults to LoggingExecutor (dry run).
    reward_config : RewardConfig
    super_charge_fn / kills_fn : optional detector hooks
        ``fn(frame, live_dict) -> float|None`` and ``-> int``; wire these when the
        super-ring / kill detectors exist. Default: super=None, kills=0.
    tick_seconds : float
        Real-time pacing per step (0 for offline/fast).
    max_ticks : int
        Truncation horizon.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        source_factory: Optional[Callable[[], object]] = None,
        executor: Optional[ActionExecutor] = None,
        reward_config: Optional[RewardConfig] = None,
        frame_size=(1280, 720),
        tick_seconds: float = 0.0,
        max_ticks: int = 6000,
        super_charge_fn: Optional[Callable] = None,
        kills_fn: Optional[Callable] = None,
        move_window_seconds: float = 3.0,
        profile_every: int = 0,
    ):
        self.source_factory = source_factory
        # >0 prints a wall-clock breakdown every N steps (see _report_profile).
        self.profile_every = int(profile_every)
        self._prof = dict.fromkeys(
            ("act", "pace", "capture", "perceive", "spatial", "reward",
             "total", "n", "reset_wait", "resets"), 0.0)
        self.executor = executor or LoggingExecutor()
        self.reward_calc = RewardCalculator(reward_config)
        self.frame_size = frame_size
        self.tick_seconds = tick_seconds
        self.max_ticks = max_ticks
        self.super_charge_fn = super_charge_fn
        self.kills_fn = kills_fn
        self.kill_attr = KillAttributor()   # default kill attribution
        # Anti-idle: positives are withheld unless the agent moved within this many
        # seconds. Attack is also suppressed unless an enemy is visible.
        self.move_window_seconds = move_window_seconds
        self._last_move_t = 0.0
        self._last_enemy_count = 0

        self.perception: Optional[LivePerception] = None
        self.source = None
        self._tick = 0
        self._last_state = GameState()
        self.path_planner = WaypointPlanner(_planner_config_for(tick_seconds))
        self.camera = CameraTracker()
        # Per-env, not the module default: each episode is a fresh match and may
        # be a different map, so the profile choice must not carry over.
        self.terrain_profiles = ProfileSelector()

        # Per-tick perception products consumed by the NEXT action. They are one
        # step old by construction: the agent acts on the most recent frame it
        # was actually shown, which is also what the policy conditioned on.
        self._terrain = None
        self._camera_delta = (0.0, 0.0)
        self._velocity = (0.0, 0.0)
        self._prev_player_pos = None
        self._prev_action = None
        self._gas_grid = None          # cached; refreshed every _GAS_REFRESH_STEPS
        self._gas_age = 0
        self._gas_drift = [0.0, 0.0]   # sub-cell camera drift not yet rolled off
        self._last_rect = None         # keeps the camera crop stable across skips
        self._next_tick_at = None      # deadline for _pace()

        if _GYM:
            self.action_space = make_action_space()
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)

    # Menu button positions as screen fractions, measured on 2424x1080
    # reference screenshots (defeated.png / endMenu.png at the repo root).
    EXIT_BUTTON_NORM = (0.538, 0.922)          # "Exit" on the defeated screen (dead-center, measured on defeated.png)
    PLAY_AGAIN_BUTTON_NORM = (0.7376, 0.9167)  # "Play Again" on the end menu
    END_MENU_WAIT = 5.0        # settle time before Play Again is tappable
    NAVIGATE_TIMEOUT = 180.0   # give up navigating after this many seconds

    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        if _GYM:
            super().reset(seed=seed)
        if self.source is not None:
            self.source.close()
        self.source = self.source_factory() if self.source_factory else None
        _t_nav = time.perf_counter()
        self._navigate_to_match()            # auto-reset: menus -> new match
        _nav = time.perf_counter() - _t_nav
        self.perception = LivePerception()   # fresh smoothing state per episode
        self.reward_calc.reset()
        self.kill_attr.reset()
        self.path_planner.reset()      # drop any commitment from the last match
        self.camera.reset()            # and any cross-match frame correlation
        self.terrain_profiles.reset()  # the next match may be a different map
        self._tick = 0
        self._last_move_t = 0.0          # not "recently moved" until it actually moves
        self._last_enemy_count = 0
        self._terrain = None
        self._camera_delta = (0.0, 0.0)
        self._velocity = (0.0, 0.0)
        self._prev_player_pos = None
        self._prev_action = None
        self._gas_grid = None          # cached; refreshed every _GAS_REFRESH_STEPS
        self._gas_age = 0
        self._gas_drift = [0.0, 0.0]   # sub-cell camera drift not yet rolled off
        self._last_rect = None         # keeps the camera crop stable across skips
        self._next_tick_at = None      # deadline for _pace()
        state = self._observe()
        self._last_state = state
        self._last_enemy_count = len(state.enemy_positions)
        # reset() does a full observe of its own; zero the per-step counters
        # afterwards so the profile reports STEP time only and the rows add up.
        # The between-match figures are deliberately NOT zeroed -- they are what
        # explains a low SB3 fps against a healthy in-match rate.
        keep_wait = self._prof["reset_wait"] + _nav
        keep_n = self._prof["resets"] + 1
        for k in self._prof:
            self._prof[k] = 0.0
        self._prof["reset_wait"] = keep_wait
        self._prof["resets"] = keep_n
        return self._encode(state), {"state": state}

    def _navigate_to_match(self):
        """Drive the menus back into a match: defeated -> Exit, end menu ->
        wait 5s -> Play Again, loading/unknown -> wait. Live sources only
        (recordings can't react to taps); no-ops without a tappable executor.
        """
        if (self.source is None or not getattr(self.source, "live", False)
                or not hasattr(self.executor, "tap_norm")):
            return
        # Pause background frame capture so the menu taps (Exit / Play Again) get
        # the USB/adb channel to themselves; continuous screencap otherwise queues
        # behind big frame transfers and makes restarting a match sluggish. grab()
        # still works — it captures synchronously while paused.
        pause = getattr(self.source, "pause", None)
        resume = getattr(self.source, "resume", None)
        if pause:
            pause()
        try:
            deadline = time.time() + self.NAVIGATE_TIMEOUT
            while time.time() < deadline:
                frame = self.source.grab()
                if frame is None:
                    time.sleep(0.5)
                    continue
                screen = get_game_state(frame)["state"]
                if screen == "in_match":
                    return
                # Tap the button on EVERY pass while the screen is showing (rather
                # than one well-timed shot). Early taps during the rank/trophy
                # animation harmlessly miss; a later one lands once the button is
                # live, and the loop stops as soon as the screen advances.
                if screen == "defeated":
                    self.executor.tap_norm(*self.EXIT_BUTTON_NORM)
                    time.sleep(1.2)
                elif screen == "match_end":
                    self.executor.tap_norm(*self.PLAY_AGAIN_BUTTON_NORM)
                    time.sleep(1.2)
                else:                            # loading / countdown / unknown
                    time.sleep(1.0)
            print("WARNING: _navigate_to_match timed out; continuing anyway")
        finally:
            if resume:
                resume()

    def _gate_weapons(self, action):
        """Suppress shots that provably cannot do anything.

        A tap with an empty clip, or a super tap at zero charge, is a pure
        no-op: it costs a frame, changes nothing, and teaches the policy
        nothing, because the outcome is identical whether it fired or not.
        Measured on a rollout before this gate existed: 89% of attacks were
        fired with an empty clip and 100% of supers were fired uncharged.

        Gating here rather than through the reward is the same trade as the gas
        cost layer -- it works from the first frame and costs no samples. The
        reward's attack_cost handles the shots that are merely UNLIKELY to land,
        which no gate can know about.

        The gates are deliberately asymmetric about uncertainty:
          * enemy visible  — required. Auto-aim has nothing to aim at otherwise.
          * ammo           — only when the reading is TRUSTED (`ammo_known`).
                             A failed read also reports 0, and the bar is only
                             located on ~15% of real frames, so trusting the
                             count alone would block nearly every attack.
          * super charge   — always applied. This comes from a fixed HUD ROI
                             that is never occluded, so 0.0 means uncharged
                             rather than unknown.
        """
        if not hasattr(action, "__len__") or len(action) < 4:
            return action, False, False
        st = self._last_state
        want_attack, want_super = int(action[2]) == 1, int(action[3]) == 1

        allow_attack = want_attack and self._last_enemy_count >= 1
        if allow_attack and st.ammo_known and st.ammo_count < 1:
            allow_attack = False
        allow_super = want_super and (st.super_charge or 0.0) >= _SUPER_READY

        if allow_attack != want_attack or allow_super != want_super:
            action = list(action)
            action[2] = 1 if allow_attack else 0
            action[3] = 1 if allow_super else 0
        return action, allow_attack, allow_super

    def step(self, action):
        t_step = time.perf_counter()
        action, fired_attack, fired_super = self._gate_weapons(action)

        t0 = time.perf_counter()
        intent = self.executor.apply(action, state=self._last_state,
                                     frame_size=self.frame_size,
                                     planner=self.path_planner,
                                     terrain=self._terrain,
                                     camera_delta=self._camera_delta)  # 1) act
        self._prof["act"] += time.perf_counter() - t0

        self._prev_action = intent.raw
        if intent.move != (0.0, 0.0):
            self._last_move_t = time.time()           # mark that we moved

        t0 = time.perf_counter()
        self._pace()
        self._prof["pace"] += time.perf_counter() - t0

        self._tick += 1
        state = self._observe()                        # 2) perceive
        self._last_state = state
        self._last_enemy_count = len(state.enemy_positions)

        # Anti-idle: withhold all positive rewards unless we moved in the last
        # `move_window_seconds`; punishments always apply.
        moved_recently = (time.time() - self._last_move_t) <= self.move_window_seconds
        t0 = time.perf_counter()
        result = self.reward_calc.compute(state, allow_positive=moved_recently,
                                          fired_attack=fired_attack,
                                          fired_super=fired_super)  # 3) score
        self._prof["reward"] += time.perf_counter() - t0

        # Episode ends when the match ends OR the moment we die (the defeated/
        # spectate screen) — no point learning from spectator frames, and the
        # next reset() navigates back into a fresh match.
        terminated = bool(state.match_over or not state.is_alive)
        truncated = self._tick >= self.max_ticks
        status = self.path_planner.status()
        info = {
            "reward_breakdown": result.breakdown,
            "episode_return": self.reward_calc.episode_return,
            "intent": repr(intent),
            "state": state,
            "waypoint_active": status.active,
            "waypoint_blocked": status.blocked,
            "camera_delta": self._camera_delta,
            "velocity": self._velocity,
        }
        self._prof["total"] += time.perf_counter() - t_step
        self._prof["n"] += 1
        if self.profile_every and self._prof["n"] >= self.profile_every:
            self._report_profile()
        return self._encode(state), result.total, terminated, truncated, info

    def actuate(self, action):
        """Apply an action to the device WITHOUT capturing or perceiving.

        Used by ActionRepeat for the in-between ticks of a decision: the agent's
        chosen action drives several real movement swipes (so it commits to a
        direction), but the expensive capture + perception + reward runs only once
        per decision (on the final, full step()).
        """
        action, _, _ = self._gate_weapons(action)
        # No fresh frame here, so no camera delta to apply: passing (0, 0) keeps
        # the latched waypoint where it is rather than drifting it by a stale
        # measurement that has already been consumed.
        intent = self.executor.apply(action, state=self._last_state,
                                     frame_size=self.frame_size,
                                     planner=self.path_planner,
                                     terrain=self._terrain,
                                     camera_delta=(0.0, 0.0))
        if intent.move != (0.0, 0.0):
            self._last_move_t = time.time()
        self._pace()

    def _pace(self):
        """Sleep until the next tick is due, rather than for a fixed duration.

        `tick_seconds` used to be a flat `time.sleep()` added ON TOP of capture
        and perception, so a step actually took `tick_seconds + compute`. With
        the default 0.1s tick and ~25ms of capture+perception that is 125ms per
        step -- the loop ran ~20% slower than requested, and got slower still
        on any frame where a heavy perception stage came due.

        Waiting on a DEADLINE instead absorbs the compute into the interval, so
        the loop runs at the rate the flag actually names and the tick rate
        stops drifting with scene complexity. Free throughput: no work is
        skipped, only dead time removed.

        Lateness is not carried forward. If a step overruns its slot we simply
        start the next one immediately and re-base the deadline; accumulating
        the debt would make the env fire a burst of catch-up steps with no
        pacing at all, which is worse than being slightly late.
        """
        if not self.tick_seconds:
            return
        now = time.perf_counter()
        due = getattr(self, "_next_tick_at", None)
        if due is None:
            due = now + self.tick_seconds
        remaining = due - now
        if remaining > 0:
            time.sleep(remaining)
            self._next_tick_at = due + self.tick_seconds
        else:
            self._next_tick_at = now + self.tick_seconds

    def close(self):
        if self.source is not None:
            self.source.close()
        self.executor.close()

    # ------------------------------------------------------------------ #
    def _report_profile(self) -> None:
        """Print where a step's wall-clock actually goes, then reset the counters.

        Exists because guessing at this is unreliable: an offline benchmark
        feeds frames from memory, so CAPTURE costs nothing and the profile is
        dominated by compute — the exact opposite of a live phone, where
        `adb screencap` is typically 150-400ms and everything else is noise.
        The only trustworthy measurement is on the real device.

        Read it as: whichever row dominates is the only one worth optimising.
          capture  -> switch transport (scrcpy); compute work will not help
          perceive -> raise the intervals in liveLoop.STAGE_INTERVALS
          act      -> adb touch latency; check Controls.hold_ms vs tick_seconds
          pace     -> the loop is idle-waiting, so lower --tick-seconds
        """
        p = self._prof
        n = max(1, p["n"])
        total = p["total"] / n
        print(f"\n[env profile] {n} steps | {1.0 / max(total, 1e-9):5.1f} fps "
              f"| {1000 * total:6.1f} ms/step")
        parts = 0.0
        for key, label in (("act", "act (adb touch)"), ("pace", "pace (idle wait)"),
                           ("capture", "capture (grab)"), ("perceive", "perceive"),
                           ("spatial", "terrain+gas+camera"), ("reward", "reward+encode")):
            parts += p[key]
            print(f"    {label:22s} {1000 * p[key] / n:7.1f} ms  "
                  f"{100 * p[key] / max(p['total'], 1e-9):4.0f}%")
        # Anything the named timers did not cover (gym/SB3 wrapper overhead,
        # the observation copy). Shown so the rows always add up to the total
        # rather than quietly disagreeing with it.
        print(f"    {'other':22s} {1000 * max(0.0, p['total'] - parts) / n:7.1f} ms")

        # Time spent BETWEEN matches, which no per-step figure can show. This is
        # the usual explanation when SB3's reported fps is far below the
        # in-match rate measured above: menu navigation, end-of-match
        # animations and matchmaking all happen inside reset(), and SB3 counts
        # that wall clock against every step in the rollout.
        if p["resets"]:
            per = p["reset_wait"] / p["resets"]
            share = p["reset_wait"] / max(p["reset_wait"] + p["total"], 1e-9)
            eff = (n + p["resets"] * 0) / max(p["total"] + p["reset_wait"], 1e-9)
            print(f"    {'-- between matches':22s} {p['resets']:.0f} reset(s), "
                  f"{per:.1f}s each, {100 * share:.0f}% of wall clock")
            print(f"    {'effective rate':22s} {eff:5.1f} fps "
                  f"(what SB3 reports, vs {1.0 / max(total, 1e-9):.1f} in-match)")
        for k in p:
            p[k] = 0.0

    def _observe(self) -> GameState:
        if self.source is None:
            return GameState(tick=self._tick)
        t0 = time.perf_counter()
        frame = self.source.grab()
        self._prof["capture"] += time.perf_counter() - t0
        if frame is None:
            # Live feed (phone/screen): a dropped frame is transient — skip this
            # tick, keep the episode going. Recording: None means it ended.
            if getattr(self.source, "live", False):
                return GameState(tick=self._tick)
            return GameState(tick=self._tick, match_over=True)
        t0 = time.perf_counter()
        live, _timings = self.perception.tick(frame)
        self._prof["perceive"] += time.perf_counter() - t0
        sc = self.super_charge_fn(frame, live) if self.super_charge_fn else None
        kills = self.kills_fn(frame, live) if self.kills_fn else self.kill_attr.update(live)
        fh, fw = frame.shape[:2]
        state = adapt_live_state(live, tick=self._tick, super_charge=sc,
                                 kills_this_tick=kills, frame_size=(fw, fh))
        t0 = time.perf_counter()
        self._update_spatial(frame, state)
        self._prof["spatial"] += time.perf_counter() - t0
        return state

    def _update_spatial(self, frame, state: GameState) -> None:
        """Terrain grid, camera scroll and player velocity for this frame.

        Order matters: terrain first, because its play rect is what the camera
        tracker crops against.
        """
        # Adopt the real capture size. The (1280, 720) default is only a guess,
        # and every normalisation in encode_observation divides by it — a phone
        # capture is 2424x1080, so keeping the guess would skew every dx/dy the
        # policy sees.
        fh, fw = frame.shape[:2]
        self.frame_size = (fw, fh)

        # SKIP THE SPATIAL WORK WITH NO PLAYER. The planner returns immediately
        # when player_pos is None (it has no frame of reference for a
        # destination), so terrain and gas would be computed and thrown away.
        # On real footage the anchor is unavailable on roughly half of all
        # steps, and these two are ~4ms of the ~12ms step, so this is the
        # single largest saving available without touching perception.
        if state.player_pos is None:
            self._terrain = None
        else:
            # find_terrain returns None when no map profile matches the frame.
            # That means "no terrain information", NOT "everything is blocked" —
            # the planner falls back to direct steering, which is far better
            # than A* routing around walls that do not exist.
            try:
                self._terrain = find_terrain(frame, player_pos=state.player_pos,
                                             selector=self.terrain_profiles)
            except Exception:
                self._terrain = None

            # Gas rides along inside the terrain dict, on the same grid, so A*
            # can add it straight into the cost map. Routing around the cloud is
            # handled here rather than by a reward term: it needs no samples to
            # learn, works from the very first frame, and — because it is a cost
            # and not a wall — still lets the agent cut through gas when that is
            # the only way out.
            if self._terrain is not None:
                self._terrain["gas"] = self._gas_for(frame, self._terrain)

        rect = (self._terrain["rect"] if self._terrain
                else self._last_rect)   # keep the camera crop stable across skips
        if rect is not None:
            self._last_rect = rect
        dx, dy, ok = self.camera.update(frame, rect)
        self._camera_delta = (dx, dy) if ok else (0.0, 0.0)

        # World-frame player velocity. The camera CHASES the player, so his
        # screen position barely changes while walking — screen delta alone
        # reads as "standing still". Subtracting the world's scroll recovers
        # the real motion: walk right, world scrolls left, velocity is right.
        if state.player_pos is not None and self._prev_player_pos is not None:
            sdx = state.player_pos[0] - self._prev_player_pos[0]
            sdy = state.player_pos[1] - self._prev_player_pos[1]
            self._velocity = (sdx - self._camera_delta[0], sdy - self._camera_delta[1])
        elif not ok:
            self._velocity = (0.0, 0.0)
        else:
            self._velocity = (-self._camera_delta[0], -self._camera_delta[1])
        self._prev_player_pos = state.player_pos

    def _gas_for(self, frame, terrain):
        """Gas grid for this terrain, recomputed at most every N steps.

        Between refreshes the cached grid is ROLLED by the camera delta, so it
        stays registered to the world while the view scrolls. Gas itself
        expands over seconds, so age costs almost nothing; misalignment would
        cost a lot, and that is the part we correct exactly.

        Sub-cell motion is accumulated in `_gas_drift` and only applied once it
        reaches a whole cell, so slow scrolling is not silently discarded.
        """
        gh, gw = terrain["occupancy"].shape
        cw, ch = terrain["cell"]

        fresh = (self._gas_grid is None
                 or self._gas_grid.shape != (gh, gw)
                 or self._gas_age >= _GAS_REFRESH_STEPS)
        if fresh:
            try:
                self._gas_grid = gas_info(frame, grid_rect=terrain["rect"],
                                          grid_size=(gw, gh))["grid"]
            except Exception:
                self._gas_grid = None
            self._gas_age = 0
            self._gas_drift = [0.0, 0.0]
            return self._gas_grid

        self._gas_age += 1
        if self._gas_grid is None:
            return None

        self._gas_drift[0] += self._camera_delta[0] / max(cw, 1e-6)
        self._gas_drift[1] += self._camera_delta[1] / max(ch, 1e-6)
        sx, sy = int(self._gas_drift[0]), int(self._gas_drift[1])
        if sx or sy:
            # Newly exposed edges have never been observed; fill them with 0
            # (no known gas) rather than wrapping the opposite edge round.
            self._gas_grid = np.roll(self._gas_grid, (sy, sx), axis=(0, 1))
            if sx > 0:
                self._gas_grid[:, :sx] = 0.0
            elif sx < 0:
                self._gas_grid[:, sx:] = 0.0
            if sy > 0:
                self._gas_grid[:sy, :] = 0.0
            elif sy < 0:
                self._gas_grid[sy:, :] = 0.0
            self._gas_drift[0] -= sx
            self._gas_drift[1] -= sy
        return self._gas_grid

    def _encode(self, state: GameState) -> np.ndarray:
        max_hp = self.reward_calc._max_health or self.reward_calc.config.default_max_health
        return encode_observation(state, max_hp, self.frame_size,
                                  velocity=self._velocity,
                                  prev_action=self._prev_action,
                                  planner_status=self.path_planner.status())


# --- convenience factories --------------------------------------------------- #
def make_video_env(video_path: str, **kwargs) -> BrawlStarsEnv:
    """Offline env that plays a recording back as a live feed (great for tests)."""
    return BrawlStarsEnv(source_factory=lambda: VideoSource(video_path), **kwargs)


def make_screen_env(region=None, executor: Optional[ActionExecutor] = None, **kwargs) -> BrawlStarsEnv:
    """Live env capturing the screen (needs `pip install mss`).

    Use this for an emulator or a scrcpy-mirrored phone window; pass the window's
    pixel region.
    """
    return BrawlStarsEnv(source_factory=lambda: ScreenSource(region), executor=executor, **kwargs)


def make_phone_env(serial=None, executor: Optional[ActionExecutor] = None, **kwargs) -> BrawlStarsEnv:
    """Live env against a PHYSICAL Android phone (e.g. Pixel 10) over adb.

    Capture = adb screencap (slow but zero-setup); control = AdbExecutor. For a
    faster loop, mirror with scrcpy and use make_screen_env instead. Both use the
    same reward/state pipeline. See docs/ANDROID_CONTROL.md.
    """
    from .actions import AdbExecutor
    ex = executor or AdbExecutor(serial=serial)
    return BrawlStarsEnv(source_factory=lambda: AdbScreencapSource(serial),
                         executor=ex, tick_seconds=kwargs.pop("tick_seconds", 0.1), **kwargs)
