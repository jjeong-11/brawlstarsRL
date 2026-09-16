"""
webui/runner.py
===============

One background session: capture -> perception -> policy -> planner -> trace.

This is the same tick `scripts/watch_live.py --trace` runs, restructured so a
web request can start and stop it and a browser can watch the annotated frames
as they are produced. It deliberately reuses `rl/debug_trace.DecisionTracer`
rather than drawing its own overlay — the whole value of the trace is that it
shows *the decision the trainer would make*, and a second renderer would drift
from the first the moment either changed.

Two modes, and the difference matters
-------------------------------------
observe   The policy is asked for an action and the planner plans a route, but
          the executor is never constructed. Nothing reaches the phone. You
          play; the agent decides alongside you. This is the default because it
          is the only mode where a bad route is *safe* to look at: on the phone
          a bad plan and bad perception look identical, since all you see is a
          brawler walking into a wall.

control   The same tick, plus `executor.apply(...)`, so the bot actually plays.

The mode is carried on the config and checked in exactly one place
(`_build_executor`), so "observe" cannot half-send anything: with no executor
object there is nothing to send with.

Episodes
--------
A session spans many matches, so this loop owns the same lifecycle
`rl/env.py:reset()` does, and for the same reasons. Getting it wrong is not
subtle:

  * Nothing taps **Exit** on the death screen, so the session sits on a menu
    forever believing it is still playing.
  * Nothing resets the fused world map, so the previous match's walls and gas
    stay on top of the new arena. The agent then routes around walls that are
    not there, in a map it has already left — and because the map is
    camera-registered, the stale layout also drags the coordinate frame with
    it, so positions look reset while the geometry does not.

The loop detects the boundary two ways — we saw the match end, or we are
suddenly in a match having not been — and `new_match()` rebuilds every piece of
carried state. What has to be rebuilt is documented there rather than here,
because that list has to stay in step with `env.reset()` and a copy in a module
docstring would not.

Threading
---------
The loop owns all the perception state and runs on its own thread. The web
layer only ever touches `status()` and `latest_jpeg()`, both of which read a
snapshot under a lock. Nothing in the perception stack is called from a request
handler, so a slow browser cannot stall the tick and a dropped connection
cannot leave a half-updated GameState behind.

The newest JPEG is kept in memory (`_jpeg`); frames are not written to disk
unless `save_traces` is on. Streaming from RAM keeps the UI live at tick rate
without the ring-buffer bookkeeping the on-disk tracer needs.
"""

from __future__ import annotations

import collections
import pathlib
import sys
import threading
import time
import types
import traceback
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np                                          # noqa: E402

from brawlers import (apply_overrides, default_abilities, default_registry,  # noqa: E402
                      profile_for)


# --------------------------------------------------------------------------- #
@dataclass
class SessionConfig:
    """Everything the UI can set before pressing Start."""

    brawler: str = "shelly"
    mode: str = "observe"                 # "observe" | "control"
    serial: Optional[str] = None          # adb device; None = the only one
    fps: float = 4.0                      # target ticks/second
    source: str = "phone"                 # "phone" | "video" (offline testing)
    video_path: Optional[str] = None
    controls_path: Optional[str] = None   # controls.json for control mode
    backend: str = "adb"                  # "adb" | "sendevent"  (control only)
    sendevent_orientation: str = "A"
    save_traces: bool = False             # also write debugOutput/trace/
    deterministic: bool = False           # policy.predict(deterministic=...)

    def validate(self) -> None:
        if self.mode not in ("observe", "control"):
            raise ValueError(f"mode must be observe or control, got {self.mode!r}")
        if self.source == "video" and not self.video_path:
            raise ValueError("source=video needs video_path")
        if self.fps <= 0:
            raise ValueError("fps must be positive")


class SessionRunner:
    """Runs one session on a background thread. Reusable: stop() then start()."""

    MAX_LOG = 200

    def __init__(self, registry=None) -> None:
        self.registry = registry or default_registry()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._jpeg: Optional[bytes] = None
        self._jpeg_seq = 0
        self._log: Deque[str] = collections.deque(maxlen=self.MAX_LOG)
        self._state: Dict[str, Any] = self._blank_state()

    # --- lifecycle -------------------------------------------------------- #
    def start(self, config: SessionConfig) -> Dict[str, Any]:
        config.validate()
        self.registry.get(config.brawler)          # raises on an unknown id
        if self.is_running:
            raise RuntimeError("a session is already running — stop it first")

        self._stop.clear()
        with self._lock:
            self._jpeg = None
            self._jpeg_seq = 0
            self._log.clear()
            self._state = self._blank_state()
            self._state.update(status="starting", config=_config_dict(config),
                               brawler=config.brawler, mode=config.mode)
        self._say(f"starting {config.brawler} in {config.mode} mode "
                  f"({config.source}, {config.fps:g} fps)")
        self._thread = threading.Thread(target=self._run, args=(config,),
                                        name="brawl-session", daemon=True)
        self._thread.start()
        return self.status()

    def stop(self, timeout: float = 8.0) -> Dict[str, Any]:
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout)
            if t.is_alive():
                # The loop checks the flag once per tick; a stuck adb screencap
                # is the realistic way to get here. Say so rather than hanging
                # the request forever.
                self._say("stop timed out waiting for the tick to finish")
        self._thread = None
        with self._lock:
            if self._state.get("status") not in ("error",):
                self._state["status"] = "stopped"
        return self.status()

    @property
    def is_running(self) -> bool:
        t = self._thread
        return t is not None and t.is_alive()

    # --- what the web layer reads ----------------------------------------- #
    def status(self) -> Dict[str, Any]:
        with self._lock:
            s = dict(self._state)
            s["running"] = self.is_running
            s["frame_seq"] = self._jpeg_seq
            s["has_frame"] = self._jpeg is not None
            s["log"] = list(self._log)[-40:]
        return s

    def latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._jpeg

    def wait_for_frame(self, last_seq: int, timeout: float = 5.0):
        """Block until a frame newer than `last_seq`. Returns (jpeg, seq).

        Polling at a fixed rate either lags the tick or burns CPU spinning on
        an unchanged frame; this hands each frame over once, as it appears.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._jpeg is not None and self._jpeg_seq != last_seq:
                    return self._jpeg, self._jpeg_seq
            if not self.is_running and last_seq != 0:
                break
            time.sleep(0.03)
        with self._lock:
            return self._jpeg, self._jpeg_seq

    # --- internals -------------------------------------------------------- #
    @staticmethod
    def _blank_state() -> Dict[str, Any]:
        return {
            "status": "idle",          # idle|starting|running|stopping|stopped|error
            "brawler": None,
            "mode": None,
            "model": None,             # ModelSlot.to_dict()
            "error": None,
            "tick": 0,
            "elapsed": 0.0,
            "fps": 0.0,
            "reward": 0.0,
            "episode_return": 0.0,
            "breakdown": {},
            "game": {},                # hp / cubes / super / left / gas / enemies
            "decision": {},            # intent label, waypoint, planner status
            "config": {},
        }

    def _say(self, line: str) -> None:
        stamped = f"{time.strftime('%H:%M:%S')}  {line}"
        with self._lock:
            self._log.append(stamped)
        print(f"[session] {line}", flush=True)

    def _publish(self, **fields) -> None:
        with self._lock:
            self._state.update(fields)

    def _publish_frame(self, jpeg: bytes) -> None:
        with self._lock:
            self._jpeg = jpeg
            self._jpeg_seq += 1

    # --- the loop --------------------------------------------------------- #
    def _run(self, cfg: SessionConfig) -> None:
        source = executor = tracer = None
        try:
            import cv2
            from perception.getGas import gas_info
            from perception.getTerrain import (GRID_H, GRID_W, ProfileSelector,
                                               find_terrain, play_rect)
            from perception.liveLoop import LivePerception
            from rl.actions import decode_action
            from rl.camera_tracker import CameraTracker
            from rl.combat import CombatConfig, CombatPolicy
            from rl.debug_trace import DecisionTracer, TraceConfig
            from rl.env import encode_observation
            from rl.kills import KillAttributor
            from rl.menus import navigate_to_match
            from rl.path_planner import PathPlannerConfig, WaypointPlanner
            from rl.rewards import RewardCalculator, RewardConfig
            from rl.state import adapt_live_state

            brawler = self.registry.get(cfg.brawler)
            if brawler.multi_body:
                # Same reason the Sirius recordings were deleted rather than
                # kept: the perception stack counts clones and summons as
                # players, so anchor, enemy count and kill attribution are all
                # suspect. Worth running, not worth trusting silently.
                self._say(f"WARNING: {brawler.name} is multi-body — clones/summons "
                          f"read as extra players, so anchor, enemy count and "
                          f"kill attribution are unreliable this session")

            # --- per-brawler priors (all empty today; see brawlers/profiles.py)
            prof = profile_for(cfg.brawler)
            reward_cfg = apply_overrides(RewardConfig(), prof.reward_overrides)
            combat_cfg = apply_overrides(CombatConfig(), prof.combat_overrides)
            planner_cfg = apply_overrides(PathPlannerConfig(), prof.planner_overrides)
            if not prof.is_default:
                self._say(f"applied {cfg.brawler} profile overrides")

            # --- what this brawler's attack can actually do ---------------- #
            # Range gates the shot and class decides auto vs manual aim; both
            # come from brawlers/abilities.py. None is fine — CombatPolicy then
            # behaves exactly as it did before any of this existed.
            ability = default_abilities().get(cfg.brawler)
            if ability is not None:
                r = (f"{ability.range_tiles:g}t ({ability.confidence})"
                     if ability.range_tiles is not None else "range unknown")
                self._say(f"{ability.brawler_class} · {r} · aim {ability.aim}"
                          + (f" — {ability.aim_reason}" if ability.aim_reason else ""))
            else:
                self._say(f"no ability data for {cfg.brawler} — "
                          f"global range gate, auto-aim")

            # --- weights -------------------------------------------------- #
            policy, slot = self.registry.load_policy(cfg.brawler)
            self._publish(model=slot.to_dict())
            self._say(slot.label)

            # --- frame source --------------------------------------------- #
            source = self._build_source(cfg)
            executor = self._build_executor(cfg)

            def new_match():
                """Everything that must NOT survive into the next match.

                This list is the counterpart of `rl/env.py:reset()` and has to
                stay in step with it. Each entry is carried state that is not
                merely stale but actively wrong once the arena changes:

                  planner.reset()   also resets `planner.world` — the fused
                                    occupancy and gas. Skipping it leaves the
                                    last match's walls layered over the new
                                    map, which is the "same environment after a
                                    reset" symptom.
                  camera            frame-to-frame correlation across a scene
                                    cut is meaningless, and the world map is
                                    registered off it, so a bad delta drags the
                                    whole coordinate frame.
                  profiles          the next match may be a different skin; a
                                    sticky profile would be applied to it.
                  LivePerception    holds temporal smoothing, so a fresh object
                                    rather than a reset method (same as env).
                """
                p = WaypointPlanner(planner_cfg)
                c = CameraTracker()
                pr = ProfileSelector()
                cb = CombatPolicy(combat_cfg, ability=ability)
                cb.reset()
                rc = RewardCalculator(reward_cfg)
                rc.reset()
                ka = KillAttributor()
                ka.reset()
                return types.SimpleNamespace(
                    perception=LivePerception(), calc=rc, killer=ka,
                    planner=p, camera=c, profiles=pr, combat=cb)

            m = new_match()

            # every_seconds=0 -> we drive capture/render ourselves each tick.
            trace_cfg = TraceConfig(every_seconds=0.0,
                                    write_jsonl=bool(cfg.save_traces))
            tracer = DecisionTracer(trace_cfg)

            self._publish(status="running")
            self._say("running — press Stop to end the session")

            started = time.time()
            tick = 0
            episode = 1
            misses = 0
            was_in_match = False
            frame_times: Deque[float] = collections.deque(maxlen=20)
            period = 1.0 / cfg.fps

            while not self._stop.is_set():
                t0 = time.time()

                frame = source.grab()
                if frame is None:
                    # A live feed drops frames; a video source returning None
                    # means the file ended.
                    if not getattr(source, "live", False):
                        self._say("video source exhausted")
                        break
                    misses += 1
                    if misses > 90:
                        self._say("too many dropped frames — is the phone "
                                  "connected and awake?")
                        break
                    time.sleep(0.1)
                    continue
                misses = 0

                live, _ = m.perception.tick(frame)
                kills = m.killer.update(live)
                state = adapt_live_state(live, tick=tick, kills_this_tick=kills)
                result = m.calc.compute(state)
                tick += 1

                screen = (live.get("game_state") or {}).get("state", "unknown")
                in_match = screen == "in_match"

                # --- episode boundary ------------------------------------- #
                # Two independent triggers, because there are two ways a match
                # ends and only one of them is ours to drive.
                #
                #   1. We saw the end: the results screen, or the mid-match
                #      death screen. In control mode we now drive the menus back
                #      into a match, which is what actually presses Exit.
                #   2. We are suddenly in a match having not been. Covers observe
                #      mode entirely — nobody asked us to tap anything, the human
                #      just played on — and also covers a menu transition we
                #      missed. This edge is the reliable one, so it is checked
                #      even when (1) already fired.
                if state.match_over or not state.is_alive:
                    self._say(f"match over ({screen}) after {tick} ticks — "
                              f"return {m.calc.episode_return:+.2f}")
                    self._publish(status="between-matches")
                    if executor is not None:
                        # defeated -> Exit, results -> Play Again. Blocks until
                        # we are back in a match or it times out; it pauses the
                        # capture thread itself so the taps get the adb channel.
                        navigate_to_match(source, executor, verbose=True)
                    m = new_match()
                    episode += 1
                    tick = 0
                    was_in_match = False
                    self._publish(status="running", episode=episode)
                    continue

                if in_match and not was_in_match:
                    if tick > 1:      # not the very first frame of the session
                        self._say(f"new match detected — clearing the world map, "
                                  f"camera and terrain profile")
                        m = new_match()
                        episode += 1
                        tick = 0
                        self._publish(episode=episode)
                        was_in_match = True
                        continue
                    was_in_match = True
                elif not in_match:
                    was_in_match = False

                fh, fw = frame.shape[:2]

                # --- terrain + gas (mirrors rl/env.py's per-step perception) --
                terrain = None
                if state.player_pos is not None:
                    try:
                        terrain = find_terrain(frame, player_pos=state.player_pos,
                                               selector=m.profiles)
                    except Exception:
                        terrain = None
                rect = terrain["rect"] if terrain else play_rect(frame)
                try:
                    gas = gas_info(frame, grid_rect=rect,
                                   grid_size=(GRID_W, GRID_H))["grid"]
                except Exception:
                    gas = None
                if terrain is None and rect is not None:
                    m.planner.world.set_geometry(rect, (GRID_W, GRID_H))

                dx, dy, ok = m.camera.update(frame, rect)
                camera_delta = (dx, dy) if ok else (0.0, 0.0)

                # --- the policy ------------------------------------------- #
                if policy is not None:
                    max_hp = m.calc._max_health or m.calc.config.default_max_health
                    obs = encode_observation(state, max_hp, (fw, fh),
                                             planner_status=m.planner.status())
                    action, _ = policy.predict(obs, deterministic=cfg.deterministic)
                    action = [int(v) for v in np.asarray(action).ravel()[:2]]
                    policy_source = slot.source
                else:
                    # Random still exercises relocation, A*, clearance and gas
                    # escape against real frames — it just says nothing about
                    # the policy, so the UI must show which case this is.
                    action = [int(np.random.randint(16)), int(np.random.randint(3))]
                    policy_source = "none"

                combat = m.combat.decide(state, frame_size=(fw, fh),
                                         world=m.planner.world)

                kwargs = dict(state=state, frame_size=(fw, fh), planner=m.planner,
                              terrain=terrain, camera_delta=camera_delta,
                              gas_grid=gas, combat=combat)
                if executor is None:
                    intent = decode_action(action, **kwargs)     # nothing is sent
                else:
                    intent = executor.apply(action, **kwargs)

                # --- the picture ------------------------------------------ #
                try:
                    annotated = tracer.render(frame, m.planner, state, intent=intent,
                                              reward=result, live=live, tick=tick)
                    ok_enc, buf = cv2.imencode(
                        ".jpg", annotated,
                        [cv2.IMWRITE_JPEG_QUALITY, trace_cfg.jpeg_quality])
                    if ok_enc:
                        self._publish_frame(buf.tobytes())
                except Exception as e:
                    # Never take the loop down for a drawing bug.
                    self._say(f"render failed: {e!r}")

                if cfg.save_traces:
                    tracer.capture(frame, m.planner, state, intent=intent,
                                   reward=result, live=live, tick=tick, force=True)

                frame_times.append(time.time() - t0)
                self._publish(
                    tick=tick,
                    episode=episode,
                    elapsed=round(time.time() - started, 1),
                    fps=round(len(frame_times) / max(1e-6, sum(frame_times)), 2),
                    reward=round(result.total, 3),
                    episode_return=round(m.calc.episode_return, 2),
                    breakdown={k: round(v, 3)
                               for k, v in result.breakdown.items() if abs(v) > 1e-9},
                    game=_game_summary(state, live),
                    decision=_decision_summary(intent, m.planner, action,
                                               policy_source, combat),
                )

                sleep_for = period - (time.time() - t0)
                if sleep_for > 0:
                    self._stop.wait(sleep_for)

            self._publish(status="stopped")
            self._say(f"stopped after {episode} match(es), {tick} ticks in this one "
                      f"— return {m.calc.episode_return:+.2f} "
                      f"{dict(m.calc.episode_breakdown)}")

        except Exception as e:
            self._publish(status="error", error=f"{type(e).__name__}: {e}")
            self._say(f"ERROR {type(e).__name__}: {e}")
            traceback.print_exc()
        finally:
            for obj, name in ((executor, "executor"), (source, "source"),
                              (tracer, "tracer")):
                if obj is None:
                    continue
                try:
                    obj.close()
                except Exception as e:
                    self._say(f"closing {name} failed: {e!r}")

    # --- construction helpers --------------------------------------------- #
    def _build_source(self, cfg: SessionConfig):
        if cfg.source == "video":
            from perception.liveLoop import VideoSource
            self._say(f"frame source: video {cfg.video_path}")
            return VideoSource(cfg.video_path)
        from perception.liveLoop import AdbScreencapSource
        src = AdbScreencapSource(cfg.serial)
        self._say(f"frame source: adb screencap"
                  f"{f' ({cfg.serial})' if cfg.serial else ''}")
        return src

    def _build_executor(self, cfg: SessionConfig):
        """The single place control mode differs from observe mode.

        Observe returns None, and with no executor object there is nothing that
        *could* send a touch — the safety is structural, not a flag checked in
        several places that one of them might forget.
        """
        if cfg.mode != "control":
            self._say("observe mode: no executor built — nothing is sent to the phone")
            return None

        from rl.actions import Controls
        controls_path = cfg.controls_path or (
            str(_ROOT / "controls.json") if (_ROOT / "controls.json").exists() else None)
        if controls_path is None:
            self._say("WARNING: no controls.json — using coordinates derived from "
                      "screen size. Calibrate first (docs/ANDROID_CONTROL.md) "
                      "or the taps will miss.")
        controls = Controls.from_json(controls_path) if controls_path else None

        if cfg.backend == "sendevent":
            from rl.sendevent_backend import SendeventExecutor
            flip_short, flip_long = ((False, True)
                                     if cfg.sendevent_orientation.upper() == "A"
                                     else (True, False))
            self._say("CONTROL MODE: sendevent executor — the bot is playing")
            return SendeventExecutor(serial=cfg.serial, controls=controls,
                                     flip_short=flip_short, flip_long=flip_long)
        from rl.actions import AdbExecutor
        self._say("CONTROL MODE: adb executor — the bot is playing")
        return AdbExecutor(serial=cfg.serial, controls=controls)


# --------------------------------------------------------------------------- #
def _config_dict(cfg: SessionConfig) -> dict:
    from dataclasses import asdict
    return asdict(cfg)


def _game_summary(state, live) -> dict:
    gs = (live or {}).get("game_state", {})
    return {
        "state": gs.get("state"),
        "hp": state.health,
        "cubes": state.cube_count,
        "ammo": state.ammo_count if state.ammo_known else None,
        "super": round((state.super_charge or 0.0) * 100),
        "players_left": state.players_left,
        "in_gas": bool(state.in_gas),
        "enemies": len(state.enemy_positions),
        "boxes": state.n_boxes,
        "alive": bool(state.is_alive),
        # Diagnostic, and the reason it is here: a WRONG anchor and a MISSING
        # anchor look the same on the picture but are different bugs.
        "anchor_source": state.anchor_source,
    }


def _decision_summary(intent, planner, action, policy_source, combat) -> dict:
    st = planner.status() if planner is not None else None
    wp = planner.waypoint_pixels() if planner is not None else None
    return {
        # Intent.move_label, e.g. "NE/far(committed)" or "S/near(GAS-ESCAPE)".
        "intent": str(getattr(intent, "move_label", "")),
        "aim_mode": str(getattr(intent, "aim_mode", "auto")),
        "aim": ([round(float(v), 3) for v in intent.aim]
                if getattr(intent, "aim", None) else None),
        "move": [round(float(v), 3) for v in (getattr(intent, "move", (0.0, 0.0)) or (0, 0))],
        "attack": bool(getattr(intent, "fire_attack", False)),
        "super": bool(getattr(intent, "fire_super", False)),
        "raw_action": list(action),
        "policy": policy_source,          # "brawler" | "base" | "none"
        "combat": [bool(combat[0]), bool(combat[1])],
        "combat_reason": str(getattr(combat, "reason", "")),
        "waypoint": [round(float(v)) for v in wp] if wp else None,
        "planner": {
            "active": bool(getattr(st, "active", False)),
            "blocked": bool(getattr(st, "blocked", False)),
            "replanned": bool(getattr(st, "replanned", False)),
            "escaping_gas": bool(getattr(st, "escaping_gas", False)),
        } if st is not None else {},
    }


def list_adb_devices() -> List[dict]:
    """Connected phones, for the device picker. Never raises."""
    import shutil
    import subprocess
    if shutil.which("adb") is None:
        return []
    try:
        out = subprocess.run(["adb", "devices"], capture_output=True, text=True,
                             timeout=6).stdout
    except Exception:
        return []
    devices = []
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2:
            devices.append({"serial": parts[0], "state": parts[1]})
    return devices
