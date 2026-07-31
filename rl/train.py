"""
rl/train.py
===========

Stable-Baselines3 PPO entry point. PPO maximizes the discounted sum of the
rewards in rewards.py.

    python scripts/train_rl.py                              # offline, plays a recording
    python scripts/train_rl.py --live --serial <SERIAL>     # live on a physical phone
    python scripts/train_rl.py --live --controls controls.json --serial <SERIAL>

Offline caveat: in a recording the agent's actions don't change the next frame,
so offline mode is for smoke-testing the plumbing and reward shaping — real policy
learning happens live on the phone. See docs/ANDROID_CONTROL.md.
"""

from __future__ import annotations

import argparse
import pathlib

from .env import make_video_env, make_phone_env
from .rewards import RewardConfig
from .actions import LoggingExecutor

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_DEFAULT_VIDEO = _ROOT / "media" / "testvideos" / "test_game1.mp4"


def make_env(live: bool = False, serial=None, controls_path=None, tick_seconds: float = 0.1,
             action_repeat: int = 1, use_scrcpy: bool = False, use_sendevent: bool = False,
             sendevent_orientation: str = "A", sendevent_device: str = None,
             profile_every: int = 0, trace_seconds: float = 0.0):
    reward_config = RewardConfig()   # spec defaults; tune here
    extra = {"trace_seconds": trace_seconds} if trace_seconds else {}

    if not live:
        # The recordings are gitignored (612MB, two of them over GitHub's 100MB
        # per-file limit), so a fresh clone will not have them. Nothing at
        # runtime needs them — say what is missing rather than failing inside
        # cv2 with an unhelpful error.
        if not _DEFAULT_VIDEO.exists():
            raise SystemExit(
                f"Offline mode needs a recording, and {_DEFAULT_VIDEO} is not "
                f"present.\n"
                f"Gameplay recordings are not committed (see .gitignore).\n\n"
                f"  * to train for real:   add --live --serial <SERIAL>\n"
                f"  * to smoke-test offline: drop any Brawl Stars screen "
                f"recording at that path\n"
                f"  * to check perception:  python scripts/check_perception.py "
                f"--image your_screenshot.png")
        env = make_video_env(str(_DEFAULT_VIDEO), reward_config=reward_config,
                             executor=LoggingExecutor(), tick_seconds=0.0, **extra)
    elif use_sendevent:
        # Smooth movement via raw multitouch (no scrcpy/PyAV needed).
        from .sendevent_backend import make_sendevent_env
        from .actions import Controls
        controls = Controls.from_json(controls_path) if controls_path else None
        orientation = sendevent_orientation.upper()
        if orientation not in {"A", "B"}:
            raise ValueError("sendevent_orientation must be 'A' or 'B'")
        # Matches SendeventExecutor.calibrate(): two possible landscape rotations.
        flip_short, flip_long = ((False, True) if orientation == "A"
                                 else (True, False))
        env = make_sendevent_env(serial=serial, controls=controls,
                                 reward_config=reward_config, tick_seconds=tick_seconds,
                                 flip_short=flip_short, flip_long=flip_long,
                                 device=sendevent_device,  # None -> auto-detect
                                 **extra)
    elif use_scrcpy:
        # Smooth path: scrcpy for held-touch movement + fast capture.
        from .scrcpy_backend import make_scrcpy_env
        from .actions import Controls
        controls = Controls.from_json(controls_path) if controls_path else None
        env = make_scrcpy_env(serial=serial, controls=controls,
                              reward_config=reward_config,
                              tick_seconds=min(tick_seconds, 0.05), **extra)
    else:
        # adb path: works everywhere, but movement is bursty (see --scrcpy).
        from .actions import AdbExecutor, Controls
        controls = Controls.from_json(controls_path) if controls_path else None
        # A movement swipe blocks the device for hold_ms, so it must fit inside
        # one tick or swipes queue and the loop caps out well below the
        # requested rate. See Controls.tuned_for_tick.
        if controls is not None:
            controls = controls.tuned_for_tick(tick_seconds)
        executor = AdbExecutor(serial=serial, controls=controls)  # None -> auto from screen
        env = make_phone_env(serial=serial, executor=executor,
                             reward_config=reward_config, tick_seconds=tick_seconds,
                             **extra)

    if profile_every:
        env.profile_every = int(profile_every)

    # Frame-skip: hold each action for N ticks so it produces a real, learnable
    # change. Guarded import so the no-gym offline dry run still builds the env.
    if action_repeat and action_repeat > 1:
        try:
            from .wrappers import ActionRepeat
            env = ActionRepeat(env, action_repeat)
        except Exception:
            print("(gymnasium not available — skipping action-repeat wrapper)")
    return env


def _default_controls_path(explicit):
    if explicit:
        return explicit
    guess = _ROOT / "controls.json"
    return str(guess) if guess.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="play a real match on the phone")
    ap.add_argument("--serial", default=None, help="adb device serial (needed if >1 device)")
    ap.add_argument("--controls", default=None,
                    help="path to controls.json (calibrated button coords); "
                         "defaults to ./controls.json if present")
    ap.add_argument("--tick-seconds", type=float, default=0.1, help="seconds per step (live)")
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--save-every", type=int, default=2000,
                    help="checkpoint every N steps (progress is safe if you stop)")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any saved model and start training from scratch")
    ap.add_argument("--ent-coef", type=float, default=0.01,
                    help="entropy bonus — higher = more exploration (default 0.01). "
                         "Was 0.03, which was sized for the old 9-direction action "
                         "space (3.58 nats). The polar space is 5.26 nats, so the "
                         "same coefficient buys 47%% more bonus: at 0.03 it pays "
                         "+0.152/step for randomness against a survive_tick of "
                         "+0.09, and a 114k-step run sat at 96%% of maximum "
                         "entropy — i.e. still uniformly random. The planner also "
                         "supplies commitment now, so far less entropy-driven "
                         "exploration is needed than the twitchy 8-way policy "
                         "required. Drop to 0.005 to sharpen faster.")
    ap.add_argument("--action-repeat", type=int, default=1,
                    help="repeat each action for N ticks (frame-skip). Default 1 = "
                         "off: the agent decides fresh every tick. NOTE: the "
                         "planner now provides commitment on its own (a waypoint "
                         "is held for 8-26 decisions), so this is far less "
                         "necessary than it was under the 8-direction policy. If "
                         "you do raise it, scale PathPlannerConfig.commit_ticks "
                         "DOWN by the same factor or commitments last N times "
                         "longer in wall-clock terms than intended.")
    ap.add_argument("--scrcpy", action="store_true",
                    help="use scrcpy for smooth movement + fast capture "
                         "(pip install scrcpy-client).")
    ap.add_argument("--sendevent", action="store_true",
                    help="smooth movement via raw multitouch over adb — no extra "
                    "deps. Recommended if scrcpy won't install.")
    ap.add_argument("--sendevent-orientation", choices=("A", "B"), default="A",
                    help="raw-touch orientation selected by calibrate_sendevent.py "
                    "(default: A)")
    ap.add_argument("--profile", type=int, default=0, metavar="N",
                    help="print a wall-clock breakdown of where each step's time "
                         "goes, every N steps (try 200). Tells you whether you are "
                         "capture-bound, perception-bound or idle-waiting — which "
                         "is the only way to know what is worth optimising.")
    ap.add_argument("--trace", type=float, default=0.0, metavar="SECONDS",
                    help="every SECONDS, write an annotated frame to "
                         "debugOutput/trace/ showing the route the planner "
                         "intends to walk: fused occupancy + gas, the A* path, "
                         "the latched waypoint, the joystick vector, the decoded "
                         "action and the reward breakdown. Try 2. This is how "
                         "you tell a perception bug from a bad destination from "
                         "a bad route -- they look identical from the outside.")
    ap.add_argument("--sendevent-device", default=None,
                    help="raw touch device, e.g. /dev/input/event2. Default: "
                         "auto-detect from `adb shell getevent -pl` (the node "
                         "advertising ABS_MT_POSITION_X/Y).")
    args = ap.parse_args()

    controls_path = _default_controls_path(args.controls) if args.live else None
    if args.live and controls_path is None:
        print("WARNING: no controls.json found — using auto coordinates from screen "
              "size. Calibrate first (see docs/ANDROID_CONTROL.md) or the taps may miss.")

    try:
        env = make_env(live=args.live, serial=args.serial,
                       controls_path=controls_path, tick_seconds=args.tick_seconds,
                       action_repeat=args.action_repeat, use_scrcpy=args.scrcpy,
                       use_sendevent=args.sendevent,
                       sendevent_orientation=args.sendevent_orientation,
                       sendevent_device=args.sendevent_device,
                       profile_every=args.profile, trace_seconds=args.trace)
    except RuntimeError as e:
        print("Could not start live env:", e)
        return

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.env_checker import check_env
    except Exception:
        print("stable-baselines3 / gymnasium not installed:\n"
              "    pip install stable-baselines3 gymnasium\n"
              "Env built OK; skipping training.")
        env.close()
        return

    # Each control design gets its own save name and checkpoint directory.
    #   checkpoints/           v3, MultiDiscrete([9, 2, 2]),      46-dim obs
    #   checkpoints_waypoint/  15x15 grid waypoints,              46-dim obs
    #   checkpoints_polar/     MultiDiscrete([16, 3, 2, 2]),      60-dim obs
    # Weights are not transferable between them (both the action head and the
    # input layer change shape), so they must never share a path — a stale
    # resume would either crash or, worse, silently load a policy whose action
    # indices mean something completely different.
    save_path = _ROOT / "brawlstars_ppo_polar"     # -> brawlstars_ppo_polar.zip
    ckpt_dir = _ROOT / "checkpoints_polar"
    ckpt_dir.mkdir(exist_ok=True)
    tb_dir = _ROOT / "tb_logs"                     # view: tensorboard --logdir tb_logs

    # Resume from the last save unless --fresh (so stopping never loses progress).
    if save_path.with_suffix(".zip").exists() and not args.fresh:
        print(f"Resuming from {save_path.name}.zip")
        try:
            model = PPO.load(str(save_path), env=env)
        except (ValueError, AssertionError, KeyError) as e:
            print(f"\nCould not resume: {e}\n"
                  f"The saved policy's shapes do not match this env "
                  f"(obs {env.observation_space.shape}, act {env.action_space}).\n"
                  f"That means {save_path.name}.zip predates a change to the "
                  f"action space or observation layout.\n"
                  f"Re-run with --fresh to start this design from scratch.")
            env.close()
            return
        model.tensorboard_log = str(tb_dir)       # keep logging on resume
        model.ent_coef = args.ent_coef            # let --ent-coef apply on resume
        reset_timesteps = False
    else:
        if not args.live:
            check_env(env)   # skip on live: check_env would fire real taps at the phone
        model = PPO("MlpPolicy", env, n_steps=1024, batch_size=256,
                    gamma=0.995, gae_lambda=0.95, ent_coef=args.ent_coef,
                    tensorboard_log=str(tb_dir), verbose=1)
        reset_timesteps = True

    # Periodic checkpoints -> checkpoints/. Progress survives even a hard crash.
    checkpoint = CheckpointCallback(save_freq=args.save_every, save_path=str(ckpt_dir),
                                    name_prefix="brawlstars_ppo")
    print(f"Training. ent_coef={args.ent_coef} action_repeat={args.action_repeat} | "
          f"checkpoints every {args.save_every} steps -> {ckpt_dir.name}/.")
    print(f"Watch learning:  tensorboard --logdir {tb_dir.name}   "
          "(look at rollout/ep_rew_mean and train/entropy_loss)")
    print("Ctrl+C saves and exits safely; re-run the same command to resume.")
    try:
        model.learn(total_timesteps=args.timesteps, callback=checkpoint,
                    reset_num_timesteps=reset_timesteps)
    except KeyboardInterrupt:
        print("\nInterrupted — saving current progress...")
    finally:
        model.save(str(save_path))
        env.close()
        print(f"Saved {save_path.name}.zip  (resume: same command | restart: add --fresh)")


if __name__ == "__main__":
    main()
