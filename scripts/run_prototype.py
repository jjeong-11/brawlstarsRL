#!/usr/bin/env python3
"""
Prototype RL loop: perception -> GameState -> reward, on real footage.

This is the working prototype. It runs the full perception stack on a recorded
match (or the live screen), converts each tick into a GameState, scores it with
the RewardCalculator, and prints the per-tick reward + a running episode return —
exactly the signal an RL agent would learn from. No model training required to
watch the reward function come alive on real gameplay.

    python scripts/run_prototype.py --video media/testvideos/test_game1.mp4
    python scripts/run_prototype.py --video media/testvideos/test_game3.mp4 --start 15 --duration 25
    python scripts/run_prototype.py --screen                 # needs `pip install mss`

Once an agent is trained, rl/env.py wraps this exact loop in the Gymnasium API and
actions.ActionExecutor closes the loop by sending inputs back to the game.
"""
import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from perception.liveLoop import VideoSource, ScreenSource, LivePerception  # noqa: E402
from rl.state import adapt_live_state  # noqa: E402
from rl.rewards import RewardCalculator, RewardConfig  # noqa: E402
from rl.kills import KillAttributor  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="play a recorded match as a live feed")
    src.add_argument("--screen", action="store_true", help="capture the real screen (needs mss)")
    ap.add_argument("--region", type=int, nargs=4, metavar=("L", "T", "W", "H"))
    ap.add_argument("--start", type=float, default=0.0, help="seconds to skip into a --video")
    ap.add_argument("--fps", type=float, default=10.0, help="target ticks per second")
    ap.add_argument("--duration", type=float, default=None, help="stop after N seconds")
    args = ap.parse_args()

    source = VideoSource(args.video) if args.video else ScreenSource(args.region)
    if args.video and args.start:
        source.start -= args.start   # jump forward into the recording

    perception = LivePerception()
    calc = RewardCalculator(RewardConfig())
    calc.reset()
    killer = KillAttributor()
    killer.reset()

    started = time.time()
    ticks = 0
    misses = 0
    prev_screen = None

    print("tick   time   reward   return   | state")
    try:
        while True:
            if args.duration and time.time() - started > args.duration:
                break
            t0 = time.time()
            frame = source.grab()
            if frame is None:
                if not getattr(source, "live", False):
                    print("recording ended")
                    break
                misses += 1
                if misses > 90:
                    print("live source: too many dropped frames, stopping")
                    break
                time.sleep(0.05)
                continue
            misses = 0

            live, _ = perception.tick(frame)
            kills = killer.update(live)
            state = adapt_live_state(live, tick=ticks, kills_this_tick=kills)
            result = calc.compute(state)
            ticks += 1

            screen = live["game_state"]["state"]
            if screen == "in_match":
                bd = {k: round(v, 3) for k, v in result.breakdown.items() if abs(v) > 1e-9}
                print(f"{ticks:4d} {time.time()-started:6.1f}s {result.total:+7.3f} "
                      f"{calc.episode_return:+8.2f}  | HP:{state.health} cubes:{state.cube_count} "
                      f"left:{state.players_left} gas:{state.in_gas} enemies:{len(state.enemy_positions)} {bd}",
                      flush=True)
            elif screen != prev_screen:
                rank = live["game_state"].get("rank")
                print(f"{ticks:4d} {time.time()-started:6.1f}s {'':7} {calc.episode_return:+8.2f}  "
                      f"| -- {screen}{f' (rank {rank})' if rank else ''} --", flush=True)
            prev_screen = screen

            sleep_for = (1.0 / args.fps) - (time.time() - t0)
            if sleep_for > 0:
                time.sleep(sleep_for)
    finally:
        source.close()
        print("\n" + "=" * 60)
        print(f"episode return: {calc.episode_return:+.2f}  over {ticks} ticks")
        print("reward breakdown totals:")
        for k, v in sorted(calc.episode_breakdown.items(), key=lambda kv: -abs(kv[1])):
            print(f"    {k:14s} {v:+.3f}")
        if not calc.episode_breakdown:
            print("    (no in-match reward events captured in this window)")


if __name__ == "__main__":
    main()
