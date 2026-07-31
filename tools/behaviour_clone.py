#!/usr/bin/env python3
"""
tools/behaviour_clone.py
========================

Learn from a recording of YOU playing, before PPO ever touches the phone.

    python tools/behaviour_clone.py extract my_match.mp4 --out bc_data.npz
    python tools/behaviour_clone.py train bc_data.npz --out brawlstars_move.zip

WHY
---
This project's binding constraint is samples, not algorithms. A live phone loop
at ~10 steps/s produces ~36k steps an hour, and PPO on a 60-dim observation
needs low hundreds of thousands of steps before policy structure appears. So an
overnight run here is a five-minute run in a simulator.

A recording is the one source of experience that does not cost live phone time.
Even a mediocre initialisation is worth tens of thousands of live steps, and
live steps are the scarce resource.

THE PROBLEM: A RECORDING HAS NO ACTIONS
---------------------------------------
Frames are recoverable; the joystick input that produced them is not. But it
can be INFERRED, and the pieces already exist:

  * `rl/camera_tracker.py` measures how far the world scrolled between frames.
    The camera follows the player, so world scroll is the player's motion
    negated. That gives the HEADING actually walked, which is head 0.
  * Sustained motion in one direction implies a longer commitment, which is
    head 1. Distance tier is read from how long the heading held steady rather
    than from speed -- speed is roughly constant in this game, and what the
    tiers actually encode is how far ahead the human was thinking.

Attack and super are no longer part of the action space (`rl/combat.py`), so
they do not need recovering at all -- which removes the least reliable part of
this idea. Nothing here has to infer intent from an ammo counter.

WHAT THE LABELS ARE AND ARE NOT
-------------------------------
These are RECONSTRUCTED labels, not recorded ones. They carry two known errors:

  1. Motion is measured, intent is not. A player pushed off a wall, or knocked
     back, produces motion they did not choose. `--min-speed` drops frames with
     too little motion to read a heading from, which removes the worst of it.
  2. Camera scroll is only the player's motion while the camera is actually
     following. During a death cam or a spectate transition it is not, so
     frames outside `in_match` are dropped.

So this is a warm start, not a teacher. Expect it to help PPO leave the ground,
not to produce a good policy on its own.

BRAWLER CHOICE MATTERS
----------------------
Record with a SINGLE-BODY brawler. This project deleted its previous recordings
because they were played on Sirius, whose clones the perception stack reads as
additional players -- corrupting the anchor, the enemy count and every number
derived from them. No summons, no clones, no pets.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from math import atan2, hypot, pi

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402


# --------------------------------------------------------------------------- #
def heading_from_velocity(vx: float, vy: float, n_headings: int) -> int:
    """World-frame velocity -> the polar heading index the policy would emit.

    Heading 0 points RIGHT and indices advance clockwise on screen (y grows
    downward), matching `rl/path_planner.WaypointPlanner.target_for`. Getting
    this convention wrong would silently train the policy to walk mirrored.
    """
    angle = atan2(vy, vx)                    # -pi..pi, clockwise on screen
    if angle < 0:
        angle += 2 * pi
    return int(round(angle / (2 * pi) * n_headings)) % n_headings


def distance_tier_from_run(run_length: int, tiers=(4, 12)) -> int:
    """How long one heading was held -> which commitment tier that implies.

    Read from DURATION rather than speed. Walking speed is near-constant in
    Brawl Stars, so speed carries almost no information about intent, whereas
    "kept going the same way for two seconds" is exactly what the far tier
    means.
    """
    if run_length <= tiers[0]:
        return 0
    if run_length <= tiers[1]:
        return 1
    return 2


# --------------------------------------------------------------------------- #
def extract(video: str, out: str, max_frames: int = 0, min_speed: float = 1.5,
            stride: int = 1) -> int:
    """Run perception over a recording and write (observation, action) pairs."""
    from perception.liveLoop import LivePerception
    from perception.getTerrain import find_terrain, play_rect, ProfileSelector, GRID_W, GRID_H
    from perception.getGas import gas_info
    from rl.camera_tracker import CameraTracker
    from rl.path_planner import WaypointPlanner
    from rl.state import adapt_live_state
    from rl.rewards import RewardCalculator, RewardConfig
    from rl.env import encode_observation
    from rl.actions import N_HEADINGS

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"could not open {video}")

    perception = LivePerception()
    camera = CameraTracker()
    profiles = ProfileSelector()
    planner = WaypointPlanner()
    calc = RewardCalculator(RewardConfig())
    calc.reset()

    obs_rows, headings, speeds, prev_pos = [], [], [], None
    rect = None
    frames = skipped_state = skipped_slow = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        if stride > 1 and frames % stride:
            continue
        if max_frames and len(obs_rows) >= max_frames:
            break

        live, _ = perception.tick(frame)
        if (live.get("game_state") or {}).get("state") != "in_match":
            skipped_state += 1
            continue

        fh, fw = frame.shape[:2]
        live["frame_size"] = (fw, fh)
        state = adapt_live_state(live, tick=frames, frame_size=(fw, fh))
        if state.player_pos is None:
            skipped_state += 1
            continue

        terrain = None
        try:
            terrain = find_terrain(frame, player_pos=state.player_pos, selector=profiles)
        except Exception:
            pass
        rect = terrain["rect"] if terrain else (rect or play_rect(frame))
        try:
            gas = gas_info(frame, grid_rect=rect, grid_size=(GRID_W, GRID_H))["grid"]
        except Exception:
            gas = None

        dx, dy, cam_ok = camera.update(frame, rect)
        # World scroll is the player's motion negated: walk right, the world
        # slides left.
        vx, vy = (-dx, -dy) if cam_ok else (0.0, 0.0)
        speed = hypot(vx, vy)
        if speed < min_speed:
            skipped_slow += 1
            prev_pos = state.player_pos
            continue

        max_hp = calc._max_health or calc.config.default_max_health
        obs_rows.append(encode_observation(state, max_hp, (fw, fh),
                                          velocity=(vx, vy),
                                          planner_status=planner.status()))
        headings.append(heading_from_velocity(vx, vy, N_HEADINGS))
        speeds.append(speed)
        prev_pos = state.player_pos

    cap.release()
    if not obs_rows:
        raise SystemExit(
            "no usable frames.\n"
            "  * frames outside 'in_match' are dropped (menus, death cam)\n"
            "  * frames slower than --min-speed are dropped (no readable heading)\n"
            f"  read {frames} frames: {skipped_state} not in-match/no anchor, "
            f"{skipped_slow} too slow")

    # Distance tier from run length: how long each heading was held.
    tiers = np.zeros(len(headings), np.int64)
    i = 0
    while i < len(headings):
        j = i
        while j + 1 < len(headings) and headings[j + 1] == headings[i]:
            j += 1
        tiers[i:j + 1] = distance_tier_from_run(j - i + 1)
        i = j + 1

    obs = np.asarray(obs_rows, np.float32)
    act = np.stack([np.asarray(headings, np.int64), tiers], axis=1)
    np.savez_compressed(out, obs=obs, act=act)

    print(f"read {frames} frames -> {len(obs)} usable samples")
    print(f"  dropped: {skipped_state} not in-match / no anchor, "
          f"{skipped_slow} below --min-speed")
    print(f"  heading histogram: {np.bincount(act[:, 0], minlength=N_HEADINGS).tolist()}")
    print(f"  tier histogram:    {np.bincount(act[:, 1], minlength=3).tolist()}")
    print(f"wrote {out}  (obs {obs.shape}, act {act.shape})")
    return len(obs)


# --------------------------------------------------------------------------- #
def train(data: str, out: str, epochs: int = 12, batch: int = 256,
          lr: float = 3e-4) -> None:
    """Fit the policy's action head to the recovered labels, then save it.

    Deliberately writes a normal SB3 checkpoint, so `scripts/train_rl.py` picks
    it up as an ordinary resume and PPO continues from it. There is no separate
    "pretrained" code path to keep in sync.
    """
    import torch
    from torch import nn

    d = np.load(data)
    obs, act = d["obs"], d["act"]
    print(f"{len(obs)} samples, obs dim {obs.shape[1]}")

    from rl.train import make_env
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    env = make_env(live=False)
    venv = DummyVecEnv([lambda: Monitor(env)])
    if obs.shape[1] != venv.observation_space.shape[0]:
        raise SystemExit(
            f"observation width mismatch: data has {obs.shape[1]}, env expects "
            f"{venv.observation_space.shape[0]}.\n"
            f"The dataset was extracted under a different --action-delay or a "
            f"different observation layout. Re-extract it.")

    try:
        from sb3_contrib import RecurrentPPO as Algo
        policy_name = "MlpLstmPolicy"
    except Exception:
        from stable_baselines3 import PPO as Algo
        policy_name = "MlpPolicy"
    model = Algo(policy_name, venv, verbose=0)

    policy = model.policy
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    obs_t = torch.as_tensor(obs, device=model.device)
    act_t = torch.as_tensor(act, device=model.device)
    n = len(obs_t)

    for epoch in range(epochs):
        perm = torch.randperm(n, device=model.device)
        total = 0.0
        for k in range(0, n, batch):
            idx = perm[k:k + batch]
            dist = policy.get_distribution(obs_t[idx])
            # MultiDiscrete: one categorical per head, summed log-likelihood.
            logp = dist.log_prob(act_t[idx])
            loss = -logp.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        print(f"  epoch {epoch + 1:2d}/{epochs}  NLL {total / n:.4f}")

    model.save(out)
    env.close()
    print(f"\nwrote {out}")
    print("Continue with PPO from here:\n"
          "    python scripts/train_rl.py --live --serial <SERIAL>\n"
          "(it resumes from this checkpoint automatically -- do NOT pass --fresh)")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="recording -> (obs, action) dataset")
    e.add_argument("video")
    e.add_argument("--out", default="bc_data.npz")
    e.add_argument("--max-frames", type=int, default=0)
    e.add_argument("--stride", type=int, default=1,
                   help="use every Nth frame (a 60fps recording is 6x denser "
                        "than the 10Hz loop the policy will actually run at)")
    e.add_argument("--min-speed", type=float, default=1.5,
                   help="px/frame of world scroll below which the heading is "
                        "not readable and the frame is dropped")

    t = sub.add_parser("train", help="dataset -> a checkpoint PPO can resume")
    t.add_argument("data")
    t.add_argument("--out", default="brawlstars_move.zip")
    t.add_argument("--epochs", type=int, default=12)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--lr", type=float, default=3e-4)

    args = ap.parse_args()
    if args.cmd == "extract":
        extract(args.video, args.out, args.max_frames, args.min_speed, args.stride)
    else:
        train(args.data, args.out, args.epochs, args.batch, args.lr)


if __name__ == "__main__":
    main()
