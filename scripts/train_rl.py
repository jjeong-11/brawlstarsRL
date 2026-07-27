#!/usr/bin/env python3
"""Train the PPO agent (thin wrapper so packages resolve).

    python scripts/train_rl.py                 # offline, learns on a recorded match
    python scripts/train_rl.py --live          # once capture + AdbExecutor are wired

Requires:  pip install stable-baselines3 gymnasium
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rl.train import main  # noqa: E402

if __name__ == "__main__":
    main()
