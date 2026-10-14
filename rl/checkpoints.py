"""
rl/checkpoints.py
=================

Load a saved policy whichever algorithm wrote it.

`rl/train.py` writes RecurrentPPO checkpoints when sb3-contrib is installed and
plain PPO checkpoints otherwise -- and the shipped `brawlstars_move.zip` is a
plain PPO (MlpPolicy) file. Every loader used to assume one or the other:
`BrawlerRegistry.load_policy` tried RecurrentPPO first, failed on the shipped
file with an AttributeError, and silently fell back to RANDOM headings for
every brawler, while `rl/train.py`'s resume path did not catch that error at
all and crashed. One function, tried in order, used everywhere.
"""

from __future__ import annotations

import pathlib
from typing import Optional, Tuple


def algorithms():
    """Available SB3 algorithm classes, most specific first."""
    algos = []
    try:
        from sb3_contrib import RecurrentPPO
        algos.append(RecurrentPPO)
    except Exception:
        pass
    try:
        from stable_baselines3 import PPO
        algos.append(PPO)
    except Exception:
        pass
    return algos


def load_policy(path, env=None, device: str = "auto") -> Tuple[object, Optional[str]]:
    """(model, algorithm name) for the checkpoint at `path` (with or without
    .zip). Raises the last error if no algorithm can read it."""
    p = pathlib.Path(path)
    stem = p.with_suffix("") if p.suffix == ".zip" else p
    last = None
    for algo in algorithms():
        try:
            model = algo.load(str(stem), env=env, device=device)
            return model, algo.__name__
        except Exception as e:          # wrong algorithm for this file: try the next
            last = e
    if last is None:
        raise RuntimeError("stable-baselines3 is not installed")
    raise last


def is_recurrent(model) -> bool:
    return type(model).__name__ == "RecurrentPPO"


class PolicyStepper:
    """`predict()` with the LSTM state carried between calls when there is one.

    A recurrent policy asked for one action at a time with no state is a
    feed-forward policy with extra steps: it forgets everything between ticks,
    which is the one thing it was trained not to do.
    """

    def __init__(self, model, deterministic: bool = False):
        import numpy as np
        self.model = model
        self.deterministic = deterministic
        self.recurrent = is_recurrent(model)
        self._np = np
        self.reset()

    def reset(self) -> None:
        self._state = None
        self._start = self._np.ones((1,), dtype=bool)

    def __call__(self, obs):
        np = self._np
        if self.recurrent:
            action, self._state = self.model.predict(
                obs, state=self._state, episode_start=self._start,
                deterministic=self.deterministic)
            self._start = np.zeros((1,), dtype=bool)
        else:
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return [int(v) for v in np.asarray(action).ravel()[:2]]
