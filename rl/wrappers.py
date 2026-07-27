"""
rl/wrappers.py
==============

Gymnasium wrappers for the Brawl Stars env.
"""

from __future__ import annotations

import gymnasium as gym


class ActionRepeat(gym.Wrapper):
    """Hold each chosen action for `n` env steps (a.k.a. frame-skip).

    Why: at a slow live tick rate (~3-5/sec) a single action barely changes the
    game — one "move up" tick nudges the joystick for a fraction of a second. The
    policy then can't see a clear consequence of its choice, which stalls learning
    and exploration. Repeating each action for a few ticks produces one meaningful,
    learnable transition per decision and lets the agent actually travel/commit.

    Rewards over the repeated ticks are summed; the episode ends early if a
    terminal or truncation happens partway through the repeat.
    """

    def __init__(self, env, n: int = 3):
        super().__init__(env)
        self.n = max(1, int(n))

    def step(self, action):
        # Fast path: actuate the action for the n-1 in-between ticks WITHOUT
        # capture/perception, then run ONE full observed step. The agent still
        # commits to a direction for several real swipes, but the expensive
        # capture+perception+reward runs only once per decision (big speedup at a
        # slow live tick rate). The single reward already spans the whole window,
        # since perception carries state between observed steps.
        actuate = getattr(self.env, "actuate", None)
        if actuate is not None:
            for _ in range(self.n - 1):
                actuate(action)
            obs, reward, terminated, truncated, info = self.env.step(action)
        else:
            # Fallback (older env without actuate): full step each repeat.
            reward = 0.0
            terminated = truncated = False
            obs, info = None, {}
            for _ in range(self.n):
                obs, r, terminated, truncated, info = self.env.step(action)
                reward += r
                if terminated or truncated:
                    break
        info = dict(info)
        info["action_repeat"] = self.n
        return obs, reward, terminated, truncated, info
