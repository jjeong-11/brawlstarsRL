"""
rl — reinforcement-learning layer for the Brawl Stars Showdown bot.

    perception/liveLoop.py  ->  state.adapt_live_state  ->  GameState
    GameState               ->  rewards.RewardCalculator (the learning signal)
    actions.ActionExecutor  ->  the device
    env.BrawlStarsEnv       ->  Gymnasium glue for all of the above
    train.py                ->  Stable-Baselines3 PPO
"""

from .state import GameState, adapt_live_state, MISSING_EXTRACTORS
from .rewards import RewardConfig, RewardResult, RewardCalculator
from .kills import KillAttributor, KillConfig

__all__ = [
    "GameState", "adapt_live_state", "MISSING_EXTRACTORS",
    "RewardConfig", "RewardResult", "RewardCalculator",
    "KillAttributor", "KillConfig",
]
