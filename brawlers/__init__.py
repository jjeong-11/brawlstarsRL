"""
brawlers/
=========

The roster, one weights slot per brawler, and the hook for per-brawler priors.

    from brawlers import default_registry, profile_for

    reg  = default_registry()
    slot = reg.slot("piper")          # models/piper/policy.zip -> falls back to base
    model, slot = reg.load_policy("piper")

See `registry.py` for why weights are per brawler and `profiles.py` for what
goes in the scripted-advantage layer.
"""

from .registry import (Brawler, BrawlerRegistry, ModelSlot, default_registry,
                       DEFAULT_BASE_POLICY, DEFAULT_MODELS_ROOT, DEFAULT_ROSTER)
from .profiles import BrawlerProfile, PROFILES, apply_overrides, profile_for
from .abilities import (AUTO, MANUAL, Ability, AbilityBook, LONG_RANGE_TILES,
                        default_abilities)

__all__ = [
    "Brawler", "BrawlerRegistry", "ModelSlot", "default_registry",
    "DEFAULT_BASE_POLICY", "DEFAULT_MODELS_ROOT", "DEFAULT_ROSTER",
    "BrawlerProfile", "PROFILES", "apply_overrides", "profile_for",
    "Ability", "AbilityBook", "default_abilities", "AUTO", "MANUAL",
    "LONG_RANGE_TILES",
]
