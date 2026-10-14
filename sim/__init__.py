"""
sim/
====

An offline Solo Showdown simulator, so that strategy changes can be MEASURED.

The live loop produces ~36k steps an hour and a match takes minutes, so on the
phone "did this make the bot win more?" is a question that takes days of play
to answer -- and every answer is confounded by matchmaking, map rotation and
perception noise. This package answers it in minutes, on the 71 real arena
layouts from `perception/mapdb.py`.

It is deliberately NOT a reimplementation of the bot. It replaces only the two
ends of the pipeline -- the screen and the touch screen -- and drives the real
middle: `rl/enemy_tracker.py`, `rl/tactics.py`, `rl/path_planner.py` (world
map, A*, gas escape), `rl/combat.py`, and `rl/env.encode_observation` for
policies. A synthesised `GameState`, terrain grid, gas grid and camera delta go
in; a joystick vector and a `CombatDecision` come out. What it measures is
therefore the decision layer that actually ships.

    sim/arena.py     map + bushes + boxes + gas + projectiles + regen
    sim/bots.py      the nine opponents
    sim/harness.py   GameState/terrain/camera synthesis + the agent stack
    sim/evaluate.py  CLI: compare controllers over many seeded matches
    sim/gym_env.py   Gymnasium env with the live observation layout, for
                     pretraining a policy before it ever touches the phone

What it is NOT: a faithful copy of Brawl Stars. Brawler stats are class
archetypes, projectiles are simple, bots are scripted. Treat absolute win
rates as meaningless and DIFFERENCES between controllers on the same seeds as
the signal.
"""
