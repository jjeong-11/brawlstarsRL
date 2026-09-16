# models/ — one directory per brawler

Weights are per brawler. A single policy has to average over every brawler in
the game, and they want contradictory things over the *same* observation
vector: Edgar wants to close distance, Piper wants the opposite, Barley wants a
wall in between. One network trained on all of them learns the mean of those
contradictions, which is nobody's strategy.

```
models/
  shelly/
    policy.zip          RecurrentPPO weights for this brawler
    vecnormalize.pkl    the reward normaliser — part of the model, not a side
                        file: resuming without it restarts the running reward
                        statistics from scratch and the value function sees a
                        step change in scale it did not cause
    checkpoints/        periodic saves during this brawler's run
    meta.json           what produced these weights
  piper/
    ...
```

## Nothing is trained yet

`brawlers/registry.py` resolves in this order:

1. `models/<id>/policy.zip` — trained for this brawler
2. `brawlstars_move.zip` — the shared base policy at the repo root
3. nothing — the caller falls back to random headings

So every brawler currently loads the base policy, and the UI says so in as many
words. The fallback is not scaffolding to be deleted later: it stays useful as
the initialisation each new brawler's run forks from, so brawler #40 does not
start from random weights.

## Adding a brawler's weights

Nothing to register. Drop the file at `models/<id>/policy.zip` (ids are the
`id` field in `brawlers/roster.json`) and it is picked up on the next session —
`slot()` checks the filesystem each call, so there is no cache to invalidate.

```python
from brawlers import default_registry
reg  = default_registry()
slot = reg.prepare("piper")     # creates models/piper/ + checkpoints/
print(slot.policy_path)         # where training should save
```

## Weights are not transferable across designs

`rl/train.py` keeps `checkpoints_polar/` and `checkpoints_move/` apart because
the action head changes shape between control designs, and a stale resume would
either crash or — worse — load a policy whose action indices mean something
completely different. The same rule applies here: if the action space or
observation layout changes, every `models/*/policy.zip` predates it. The
registry catches the load failure and falls back to random headings rather than
taking the session down, but that is a diagnostic, not a fix. Retrain.

## Not in git

`models/*/` is gitignored alongside the other checkpoint directories — the
weights are large, regenerable, and specific to one machine's calibration.
This README and the directory itself are kept.
