# Suggested improvements

Ordered by expected value per hour of work, not by how interesting they are.
Everything here is a suggestion — the pathfinding, cleanup and tracing items
from this pass are already done and are listed at the bottom for reference.

A recurring theme: **this project's bottleneck is samples, not algorithms.** A
live phone loop at ~10 steps/s produces ~36k steps per hour. PPO on a 60-dim
observation typically needs low hundreds of thousands of steps to show real
policy structure, so a run that would be trivial in a simulator is an overnight
job here. That makes anything which raises sample throughput, or reduces the
number of samples needed, worth far more than anything which makes the policy
marginally better per sample.

---

## 1. Highest value

### 1.1 The reward is dense but the credit assignment is ~2 seconds late  — **DONE**

`Controls.hold_ms` is clamped to the tick period and `AdbExecutor` warns about
queueing, which is the right instinct — but the loop still has the frame the
policy acted on being *one step older* than the frame it is rewarded against,
plus adb touch latency, plus the game's own animation lag. Under a far-tier
commitment (26 decisions) an action can be credited with a reward caused by a
decision two seconds earlier.

Worth measuring directly: fire a known action (a hard left swipe) and count
frames until the anchor's world velocity actually changes. If it is more than
1–2 ticks, shift the reward by that many steps before it reaches PPO. This is a
handful of lines and it is the single most likely reason a run "trains" without
improving.

### 1.2 Episodes are extremely long and the terminal reward is huge  — **DONE**

`win` is +50 and `placement` +3, against `survive` at +0.09/tick. A five-minute
match at 10 Hz is ~3000 steps, so with `gamma=0.995` the discount over a full
episode is `0.995^3000 ≈ 3e-7` — the win bonus is **mathematically invisible**
from the start of the match. The agent can only ever learn from the dense terms.

Two options, and they compose:

* Raise `gamma` to ~0.999 (horizon ~1000 steps rather than ~200).
* Re-express the terminal rewards as dense shaping. `placement` already has a
  natural per-step form: reward the *decrease* in `players_left` while alive.

Right now the shape of the reward function is telling the agent "survive and
collect cubes" and nothing else, which is fine — but it should be a deliberate
choice, not a side effect of the discount.

### 1.3 Run more than one phone

`SubprocVecEnv` with two or three devices is close to linear speedup and
requires no algorithmic change — `BrawlStarsEnv` already takes a `serial`. This
is the cheapest possible way to turn an overnight run into a two-hour one. The
main work is making `controls.json` per-serial.

### 1.4 Behaviour cloning from your own play  — **DONE**

You already have four gameplay recordings and a perception stack that turns
frames into `GameState`. What you do not have is the *action* that produced each
frame — but you can recover a usable approximation: the world-frame velocity
from `camera_tracker` gives the heading actually walked, and an ammo drop gives
an attack. That is enough to pretrain the policy head with supervised learning
before PPO ever touches the phone.

Even a mediocre initialisation is worth tens of thousands of live steps, and
live steps are the scarce resource.

---

## 2. Perception

### 2.1 Terrain profiles do not scale  — **DONE**

Six hand-calibrated HSV profiles, and `select_profile` refuses below 30%
coverage — at which point the planner loses all wall knowledge. Showdown rotates
skins, so this list will always be behind.

The structural fix is to stop classifying colours and start classifying
*texture*: a tiny CNN or even a random-forest on 8×8 patches, trained on the
frames you already have, would generalise across skins in a way that a hue band
cannot. Training data is nearly free — `calibrate_map.py` plus the existing
recordings.

A cheaper interim fix: derive the profile *automatically* per match from the
first few in-match frames by k-means, instead of matching against a fixed table.
`_auto_patches` already does most of this; it just is not wired into the live
path.

### 2.2 `ammo_known` is true on only ~15% of frames  — **DONE**

The code is admirably careful about this (`_gate_weapons` refuses to trust a 0
reading), but the underlying problem is that the ammo bar is only located
sometimes. Since the bar is drawn at a fixed offset from the player anchor, and
the anchor is tracked, the search window could be much tighter and much more
reliable. Getting this to ~90% would let the attack gate actually work, and 89%
of attacks were previously fired on an empty clip.

### 2.3 Kill attribution is heuristic  — **DONE**

`rl/kills.py` infers knockouts from `players_left` decreasing, which credits the
agent for kills it did not make. In Solo Showdown with 10 players that is a
large fraction. The on-screen defeat banner is the ground truth and is already
noted as the remaining gap in `state.MISSING_EXTRACTORS`.

---

## 3. RL design

### 3.1 The observation has no history  — **DONE**

60 dims, all describing the current instant except velocity and the previous
action. The agent cannot represent "I have been walking into this wall for a
second" or "an enemy was here and went into a bush". Cheapest fix by far is
SB3's `VecFrameStack` with 4 frames; a `RecurrentPPO` (sb3-contrib) is the
principled version but costs more per sample.

The planner status fields help a lot here, and adding two more would help more:
whether the last decision was overridden (gas escape), and how long the current
commitment has left in *seconds* rather than as a fraction.

### 3.2 `ent_coef` is still probably too high  — **DONE**

The docstring already tells this story well — 0.03 was sized for a 3.58-nat
action space and the polar space is 5.26 nats. At 0.01 the entropy bonus is
still ~0.05/step against a `survive` of 0.09/step, i.e. a third of the dense
signal is being paid for randomness. Worth annealing: 0.01 → 0.002 over the
first 200k steps.

### 3.3 Attack and super are learned, but auto-aim means they barely need to be  — **DONE**

Both are binary taps with auto-targeting. Given the gating in `_gate_weapons`
already refuses shots that cannot land, a scripted "always attack when an enemy
is visible and ammo is known" policy is likely within a few percent of optimal —
and would free the policy to spend its entire capacity on movement, which is the
part that is actually hard. Worth A/B testing: it removes two action dimensions.

### 3.4 Reward normalisation  — **DONE**

`VecNormalize(norm_reward=True)` is standard for a reward with terms spanning
0.09 to 50. Without it the value function spends most of its capacity on the
rare large terms.

---

## 4. Engineering

### 4.1 There is no test runner  — **DONE**

Three separate `scripts/test_*.py` files, each with a hand-rolled
`TESTS = [...]` list and a `main()` that counts failures. They work, and the
synthetic scrolling world in `test_waypoint.py` is genuinely good — but they
cannot be run as one suite, cannot be filtered, and CI cannot report them. Moving
to `pytest` is mostly mechanical (the assertions are already plain `assert`) and
would let `python -m pytest` be the one command.

### 4.2 `env.py` is doing too much  — **DONE**

847 lines covering the gym interface, menu navigation, wall-clock profiling, gas
caching, camera tracking and spatial fusion. The menu navigation in particular
(`_navigate_to_match`, the button constants, the tap rate limiting) is a
self-contained concern that would read better as `rl/menus.py`.

### 4.3 `.venv/` is 863 MB and committed to nothing

Correctly gitignored, but worth knowing it exists — `pip install torch` pulls
~800 MB of CUDA libraries that a CPU-only Mac never uses. `pip install
torch --index-url https://download.pytorch.org/whl/cpu` is ~200 MB.

### 4.4 Pin the dependencies

`requirements.txt` has no version bounds. `stable-baselines3` and `gymnasium`
have both made breaking API changes; a fresh clone in six months is not
guaranteed to run. `pip freeze > requirements.lock.txt` costs nothing.

---

## 5. Done in this pass

For reference, so the list above is not read as "nothing has been done".

| area | change |
| --- | --- |
| pathfinding | `rl/world_map.py`: persistent camera-registered occupancy + gas map, 3 screens wide |
| pathfinding | agent-footprint clearance costs + hard inflation, with automatic relaxation through narrow gaps |
| pathfinding | any-angle string pulling, made cost-aware so it cannot straighten a route back through gas |
| pathfinding | steering low-pass so a one-cell change in the A* frontier stops swinging the joystick |
| gas | Dijkstra-to-safe-ground escape, replacing the centroid vector that pointed *into* a closing ring |
| gas | asymmetric gas fusion (fast up, slow down) + extrapolation into unobserved ground |
| gas | gas decoupled from the terrain profile, so uncalibrated maps are no longer gas-blind |
| perf | A* inner loop on a flat list rather than numpy scalar indexing: 9.0 ms → 2.3 ms |
| debugging | `rl/debug_trace.py` + `--trace SECONDS` on both `train_rl.py` and `watch_live.py` |
| tests | 31 → 38, with a named regression test per failure mode above |
| hygiene | `.gitignore` inline-comment bug that left 229 data files tracked; 417 → 137 tracked files |
| hygiene | 611 MB of committed video stripped from git history |
| maps | `perception/mapdb.py` + `localize.py`: 71 arenas, absolute position, true borders |
| perception | ammo decoupled from the HP digit line via a learned anchor offset |
| perception | anchor selection seeded by its own previous position, not screen centre |
| RL | attack/super scripted out of the action space (192 -> 48 combinations) |
| RL | RecurrentPPO, gamma 0.999, ent_coef 0.004 + anneal, VecNormalize |
| RL | `--action-delay N` puts recent actions in the observation |
| eng | pytest suite under `tests/` (62 tests); menus split out of `env.py` |

## 6. Performance

Measured per in-match step, capture excluded (`media/fixtures/showdown.png`, forced in-match
path, 500 steps):

| stage | ms | note |
|---|---|---|
| perceive | 3.6 | LivePerception, stage-scheduled |
| terrain + gas + camera | 4.0 | gas recomputed every 3 steps |
| act (planner + A* + localiser) | 2.2 | |
| reward + encode | 0.0 | |
| **total compute** | **10.0** | ~100 fps if capture were free |

Capture dominates on a phone, so end-to-end throughput is set by transport:
scrcpy ~14 fps, `adb screencap` 3-6 fps. That is unchanged from before this
work — compute is still an order of magnitude below capture.

Two regressions were found and fixed while measuring:

* **The localiser retried forever.** A full search is ~25 ms and it re-ran
  every 20 ticks for the whole match on any arena it could not place. It now
  skips when the explored region has not grown (the search is deterministic
  given the patch, so identical evidence cannot give a different answer) and
  backs off exponentially. Step cost 14.4 ms -> 8.1 ms.
* **A\* searched the whole 144x81 map.** A waypoint is at most ~15 cells away,
  so most expansions were in the wrong direction. Bounding the search to the
  start/goal box plus 14 cells took `act` from 7.0 ms to 1.8 ms.

## 7. Still open after this pass

**Reading the defeat banner** is the remaining kill-attribution gap. The
heuristic is now gated on the agent having actually fired and on the enemy being
inside auto-aim range, which removes the obviously-wrong credits — but in a
10-player lobby, deaths that happen off-screen while the agent is mid-fight are
still credited. "<name> defeated <name>" appears top-centre for ~2s and is the
ground truth; the username classifier already exists, so the work is an ROI plus
a template match on "by". It needs footage to calibrate against, which this repo
does not have.

Until then, `kill` at +5.0 makes a false credit expensive. Consider ~2.0.

**A/B the scripted combat.** Section 3.3 is done, but the claim that it beats
the learned heads is a well-motivated guess, not a measurement. The test is the
same number of live steps with `MultiDiscrete([16,3])` + script versus the old
`[16,3,2,2]`.

