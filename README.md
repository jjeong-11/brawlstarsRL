# Brawl Stars RL Bot

A reinforcement-learning bot that plays Brawl Stars **Solo Showdown** by reading
the screen and choosing what to do.

## This bot is NOT intended to be a CHEAT or HACK

Unlike Brawl Stars Cheats, Brawl Stars RL Bot does NOT 
modify the game, read its memory, intercept network traffic, or reveal hidden
information. It observes only the pixels displayed on the screen, just as a human
player does, and sends standard movement and attack inputs through the 
operating system. The reinforcement learning policy decides actions from visual 
observations RATHER THAN following hardcoded if-then scripts like popular 
hacks/cheats used in ranked. The bot has no access to information 
unavailible to human players. It cannot see enemies in bushes, predict 
future events, or obtain exact game-state values from memory. All 
information is extracted from in-game pixels using computer vision, making 
its observations imperfect and sometimes inaccurate. 

In fact, the system operates at a disadvantage compared to human players. 
The reinforcement learning pipeline currently processes observations at 
approximately 10 frames per second introducing aprroximately hundred milliseconds of 
latency between observing the game and responding. On the other hand, humans percieve and react to visual updates occurring at 60-120 frames per second, giving them substantially faster reaction times. 

Although this project automates gameplay, I do NOT promote or encourage botting 
on team-gamemodes or using bots to push the Ranked Gamemode. The purpose
of this project is to explore reinforcement learning, computer vision, and 
autonomous decision making in a real-time game environment. It is intended as a research and educational project rather than a competitive tool. 

Please refer to DISCLAIMER.md for further information. 

## Folder layout

```
projectv2/
├── perception/          Computer-vision stack: read the screen -> game facts
│   ├── getAnchor.py         player position (green floor ring)
│   ├── getHealth.py         HP        getAmmo.py    ammo
│   ├── getCube.py           cube count (HUD)   getPickups.py  ground cubes
│   ├── getEnemies.py        enemies + boxes (uses username_classifier)
│   ├── getGas.py            is the player standing in the gas zone
│   ├── getSuper.py          super-charge level from the skull button (0..1)
│   ├── getTerrain.py        floor/wall/bush -> walkability grid for pathfinding
│   ├── mapdb.py             72 published arena layouts -> ground-truth occupancy
│   ├── localize.py          which arena is this, and where am I on it
│   ├── getGameState.py      match_end / loading / in_match + brawlers-left + rank
│   ├── digitReader.py       fast template digit OCR (+ digit_*.npz data)
│   ├── username_classifier.py  (+ username_classifier_weights.json)
│   ├── callTools.py         one-frame debug harness
│   └── liveLoop.py          LivePerception: runs the whole stack per tick,
│                            smooths noise, and emits one "live state" dict
├── rl/                  Reinforcement-learning layer  (NEW)
│   ├── state.py             GameState + adapt_live_state(live dict -> GameState)
│   ├── rewards.py           RewardConfig + RewardCalculator  ← the learning signal
│   ├── kills.py             KillAttributor (credits the agent's knockouts)
│   ├── combat.py            scripted attack/super (NOT learned - see below)
│   ├── menus.py             between-match navigation (Exit / Play Again)
│   ├── actions.py           action space + ActionExecutor (Logging / ADB phone)
│   ├── camera_tracker.py    frame-to-frame world scroll (keeps waypoints still)
│   ├── world_map.py         persistent, camera-registered occupancy + gas map
│   ├── path_planner.py      latched destination + A* over that map
│   ├── debug_trace.py       annotated "what did it decide, and where is it going"
│   ├── env.py               BrawlStarsEnv (Gymnasium) — glues everything
│   └── train.py             Stable-Baselines3 RecurrentPPO
├── scripts/             Entry points you actually run
│   ├── run_live.py          perception-only status loop
│   ├── run_prototype.py     perception -> reward loop on real footage  ← the prototype
│   ├── train_rl.py          PPO training
│   ├── watch_live.py        watch rewards + decisions without sending input
│   ├── measure_latency.py   how many ticks until an action reaches the screen
│   └── check_perception.py  PRE-FLIGHT: per-stage perception health check
├── tests/               pytest suite: `python -m pytest`
│   ├── test_planner.py      planner + world map + gas, in a scrolling sim
│   ├── test_rewards.py      reward engine
│   ├── test_mapdb.py        arena extraction sanity
│   └── test_localize.py     identification, and refusing to guess
├── tools/               Offline utilities (harvest templates, classifier, calibrate_map.py)
├── media/               showdownmaps/ (72 arena layouts) testphotos/ crops/
│                        gasphotos/ (labelled gas fixtures)
├── training_data/       username-classifier dataset
├── debugOutput/         annotated frames written at runtime (trace/ = decision traces)
├── requirements.txt     README.md      LICENSE
```

Everything the bot imports lives in the `perception/` and `rl/` packages; the
`scripts/` are thin launchers that put the repo root on the path so those
packages resolve no matter where you run them from.

## How it all connects

One game tick flows top to bottom:

```
   screen frame
      │
   perception/liveLoop.py  ── LivePerception.tick(frame) ──►  live state dict
      │        (anchor, hp, ammo, hud_cubes, enemies, boxes, cubes,
      │         in_gas, game_state{state, brawlers_left, rank})
      ▼
   rl/state.py  ── adapt_live_state(live) ──►  GameState   (one typed snapshot)
      │
      ├──►  rl/rewards.py   RewardCalculator.compute(prev → curr)  ──►  reward (+breakdown)
      └──►  rl/env.py       encode_observation(state)              ──►  obs vector
      │
   rl/env.py   BrawlStarsEnv.step(action) returns (obs, reward, terminated, truncated, info)
      │
   rl/train.py  Stable-Baselines3 PPO maximizes the reward, and emits the next action
      │
   rl/actions.py  ActionExecutor turns that action into taps/swipes  ──►  the game
      ▲                                                                      │
      └──────────────────────────  next frame  ◄─────────────────────────────┘
```

The **reward engine is the hub**: perception feeds it (through `GameState`), the
env calls it every step, PPO consumes its output, and the resulting action changes
the next frame. Because `env.py` takes an injected frame *source* and *executor*,
the same loop runs offline on a recorded match (`VideoSource` + `LoggingExecutor`)
or live (`ScreenSource` + `AdbExecutor`) with no code changes.

## The reward function (rl/rewards.py)

| Signal | Term | Default | Where the data comes from |
|---|---|---|---|
| Collect powercube | `+1.0` per cube | `cube_pickup` | `hud_cubes` (getCube) |
| Charge super (rises only on damage) | `+4.0 × Δ` | `super_charge_full` | `super_charge` (getSuper) |
| Knock out an enemy | `+5.0` | `kill` | `kills_this_tick` (rl/kills.py) |
| Survive a tick, zone-safe | `+0.09` | `survive_tick` | `is_alive` + `in_gas` |
| Improve placement | `+3.0` per player | `placement` | `brawlers_left` (getGameState) |
| Win the match | `+50.0` terminal | `win` | `rank == 1` (getGameState) |
| Move toward a box | `γΦ(s′) − Φ(s)` | `box_progress` | `boxes` (getEnemies) |
| Regain health | `+3.0 × (HP gained/bar)` | `health_regain` | `hp` (getHealth) |
| Take damage | `−3.0 × (HP lost/bar)` | `damage_taken` | `hp` (getHealth) |
| Fire attack / super | `−0.05` / `−0.10` per tap | `attack_cost` | the gated action |
| Die | `−30.0` terminal | `death` | `rank ≥ 2` |
| Stand in gas | `−0.5` per tick | `gas_tick` | `in_gas` (getGas) |

### Trigger discipline and healing

There were two behaviours the first version got badly wrong, both measured on a rollout:

| | before | after |
|---------
| attacks fired with an empty clip | 89% | **0%** |
| supers fired uncharged | 100% | **0%** |
| reward earned for regaining HP | +0.000 | pays `health_regain` |

**Firing was free and super was ungated**, so nothing discouraged holding the
trigger down. `env._gate_weapons` now blocks shots that provably cannot do
anything — no visible target, confirmed-empty clip, uncharged super — which
costs no samples to learn, and `attack_cost` handles the shots that are merely
*unlikely* to land. The cost is deliberately well under the value of a hit
(~+0.4 through `super_charge`), so the agent trims spray-and-pray without
becoming gun-shy.

**Healing was worth exactly zero.** Brawl Stars regenerates HP only when out of combat, so
healing is not an action — it is the result of disengaging. The engine only
looked at HP *losses*, so retreating had no upside while fighting always paid off
super charge and kills (unless it died :skull). `health_regain` is set **equal to** `damage_taken`, which
makes health a potential: a damage-then-heal cycle nets zero, so the agent
cannot farm it by getting hurt on purpose.

**The ammo gate is `ammo_known`-guarded for a reason.** `find_ammo_info`
returns `0` both for "empty clip" and "could not find the bar", and on real
footage it only reads successfully on **~8% of frames** (it needs the health bar
located first, and that path has since been improved to ~48%). Gating on the raw
count would block nearly every attack the
agent ever attempts. Super has no such problem — its ROI is a fixed HUD position
that is never occluded, so `0.0` there genuinely means uncharged.

### Why `box_progress` is potential-based

`cube_pickup` only pays at the moment of contact, which is very sparse to find by
random walking, so `box_progress` adds a dense gradient toward boxes. The obvious
implementation — "reward getting closer" — is farmable: step toward a box, step
back, repeat forever. So the term is instead
`F = γΦ(s′) − Φ(s)` with `Φ = β(1 − normalised distance to nearest box)`, which
[Ng, Harada & Russell (1999)](https://people.eecs.berkeley.edu/~russell/papers/ml99-shaping.pdf)
proves cannot change the optimal policy, and which telescopes around any closed
loop so oscillating earns nothing. Measured: 30 steps of oscillating scores
**−0.08**, 30 steps of real progress scores **+0.14**.

Three details that are easy to get wrong and are handled explicitly:

- **Picking a box up deletes it**, collapsing Φ on the exact tick you succeed —
  which would be a large penalty for doing the right thing. Shaping is suppressed
  on any tick where `cube_pickup` fires.
- **A jump in nearest-box distance means the box changed identity** (occluded,
  collected, newly detected), not that the player moved. Those ticks are skipped.
- **`RewardConfig.gamma` must match PPO's `gamma`** in `train.py`. Policy
  invariance only holds when they agree.

### Gas is handled in the planner, not the reward

There is deliberately no "moved toward gas" penalty. Gas is a **cost layer in the
A\* grid** (`getGas.gas_info(grid_rect=...)` → `terrain["gas"]`), so the agent
routes around the cloud from the very first frame at zero sample cost, instead of
spending tens of thousands of steps learning it from a reward. Destinations that
land in gas are relocated at latch time.

Crucially it is a *cost*, not a wall: sometimes the safe zone is only reachable
through the cloud, and an agent taught "never enter gas" would corner itself and
die in exactly the endgame where that is fatal. A\* takes the detour when one
exists and crosses when one does not.

### Staying off the map border

The zone closes **inward**, so the map edge is where gas arrives first — and
the planner had a bias straight towards it. A heading pointing off-map was
relocated to the nearest legal cell, which is by definition the border ring, so
destinations collapsed onto the edge. Measured from a spot near an edge, **46%
of all 48 heading/distance combinations landed in the outer 3 cells**, and
distinct destinations dropped from 48 to 43.

Two corrections, both planner-side:

- **`border_cost`** — out-of-bounds decoration segments as unwalkable, so
  "blocked area nearby" is a reliable proxy for "close to the map edge" with no
  map knowledge required. `_openness()` measures it and A\* pays a penalty for
  closed-in ground.
- **`min_destination_openness`** — a destination in closed ground is pulled back
  along the ray toward the player until it reaches open space, instead of being
  snapped to the nearest legal cell.

Plus **`gas_dilate_cells`**: gas is spread 2 cells before costing, so ground the
cloud is *about to* reach is already expensive. Costing only the visible cloud
means reacting once it is on top of you, which in the endgame is too late.

⚠️ The grid's outer ring is the **screen** edge, not the map border — mid-map
they are unrelated, and penalising it would block most long-range movement.
That is why the signal is openness rather than distance-to-frame-edge.

**Gas vs foliage is the trap in this whole subsystem.** Gas is a pale-green
puffy overlay; several maps have green bushes at the same hue. Measured across
four themes:

| | H | S | V |
|---|---|---|---|
| gas | 49–56 | **97–105** | **211–222** |
| bush | 48–100 | **166–255** | **47–233** |

Saturation and value separate them; hue does not. Getting this wrong is
invisible without ground truth — it cost this project two bugs (the gas band's
old `S ≤ 200` cap admitted bushes, and `purple_stone`'s bush class was
calibrated on gas puffs). `media/gasphotos/` holds five labelled fixtures, and
`tests/test_planner.py` asserts both that they classify correctly and that no
profile's bush box overlaps the gas box in all three channels.

Design points: terminals dominate so the agent chases the real objective; the
survival reward is tiny and unpaid in gas so hiding never wins; damage is scaled
to a fraction of the health bar so it means the same across brawlers; and every
noisy reading is sanitized (None carried forward, impossible jumps rejected) as a
second safety net on top of liveLoop's smoothing. `compute()` returns a labeled
breakdown so you can see exactly which term drives learning.

## Actions & control

The policy picks a **destination**, not a direction. The space is
`MultiDiscrete([16, 3, 2, 2])` = 16 headings × 3 commit distances × attack × super.
There's no aiming: attack and super are plain taps and Brawl Stars auto-targets the
nearest enemy, and they can co-fire with movement on one tick (kiting).

The three layers underneath:

| module | job |
| --- | --- |
| `perception/getTerrain.py` | segments floor / wall / bush → a 48×27 walkability grid (~0.8 ms) |
| `rl/camera_tracker.py` | phase-correlates consecutive frames → how far the world scrolled |
| `rl/world_map.py` | fuses those grids into one persistent 144×81 map that remembers |
| `rl/path_planner.py` | **latches** the destination on that map and A*s to it (~2.3 ms) |

Latching is the point. The camera follows the player, so a destination stored in
screen pixels retreats at exactly the player's walking speed and is never reached —
which makes a "waypoint" policy just a compass with 225 actions instead of 9. The
camera delta lets the planner hold one destination still while the agent walks
there over 8–26 decisions, so each action is a committed macro-move rather than a
twitch. While a commitment is live the movement heads are **ignored**; the
observation exposes `waypoint_active` / `progress` so the policy can tell which
steps its movement choice actually mattered on. Commitments break early on arrival,
timeout, a blocked route, gas, or an **enemy or box** first appearing.

The box interrupt matters more than it looks. Committing is the point of this
planner, but it costs something the old 8-direction policy didn't pay: the agent
can't react to what it only notices mid-commitment. Boxes appear as the camera
scrolls, and without an interrupt the agent walks straight past one for up to 26
decisions (~2.5s) — which visibly suppressed cube collecting. Interrupts fire on
the *transition* into view only, so commitments still hold while a box stays on
screen.

`AdbExecutor` then drives a **physical Android phone** (e.g. a Pixel 10) over `adb`:
movement = a held joystick drag, attack/super = taps. `rl/env.make_phone_env()` wires
capture (`adb screencap`) + control together. See
**[docs/ANDROID_CONTROL.md](docs/ANDROID_CONTROL.md)** for connecting the phone,
calibrating button coordinates, and the faster scrcpy path.

Run `python -m pytest tests/test_planner.py` to exercise the planner in a synthetic
scrolling world (arrival, wall routing, gas override, and a regression guard against
the un-latched behaviour).

### The map is persistent, and that fixes three things

`find_terrain` describes one frame in screen space. `rl/world_map.py` fuses those
frames into a single 144×81 map — three screens wide — kept registered to the world
by the same camera delta that latches waypoints. Three failures came from not having
that:

* **Wedging on walls.** A* plans for a dimensionless point on cells about as wide as
  the brawler, so the optimal route runs flush against wall faces and cuts outside
  corners exactly. The map keeps a distance transform of free space, so cells closer
  to a wall than the agent's half-width are removed from the graph and a soft cost
  pulls routes down the middle of gaps. The inflation *relaxes* if it would seal a
  legitimate one-cell doorway. Stuck detection is still there, but as a backstop.
* **Short-sightedness.** ~20% of the play area is permanently behind the HUD and was
  marked `unknown` every frame, so those cells were never resolved; a far waypoint
  that scrolled off screen had its goal clamped to the grid border. Unknown cells are
  now *skipped* rather than fused, so they fill in from later frames as the camera
  scrolls them out from under the buttons, and waypoints live in map cells so they
  stay valid off screen.
* **Walking into the gas.** See below.

### Why the agent used to walk into the smoke

Three independent causes, all fixed:

1. **The escape vector pointed the wrong way.** `getGas.safe_vector` is
   `player − gas_centroid`. Showdown's cloud closes inward as a **ring**, and the
   centroid of a ring is the middle of the *safe zone* — so "away from the centroid"
   points outward, deeper into the gas. It looked fine early (a partial cloud has an
   off-centre centroid) and was lethal late, and it overrode the entire A* plan
   whenever `in_gas` was set. Replaced by a Dijkstra outward from the player over the
   gas-weighted cost field that stops at the first genuinely safe reachable cell —
   correct for any cloud shape, because it asks *where is safe ground* instead of
   assuming the cloud is a blob.
2. **Gas was forgotten.** The cached grid was rolled by the camera delta with newly
   exposed edges filled with **zero**, so ground the agent had just fled came back
   into view labelled clean. Gas is now fused into the map asymmetrically — fast up,
   slow down — because the cloud never retreats, and it is extrapolated into ground
   the detector has not covered, because a cloud does not stop at the edge of the
   screen.
3. **Gas was gated behind terrain.** It was computed inside the `if terrain is not
   None` branch, so on any map without a colour profile the planner had no spatial
   gas at all. Gas is an engine overlay drawn identically on every map and `play_rect`
   finds the panel from the letterbox alone, so geometry and gas are now established
   independently of the profile.

### Knowing which arena you are on

`perception/mapdb.py` turns the 72 published layouts in `media/showdownmaps/`
into ground-truth occupancy grids, and `perception/localize.py` matches the
agent's accumulated map against them to recover absolute position. Once it
locks, unobserved ground comes from the real layout instead of a prior, and the
map border is known exactly rather than inferred from surrounding decoration.

Two findings worth keeping in mind:

* **The floor is identified by connectivity, not by area.** "The floor is the
  most common colour" is wrong on dense maze arenas, where the wall blocks cover
  more of the render than the ground does — those maps came out ~100% blocked.
  The floor is instead the largest *connected* single-colour region, which is
  true by construction: an arena you cannot walk across would be unplayable.
* **A single frame cannot identify a map, and fails confidently.** Matching one
  48×27 viewport scored 0.60–0.71 against *every* map in the database, with
  0.017 between first and second place. Measured discrimination against patch
  size (71 maps, 8% cell noise): 8×8 → 19/30 correct, 12×12 → 30/30. One
  viewport is about 8×8. So the localiser matches the *accumulated* world map
  and refuses to attempt identification below 12×12 — a wrong lock is
  unrecoverable, while waiting a few seconds costs nothing.

```bash
python -m perception.mapdb --build --inspect Gated_Community   # eyeball extraction
python -m perception.localize --selftest                       # 47/48, 0 wrong locks
```

### Shooting is scripted, not learned

Brawl Stars auto-aims a plain tap, so there is no aiming decision — the whole
content of "should I shoot" is *is there a target in range and do I have ammo*,
which the env was already enforcing by overriding the policy. Once the override
does the deciding, the policy's attack and super heads are decoration, so they
were removed: `MultiDiscrete([16, 3, 2, 2])` → `MultiDiscrete([16, 3])`, 192
combinations down to 48. Every sample now goes into movement, which is the part
that is actually hard. See `rl/combat.py` for what this gives up.

### Action latency

PPO credits the reward at step *t* to the action at step *t*. If the action does
not reach the screen for three ticks, every one of those credits lands on the
wrong decision, and no amount of reward shaping fixes it. Measure it first:

```bash
python scripts/measure_latency.py --serial <SERIAL>
python scripts/train_rl.py --live --serial <SERIAL> --action-delay 2
```

`--action-delay N` puts the last N actions into the observation. That is the
textbook fix rather than a hack: an MDP with a constant action delay is still
Markov provided the recent actions are part of the state.

### Seeing what it decided (`--trace`)

```bash
python scripts/train_rl.py --live --serial <SERIAL> --trace 2   # every 2 seconds
python scripts/watch_live.py --serial <SERIAL> --trace 2        # watch, send nothing
```

Writes an annotated frame to `debugOutput/trace/` plus one JSON record per decision
in `trace.jsonl`. The frame carries all three layers at once — the fused occupancy
and gas the planner actually used, the latched waypoint and the A* route to it, and
the joystick vector that came out — because from the outside a perception bug, a bad
destination and a bad route all look identical: a brawler walking into a wall. A path
that ends somewhere silly means the *destination* was wrong; a sensible destination
with a route hugging a wall means the *cost field* was wrong. The inset shows the
whole world map including the parts currently off screen, which is where the most
confusing failures live.

`watch_live.py --trace` loads the saved policy and runs the planner alongside you
without touching the phone, so you can stand next to a wall or in the gas on purpose
and see what it would have done.

### Checkpoints are not interchangeable

| directory | design | obs |
| --- | --- | --- |
| `checkpoints_polar/` | previous, `MultiDiscrete([16, 3, 2, 2])` | 60 |
| `checkpoints_move/` | current, `MultiDiscrete([16, 3])` | 60 (+5 per `--action-delay`) |

Earlier designs (`checkpoints/` for the 8-way v3, `checkpoints_waypoint/` for the
15×15 grid) have been deleted — their weights could never be loaded by this action
space anyway.

Both the action head and the input layer changed shape, so weights cannot transfer.
`train.py` writes each design to its own path and refuses a mismatched resume.

## What still needs building

The whole reward list now runs on real perception — including **super charge**
(`getSuper.py`) and **kill attribution** (`rl/kills.py`). What's left (declared in
`rl/state.MISSING_EXTRACTORS`):

- **mid-match death** — `getGameState` can't yet see the death/spectate screen
  (both recorded matches were wins), so `is_alive` is inferred from the final rank.
  Record a losing match to add that state.
- **kill banner (optional)** — reading the on-screen defeat banner would give exact
  kill attribution; `rl/kills.py` already provides a solid heuristic without it.

The control layer (`actions.AdbExecutor`) is functional over `adb` — you just
calibrate the button coordinates for your Pixel (see the Android doc).

## Running it

```bash
pip install -r requirements.txt        # + system tesseract for OCR

# watch the reward function run on a recorded match (the working prototype):
python scripts/run_prototype.py --video media/testvideos/test_game3.mp4 --start 12

# perception-only status loop:
python scripts/run_live.py --video media/testvideos/test_game1.mp4

# the whole suite (planner, rewards, arena extraction, localiser)
python -m pytest
python -m pytest tests/test_planner.py -k gas      # or a slice of it

# render one decision trace from a screenshot (no phone needed):
python -m rl.debug_trace showdown.png     # -> debugOutput/trace/trace_0000.jpg

# train PPO offline on a recording (plumbing / reward check):
python scripts/train_rl.py
```

### Finding out why the loop is slow

```bash
python scripts/train_rl.py --live --serial <SERIAL> --controls controls.json --profile 200
```

```
[env profile] 200 steps |   3.0 fps |  333.0 ms/step
    act (adb touch)            1.9 ms     1%
    pace (idle wait)           0.0 ms     0%
    capture (grab)           300.0 ms    90%     <- transport bound
    perceive                  25.0 ms     8%
    terrain+gas+camera         3.3 ms     1%
```

Whichever row dominates is the only one worth optimising, and **it is not
guessable from an offline benchmark**: offline the frames come from memory, so
capture costs nothing and compute looks like the whole story. On a phone
`adb screencap` is typically 150–400 ms and everything else is noise.

| dominant row | what to do |
|---|---|
| `act` | swipes are queueing — see below. Use `--sendevent`, which holds the touch instead of re-swiping |
| `pace` | the loop is idle-waiting, so lower `--tick-seconds` |
| `perceive` | raise the intervals in `liveLoop.STAGE_INTERVALS` |
| `capture` | change transport — `--scrcpy` is the only flag that swaps the capture source |

### The `hold_ms` trap

`input swipe x0 y0 x1 y1 <hold_ms>` **blocks the device shell for hold_ms**, so
issuing one per tick caps the loop at `1000/hold_ms` fps regardless of what you
ask for. The old 200 ms default capped it at **5 fps**.

The failure is deceptive, which is why it went unnoticed: commands go down a
persistent `adb shell` pipe, so a backlog first fills the 64 KB stdin buffer at
no measurable cost and only blocks once full. Measured on a real phone:

| tick | oversubscription | profiled `act` |
|---|---|---|
| 0.10 s | 2× | 1.1 ms — *looks fine, latency growing invisibly* |
| 0.05 s | 4× | **329 ms (84% of the step)** |

Halving the tick made the loop **slower**. And the hidden latency matters more
than the throughput: a growing backlog means the game runs an action many steps
after the policy picked it, so the reward is attributed to the wrong action.

`Controls.tuned_for_tick()` now clamps `hold_ms` to the tick budget
automatically and prints a note when it does, so swipes tile at a ~100% duty
cycle instead of queueing. **`--sendevent` avoids the problem entirely** — it
presses the joystick once and only MOVEs it to redirect, so movement costs
nothing per step and is genuinely continuous rather than re-swiped.

`--tick-seconds` is a **deadline, not a delay**: `_pace()` waits until the next
tick is due rather than sleeping a fixed amount on top of the work, so the
measured rate matches the requested one.

**Read `pace` first.** A measured phone profile came out at 98.8 ms/step with
71.3 ms (72%) of it idle — the loop was doing 27.5 ms of real work and waiting
out the rest of a 0.1 s tick. That headroom is free throughput:

| `--tick-seconds` | ms/step | fps | idle headroom |
|---|---|---|---|
| 0.10 (default) | 100 | 10.0 | 72% |
| 0.05 | 50 | 20.0 | 45% |
| 0.04 | 40 | 25.0 | 31% |

Hard ceiling is ~36 fps at 27.5 ms of work. Lowering the tick is safe:
`_planner_config_for()` rescales `commit_ticks` and `stuck_ticks` so
commitments stay fixed in **seconds** rather than decisions — at 0.05 s they
become `(16, 32, 52)`, still 0.8–2.6 s. Without that, halving the tick would
silently halve every commitment and undo the temporal abstraction while the
loop merely *looked* twice as productive.

**If SB3 reports far less than the in-match rate**, the difference is between
matches: end-of-match animations, menu navigation and matchmaking all happen
inside `reset()`, and SB3 charges that wall clock against every step. The
profile prints it as a separate line, and `--profile` also emits a
`[navigate]` breakdown per reset.

### Between-match navigation

`_navigate_to_match` used to tap a menu button and then sleep 1.0–1.2 s, which
fused two unrelated rates: how often it *looks* at the screen, and how often it
*taps*. Every transition was therefore noticed up to 1.2 s late, and a measured
reset cost 24.2 s — 71% of wall clock, almost all of it asleep.

They are now separate: **poll every 0.3 s, tap at most once per second per
screen**. Fast polling catches a transition quickly; the tap rate limit stops
that becoming a burst of taps, which would land on whatever is underneath once
the menu advances. The poll also waits out the *remainder* of its interval
rather than sleeping a flat amount on top of the synchronous screencap.

On a scripted 2.0 s menu timeline: **overshoot 1.32 s → 0.13 s**, same two taps.

The rest of a reset is the game's own end-screen animation and matchmaking,
which nothing here can shorten.

### Run this before every training session

```bash
python scripts/check_perception.py --serial <SERIAL>      # or --video clip.mp4
```

Every expensive mistake in this project has been a silent perception failure
that looked like a hard RL problem: terrain constants that fit one map and made
86% of another read as solid wall; a gas band that fired on bushes; an anchor
that locked onto crates so every HUD window searched the wrong place. None of
them raise an error — they surface as a flat training curve two days later.

`check_perception.py` puts a number on each stage and flags the low ones:

```
stage                         rate   status
terrain profile matched       100%   OK
anchor verified                49%   OK
HP read                        76%   OK
ammo read                      54%   OK
```

### Check the terrain segmentor on YOUR map first

Showdown rotates map skins with completely different palettes, and one set of
HSV bounds does not survive that. `getTerrain.py` ships five measured profiles
(`night_teal`, `purple_stone`, `magenta_crate`, `graveyard`, `starr_rail`) and
picks the best-scoring one per match. **A map matching none of them is the single most damaging failure
mode in this codebase** — before profiles existed, applying the wrong bounds
classified 1.2% of the frame, and since unrecognised pixels count as obstacles
that made 86% of the map "solid". The agent then walks into borders, because
A\* is routing around walls that do not exist.

```bash
python -m perception.getTerrain path/to/your_screenshot.png
# prints each profile's coverage and which one won
# writes debugOutput/terrain_debug.png — walls RED, bushes GREEN, floor BLUE,
# blocked planning cells outlined in white
```

Healthy numbers: **coverage 45-75%**, **blocked 30-50%**. If it says
`NO PROFILE MATCHED`, add your map — the calibrator does it by clustering, so
you never have to guess pixel coordinates:

```bash
python tools/calibrate_map.py your_screenshot.png --name my_map
# prints a ready-to-paste TerrainProfile; add it to PROFILES
python tools/calibrate_map.py shot.png --montage /tmp/m.png   # eyeball the clusters
```

Prefer clustering over hand-labelled patches. Mislabelling one patch silently
poisons a profile, and that already happened here once: `purple_stone`'s bush
class was calibrated on what turned out to be gas.

Two independent guards mean a bad profile degrades instead of lying:

- `MIN_PROFILE_COVERAGE` — no profile explains ≥30% of the frame → no grid.
- `MAX_BLOCKED_FRACTION` — the grid comes out >72% solid → no grid. This catches
  the case where the *sticky* profile choice survives a frame it no longer fits
  (a full-screen super, a death overlay, a heavy gas tint).

In both cases `find_terrain` returns `None`, and the planner falls back to
direct steering — which is far better than pathfinding over an imaginary maze.

### On the phone (Pixel 10)

```bash
# 1) copy the controls template and fill in YOUR button pixel coords:
cp controls.example.json controls.json     # then edit (see docs/ANDROID_CONTROL.md)

# 2) verify perception + calibration live BEFORE training (sends no actions):
python scripts/watch_live.py --serial <SERIAL> --save-preview
python scripts/watch_live.py --serial <SERIAL> --calibrate --controls controls.json

# 3) train live on the phone (one command):
python scripts/train_rl.py --live --serial <SERIAL> --controls controls.json
```

Offline mode is only for checking the pipeline — in a recording the agent's actions
can't change the frames, so real policy learning happens live on the phone.

**This design starts from scratch.** It writes to `brawlstars_move.zip` and
`checkpoints_move/`, so it will not touch — or try to resume from — the v3
`checkpoints/` or the 15×15 `checkpoints_waypoint/`. You do not need `--fresh`;
you need it only to abandon a `_polar` run and restart. Re-running the same
command resumes, and Ctrl+C saves first.

### What to watch in the first hour of training/running

```bash
tensorboard --logdir tb_logs
```
Run this command to see dedicated graphs to multiple functions. 

Beyond `rollout/ep_rew_mean`, the things that tell you the new machinery is
actually working live:

| Symptom | Likely cause |
|---|---|
| `Intent(...)` almost never says `(committed)` | anchor detection is failing, so the planner can't latch — check `watch_live.py` |
| `(blocked)` on most steps | terrain constants are wrong for this map — re-run the calibration above |
| Agent walks into walls | same, or `controls.json` joystick coords are off |
| Movement still jittery | commitments too short — raise `PathPlannerConfig.commit_ticks` |
| `box_progress` dominating the breakdown | lower `RewardConfig.box_shaping` (it is shaping, not an objective) |
