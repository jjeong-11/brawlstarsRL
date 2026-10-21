# Plan: learning from pro footage, then self-play

A response to the proposal from Sean's team (Team Electros players Doin,
Snoiy and possibly Memen; the PylaAI developer). The proposal: train an agent
on pro gameplay footage, move on to multi-agent self-play, and aim above
professional level, with noise robustness, update robustness and explainable
decisions. Each idea is checked below against published results and against
what this repository has already measured. The phased plan follows, with what
has been started.

---

## 1. Verdicts

| Idea | Viable? | Why |
|---|---|---|
| Learn from pro footage | **Yes, with the right design** | Labelling actions from video is a solved pattern (VPT). Brawl Stars makes it unusually cheap: the camera follows the player, so movement can be read from camera motion. Cloning whole pixel policies from tens of hours of play is not viable; using the footage to fit a structured decision layer is. |
| Multi-agent self-play | **Only in a simulator, and the sim decides how far it transfers** | The real game cannot be run at self-play scale: no API, real time only, one phone per agent, and the Terms of Service. OpenAI Five and AlphaStar trained inside the real game at very large compute. Here self-play means `sim/`, so its value depends on calibrating the sim to real data. The pro footage is exactly that data. |
| Above professional level | **Long-term stretch; not a short-term goal** | No public agent in a game this complex got there without in-game training at scale. A bot can beat pros at narrow mechanics (reaction, aim, dodging), but only once end-to-end latency is under human reaction time. Today ours is not: screen capture alone is 150-400 ms. Set measurable steps (section 4). |
| CV is slow and noisy | **Agreed; both have known fixes** | Slow: our bottleneck is `adb screencap`, not the vision. A scrcpy H.264 stream, which Pyla uses, gives frames in tens of ms. Noisy: a world-frame tracker plus training under a perception-noise model. `sim/` already randomises refresh rate, misses and read failures. |
| Noise robustness | **Yes** | Domain randomisation of perception noise in the sim (already built), plus temporal filtering (`rl/enemy_tracker.py`). Measured here: perfect versus noisy perception was within noise in the sim, so the decision layer is already fairly noise-tolerant. |
| Update robustness | **Partly** | Achievable for the decision layer if it consumes abstract features (entities, walls, gas, HP), not pixels. Map rotations already come from data (`perception/mapdb.py`), and brawler stats from data files (`brawlers/`). Perception will still need refreshing when the art changes. A learned detector, as Pyla uses, refreshes with a few hundred labelled frames; our hand-tuned colour profiles need re-calibration per skin. "Never retrain anything" is not realistic; "retrain only the perception, in hours" is. |
| Explainable decisions (xAI) | **Yes, by construction** | Keep the policy's outputs at the level of named options (escape gas, retreat, fight X, take cube, camp...) with per-option scores. `rl/tactics.py` already logs mode, reason and scores every tick. A learned policy that chooses among the same options stays explainable. A pixel-to-joystick policy would need post-hoc saliency, which is much weaker. |
| Use Pyla as a reference | **For ideas, not code** | Pyla: YOLO detectors exported to ONNX (in-game entities, tiles, close tiles), scrcpy plus PyAV for capture and control, and a scripted movement policy. It is licensed CC BY-NC 4.0, so copy no code; take the design lessons (learned detectors, stream capture). |

### The constraint that shapes everything: the Terms of Service

Supercell's Terms of Service prohibit "automation software, emulators, bots"
and unauthorised third-party software, and fair-play policy describes
permanent bans. Pyla runs against the live game regardless. This project's
`DISCLAIMER.md` already rules out live matchmaking. The plan therefore keeps
training in the simulator, uses footage the players recorded themselves and
agreed to share, and evaluates on a real phone only in Friendly Games (bots
or consenting players). Even Friendly Games are on Supercell's servers, so
use a test account, not a main account.

---

## 2. Why the design is "structured imitation", not end-to-end cloning

VPT (Baker et al., OpenAI 2022) is the reference for learning from unlabelled
video. Its inverse dynamics model (IDM), trained on about 2,000 hours of
labelled play, labelled about 70,000 hours of internet video. The IDM was two
orders of magnitude more data-efficient than behaviour cloning, because
inverting game mechanics is far easier than modelling human intent. LAPO
(ICLR 2024) goes further and recovers latent actions with no labels at all.

Two lessons for us:

1. **Labelling actions is the easy part.** In Brawl Stars it is easier still:
   - The camera keeps the player centred, so camera motion is the movement
     direction. That is a hand-written IDM that needs no training data.
   - Attacks show as ammo-bar drops and projectile spawns.
   - If the players record with Android's "Show taps" developer option on,
     every touch, including the joystick position, is drawn on the video.
2. **Behaviour cloning is the hard part, and it caps at the demonstrator.**
   Tens or hundreds of hours from three players is nowhere near VPT's 70,000.
   A pixel policy cloned from that will copy surface behaviour and compound
   its errors. This repository already tried cloning its own scripted brain
   from a 60-number observation: mean rank 5.71, against 4.72 for random
   movement.

So the footage feeds a **decision layer over perception features**:
- Where to go: the option, plus the destination.
- When to fight: the engage threshold against the target.
- How to fight: spacing and aim lead.

That layer has dozens of parameters, not millions, so it can be fitted from
hours of play, and it stays explainable. Self-play then improves it beyond
the demonstrators inside the calibrated sim.

```
pro footage --(camera-motion IDM, HUD reads, "show taps")--> (state features, action) pairs
      |                                                              |
      | calibrate sim: speeds, damage, gas timing, ranges            | fit decision layer
      v                                                              v
  sim/ (10-player Showdown) <------ self-play / PBT ------  option-level policy (explainable)
      |                                                              |
      +----------------------------> phone (scrcpy capture + held touch) in Friendly Games
```

---

## 3. Phased plan

### Phase 0: groundwork (done in this repo)
- Utility-based tactical brain; 10-player simulator on 66 real layouts with a
  perception-noise model; paired-seed evaluation. Mean place ~3.4 in the sim.
- Map localiser fixed: locks in 23/40 sim matches, 0 wrong.
- Held-touch control over scrcpy (`rl/scrcpy_control.py`).

### Phase 1: data and speed (started)
| Step | Output | Status |
|---|---|---|
| 1a | `tools/label_footage.py`: per-frame movement from camera motion, attack/super from HUD changes; "Show taps" reading still to do | **done (synthetic)** |
| 1b | `rl/scrcpy_video.py`: scrcpy H.264 capture (PyAV), latest-frame only; target < 80 ms frame age | **done; untested on a phone** |
| 1c | `sim/imitate.py`: decision-layer imitation from movement alone, validated in the sim | **partial** (see below) |
| 1d | Learned detector (YOLO to ONNX) trained on frames auto-labelled by the current perception, then corrected by hand; replaces the HSV profiles | next |

**Exit criteria:**
- Labeller: movement direction within 30° on ≥ 90% of moving frames, checked
  against touch logs.
- Capture: frame age ≤ 80 ms at ≥ 15 fps.
- Imitation fit: recovers held-out behaviour, with fitted-policy placement
  within 0.2 of the demonstrator's in the sim.

#### Phase 1 results so far

- **1a, labeller:** on synthetic footage with a known path (16 headings,
  stops, a clamped camera), the median heading error is 1.5°, 100% of moving
  frames are within 30°, no still frame reads as moving, and speed is
  recovered (~2.4 tiles/s). It still needs checking on real footage, which has
  animations, other brawlers and projectiles.
- **1b, capture:** decodes a real H.264 stream from a fake scrcpy server.
  Frame age on a phone is still unmeasured
  (`python -m rl.scrcpy_video --seconds 10`).
- **1c, imitation:** on 24 demonstration matches (22k ticks), the hidden
  demonstrator settings score highest of 32 candidates (agreement 0.629 vs
  0.618 for the best other, 0.537 for the defaults). Agreement falls with
  parameter distance from the truth (Spearman -0.71). So movement alone does
  carry the signal.
- **1c, blind fit:** a random-then-local search that never sees the truth
  recovered `camp_utility` (1% of range off), `engage_ratio` (14%) and
  `box_utility` (21%), but not the avoid/retreat settings (45-75% off). The
  fitted policy places 3.59 against the demonstrator's 3.92, the same as the
  defaults (3.60). **This misses the 0.2 exit criterion.**
- **Next for 1c:**
  - Weight ticks by how much the candidates disagree. Avoid and retreat only
    matter when an enemy is visible but not worth fighting, and those ticks
    are drowned out by easy agreement while exploring.
  - Add shorter-horizon labels.
  - Replace the random search with CMA-ES.
  - Score play-level statistics (fight rate, time near enemies, cubes) in
    addition to per-tick agreement.

### Phase 2: learn from the pros
- Run the labeller and perception over the footage to get (features, action)
  datasets, split per player.
- Fit the decision layer per player and pooled; report per-option agreement
  ("Doin engages at 1.3x HP advantage, the bot at 3.0x").
- Calibrate the sim from the same footage: move speed, projectile speed,
  damage, gas start, closing rate and final size per map, and bot behaviour
  statistics.
- **Exit:** a fitted policy beats the hand-tuned one in the calibrated sim
  and in Friendly Games against bots.

### Phase 3: self-play in the calibrated sim
- Feature-based, option-level policy initialised from the Phase 2 fit.
- Population training with past snapshots as opponents (OpenAI Five style:
  latest 80%, past 20%), with perception noise randomised.
- Keep the option interface so every decision still has a named reason.
- **Exit:** beats the Phase 2 policy in sim, with the advantage holding in
  Friendly Games.

### Phase 4: toward expert level
- Close the latency gap: capture plus perception plus decision under 150 ms.
- Fine-tune mechanics (aim lead, dodge) where the bot has a real edge.
- Evaluate against consenting players in Friendly Games, with logged
  explanations.

---

## 4. Measurable milestones instead of "beat pros"

| Milestone | Measured where |
|---|---|
| End-to-end latency ≤ 150 ms (capture + perception + decision) | phone, logged per tick |
| Perception: player, enemies, walls F1 ≥ 0.9 on held-out pro frames | labelled footage |
| Mean place ≤ 3.0 vs sim bots, then vs calibrated-sim bots | `sim/evaluate.py`, paired seeds |
| Beats bots in Friendly Showdown ≥ 50% wins | phone, test account |
| Decision agreement with pros ≥ 70% at option level | held-out footage |
| Competitive with consenting pros in Friendly Games | phone, logged |

---

## 5. What to ask the TE players to record

The cheapest possible labels come from how the footage is recorded.

1. **Turn on "Show taps"**: Settings > System > Developer options > Show
   taps. Every touch is drawn, so joystick direction, aim drags and button
   presses can all be read straight off the video.
2. Record at the device's native resolution, 30 fps or more, with the
   phone's own screen recorder (no streaming overlays or face cam).
3. Whole matches, including the lobby screen, so the brawler and map can be
   read.
4. A short note per match: brawler, mode, map, and final place.
5. Solo Showdown first, since it is what this repo targets. A few hours per
   player is enough for Phase 2's structured fit; more is better for sim
   calibration.
6. Clear consent for use in this project, and agreement on whether the
   footage may be published.

---

## 6. Sources
- Baker et al., *Video PreTraining (VPT)*, 2022: https://arxiv.org/abs/2206.11795
- Schmidt & Jiang, *Learning to Act without Actions (LAPO)*, ICLR 2024: https://arxiv.org/abs/2312.10812
- OpenAI Five / AlphaStar compute and league training, surveyed in https://arxiv.org/pdf/2408.01072 and https://arxiv.org/pdf/2111.07631
- PylaAI source (README, requirements: onnxruntime, adbutils, av): https://github.com/PylaAI/PylaAI
- Supercell Terms of Service: https://supercell.com/en/terms-of-service/
- Friendly Games with bots: https://guildorder.com/games/brawl_stars/wiki/friendly-games-and-club-practice
- scrcpy design (raw H.264 stream, latest-frame decoding): https://github.com/Genymobile/scrcpy/blob/master/doc/develop.md
