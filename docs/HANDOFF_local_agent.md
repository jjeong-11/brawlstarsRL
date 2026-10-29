# Handoff: work that needs the phone or real footage

For a coding agent (or a person) running on the machine the Android phone is
plugged into. A cloud session built and tested everything below that can be
tested without a phone. What is left needs the phone, real recordings, or
both. Read `docs/PLAN_pro_imitation.md` first for the why; this file is the
how and the current state.

Branch: `claude/busy-feynman-jke4bv`. Phone used so far: serial
`57230DLCR000M4`, a 2424x1080 screen (perception uses the 1920x1080 play area
inside it).

---

## 0. Setup (once)

```bash
git checkout claude/busy-feynman-jke4bv && git pull
python3 -m venv .venv && source .venv/bin/activate      # macOS Homebrew Python needs this
pip install -r requirements.txt flask
brew install scrcpy tesseract                           # scrcpy >= 2.1
adb devices                                             # phone listed as "device"
python3 -m pytest -q                                    # everything should pass
```

Zsh does not treat `#` in pasted lines as a comment. Paste commands without
the trailing comments.

---

## 1. Verify the phone-side pieces (about 15 minutes, in a Friendly Game)

Use a **test account and Friendly Games only**. Supercell's Terms of Service
prohibit bots (see `DISCLAIMER.md`).

| # | Command | Pass if |
|---|---|---|
| 1.1 | `python3 -m rl.scrcpy_control --serial 57230DLCR000M4 --selftest` | attack taps, super taps, then the brawler walks right **smoothly** for ~1 s |
| 1.2 | `python3 -m rl.scrcpy_control --serial 57230DLCR000M4 --walk 1` | walks a square without stopping between ticks |
| 1.3 | `python3 -m rl.scrcpy_video --serial 57230DLCR000M4 --seconds 10 --save frame.png` | ≥ 15 fps decoded; `frame.png` is the game at 2424x1080 |
| 1.4 | `python3 scripts/check_perception.py --serial 57230DLCR000M4` (during a match) | terrain profile matched; anchor verified ≥ 40% |

If 1.1 fails with "could not connect": check `scrcpy --version` (needs 2.1 or
newer), or set `SCRCPY_SERVER_PATH` to the `scrcpy-server` file and
`SCRCPY_VERSION` to its version. If touches are ignored, the server may want
the screen size in another orientation; print `ex.ctl.screen` and compare
with `adb shell wm size`.

Record the numbers in `docs/PLAN_pro_imitation.md` (Phase 1 results: 1b frame
age).

---

## 2. Play with the bot and record it (the wall-stuck bug)

```bash
python3 scripts/run_webui.py          # http://127.0.0.1:5000
```

Set these in the web UI:
- Brawler: Shelly. Mode: **Control**.
- Options > Frame capture: **scrcpy video**.
- Options > Control backend: **scrcpy**.
- Tick **Record raw frames for debugging**.

Play 2-3 Friendly Showdown matches.

Then:
```bash
git add media/recordings && git commit -m "Recordings: bot on the phone" && git push
```

What to look for, in order of likelihood:
1. **Phantom gas, fixed in commit `0211e16`.** The player's own green
   ring, HP bar and name tag read as gas, so the brain kept escaping gas that
   was not there. If the log still shows `gas` mode with no cloud on screen,
   look at `decisions.jsonl` (`mode`, `reason`) and the gas grid on that
   frame: `python3 -m rl.debug_trace media/recordings/<run>/<frame>.jpg`.
2. **The arena border.** The fence around the map has gaps in the wall
   classification, and the ground beyond it reads as floor, so A* can route
   out of the arena. Check the trace image: red = wall. If the fence has
   holes or the outside is blue, that is it. Fixes to try:
   - (a) The map localiser (`perception/localize.py`; fixed in `ce83a04`)
     marks everything outside the arena blocked once it locks. Check
     `decisions.jsonl` `localized`.
   - (b) If it never locks on the phone, debug it with the recorded frames
     (it was only ever tested in the simulator).
3. **Wedged on a corner.** With the scrcpy backend the bump sensor should now
   work (movement is continuous). If it still wedges, the WIP in commit
   `e26d23b` (wedge detector, wide A* retry) is a starting point; it was
   reverted only because it did not help in the simulator.

---

## 3. Pro footage (Phase 2), when Sean's team sends it

Recording instructions for the players are in `docs/PLAN_pro_imitation.md`
section 5. The key one is Android's "Show taps".

```bash
# 1) perception over the videos, once (slow: ~0.1-0.7 s per frame on CPU)
python3 tools/fit_from_footage.py perceive footage/doin_*.mp4 --cache cache/doin --step 2
# 2) which settings explain the player best, and where the defaults disagree
python3 tools/fit_from_footage.py score --cache cache/doin
# 3) fit the decision layer (validated in the sim: fitted 3.73 vs demonstrator 3.71)
python3 tools/fit_from_footage.py fit --cache cache/doin --out doin_fit.json
# 4) simulator calibration numbers (moving speed, first gas, elimination curve)
python3 tools/fit_from_footage.py calibrate --cache cache/doin --out doin_calib.json
```

Sanity checks before trusting a fit:
- `perceive` should report most frames in-match. If it reports 0, the
  game-state OCR is not reading "Brawlers left". Run
  `python3 -c "import cv2,sys; sys.path.insert(0,'.'); from perception.getGameState import get_game_state as g; print(g(cv2.imread('frame.png')))"`
  on a frame.
- `calibrate`'s `moving_speed_tps_median` should be 2.0-2.9 tiles/s for most
  brawlers. If it is not, the 17-tiles-across-the-screen assumption is wrong
  for that video's resolution.
- Then evaluate the fitted settings in the sim against the defaults:
  `python3 -m sim.evaluate --controllers tactical --matches 320 --seed 7000 --tactics "" --tactics "<fitted k=v,...>"`.
  A fit that plays worse in the sim may still be right (the sim is
  uncalibrated). Next, test it on the phone in Friendly Games.

Not built yet (next steps):
- Reading "Show taps" circles (exact joystick and aim positions) in
  `tools/label_footage.py`.
- Feeding `calibrate`'s output back into `sim/arena.py`'s `ArenaConfig`
  (gas start, end and speeds).

---

## 4. Things a CPU-only machine can keep running

- `python3 -m sim.league --generations 30 --out league.jsonl`: self-play over
  decision settings. Validate any champion with paired `sim.evaluate` on 320
  seeds before adopting it; see section 5 for the first run's result.
- `python3 -m sim.search`: random search vs scripted bots (it plateaued at
  ~3.4 mean place).
- A GPU would allow a learned detector (Phase 1d). Train a small PyTorch
  model on frames auto-labelled by the current perception from
  `media/recordings/`, then export to ONNX. Avoid the AGPL-licensed YOLO
  trainer unless the repository's licence allows it.

---

## 5. State of the experiments (update this when you change something)

| Item | Result |
|---|---|
| Sim mean place, current defaults | ~3.36 (seeds 7000-7319, 320 matches) |
| Localiser in sim | locks 23/40 matches, 0 wrong maps, ~0.5 tile error |
| Footage labeller (synthetic) | median heading error 1.5°, 100% within 30° |
| Imitation fit (sim, movement only) | fitted 3.73 vs demonstrator 3.71 mean place |
| scrcpy control / video | pass against fake servers; **unverified on a phone** |
| Phantom gas fix | 1 -> 0 false gas cells on a clean frame; **unverified on a phone** |
| League self-play | see `docs/PLAN_pro_imitation.md`, Phase 3 |
