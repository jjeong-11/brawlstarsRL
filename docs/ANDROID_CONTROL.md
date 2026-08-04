# Connecting a physical Android phone (Pixel 10) 

This is the "hands" of the bot. The policy outputs a **player-centred waypoint**
plus attack/super flags. `rl/path_planner.py` converts the waypoint into the next
joystick direction and `rl/actions.py:AdbExecutor` turns it into real touches.

```
policy → waypoint + fire flags → local path planner → Intent(move / attack / super) → adb → game
```

Two channels to set up: **input** (sending taps/swipes) and **capture** (getting
frames back for perception). Both run over the same USB `adb` connection to the
Pixel — no emulator required.

> **Use normal ADB for v4 control.** Pixel 10 or Newest Android versions
> blocks raw `sendevent` input, and the PyPI `scrcpy-client` package bundles 
> a server too old for the phone. It is therefore optional and deliberately 
> not installed by `requirements.txt`. The waypoint controller reduces policy-level 
> direction jitter, but ADB itself still cannot hold the game joystick continuously
> between commands.

> You may install the current official `scrcpy` desktop app for manually mirroring
> the phone or faster window capture. Do not use v4's `--scrcpy` Python backend:
> its protocol server must exactly match its client and the published Python
> package is incompatible with this device.

---

## 1. Two ways to run on a physical phone

| Path | Input | Capture | Latency | Setup |
|---|---|---|---|---|
| **adb only** (`make_phone_env`) | `adb` taps/swipes | `adb exec-out screencap` | ~150–400 ms/frame | zero extra tools — start here |
| **scrcpy mirror** (`make_screen_env`) | `adb` taps/swipes | grab the scrcpy window with `mss` | low | install scrcpy; much faster capture |

Start with **adb only** to confirm everything works end to end, then switch capture
to **scrcpy** for a faster loop. The reward/state/perception pipeline is identical
either way — only the frame source changes.

---

## 2. Connect the Pixel 10

1. On the phone: **Settings → About phone →** tap **Build number** 7× to unlock
   Developer options.
2. **Settings → System → Developer options →** enable **USB debugging**.
3. Plug the Pixel into the computer with USB. Tap **Allow** on the "Allow USB
   debugging?" prompt (check "always allow from this computer").
4. Verify:
   ```bash
   adb devices
   # List of devices attached
   # 4A1B2C3D        device        <- your device's serial
   ```
   If it says `unauthorized`, re-accept the prompt on the phone. If nothing shows,
   try a different cable/port (some cables are charge-only).

Optional — go wireless (Android 11+): `adb tcpip 5555`, then
`adb connect <phone-ip>:5555`, then unplug USB. USB is lower latency, though.

If multiple devices are attached, note the serial and pass it everywhere as
`serial="4A1B2C3D"`.

---

## 3. Screen size, rotation, and raw touch coordinates

`adb shell input tap X Y` uses display **pixel** coordinates:

```bash
adb shell wm size          # e.g. Physical size: 1080x2400  (Pixel 10, portrait)
```

Brawl Stars runs **landscape**, so the usable surface is 2400×1080. Depending on
Android version, `input tap` may expect coordinates in the portrait frame even
while the game is landscape — if taps land in the wrong place, you'll fix it in
calibration (§6) by swapping/rotating X and Y.

To read exactly what the hardware reports (and to find precise button positions),
watch the touch stream while you tap the real controls yourself:

```bash
adb shell getevent -lt      # tap the attack button; note ABS_MT_POSITION_X / _Y
```

`AdbExecutor` auto-reads `wm size` and defaults to a landscape layout, so you
usually don't set the resolution by hand.

---

## 4. Sending inputs

`AdbExecutor` uses the built-in commands, through a **persistent `adb shell`** (one
long-lived shell, commands piped in) to avoid paying adb's process-spawn cost every
action:

```bash
adb shell input tap 2040 820                 # a tap (attack / super)
adb shell input swipe 360 820 360 650 200    # a 200ms drag (movement)
```

- **A tap does not move the character.** Movement is a *drag-and-hold* on the left
  joystick, so each move action is a `swipe` from the stick center toward the
  direction, held for `hold_ms`.
- `input` spawns a small JVM on the device (~50–150 ms), so it tops out around
  5–8 actions/sec. Fine for a first working bot; for competitive speed use scrcpy's
  control channel (swap `AdbExecutor._tap/_swipe` for it — everything above stays
  the same).

---

## 5. The six outputs → touches

`decode_action` produces an `Intent` (a move direction + attack/super flags).
`AdbExecutor` maps it via the pure, testable `intent_to_touches(intent, controls)`:

| Output | Touch action | Why |
|---|---|---|
| **up** | swipe move-center → center + (0, −radius), hold `hold_ms` | drag joystick up |
| **down** | swipe center → (0, +radius) | drag joystick down |
| **left** | swipe center → (−radius, 0) | drag joystick left |
| **right** | swipe center → (+radius, 0) | drag joystick right |
| **attack** | tap the attack button | auto-aims at nearest enemy — no direction |
| **super** | tap the super button | auto-aims at nearest enemy |

Move + attack/super issue together on the same tick (that's kiting).

---

## 6. Calibrate the control positions (do this once)

The button positions live in a `Controls` object (device landscape pixels).
`AdbExecutor` guesses them from the screen size, but you should verify:

1. Get into a match and screenshot the phone:
   ```bash
   adb exec-out screencap -p > hud.png
   ```
2. Open `hud.png` and read the pixel center of the **movement joystick**, the
   **attack button**, and the **super button** (the skull button `getSuper.py`
   reads). An example in-match screenshot is at `media/fixtures/hud.png`.
3. Build the executor with those coordinates and dry-run each control:
   ```python
   from rl.actions import AdbExecutor, Controls

   ex = AdbExecutor(
       serial="4A1B2C3D",
       controls=Controls(
           move_center=(360, 820), move_radius=180,
           attack_btn=(2130, 880), super_btn=(1950, 700),
           hold_ms=200,
       ),
   )
   ex.calibrate()          # taps attack, taps super, swipes the stick right
   # or test one at a time:
   ex.apply([4, 0, 0])     # move right
   ex.apply([0, 1, 0])     # attack
   ex.apply([0, 0, 1])     # super
   ```
   Watch the phone. If taps miss, your display coordinates are rotated — swap X/Y
   or subtract from width/height until a known tap lands, then apply the same
   transform to all positions.

> Tip: `getSuper.SUPER_ROI` already locates the super button as frame fractions —
> multiply those by your capture resolution for a good `super_btn` starting guess;
> the attack button sits just below/right of it.

---

## 7. Capture the frames

**adb only (zero setup)** — `make_phone_env` uses the built-in `AdbScreencapSource`
(grabs frames with `adb exec-out screencap`) and an `AdbExecutor`:

```python
from rl.env import make_phone_env
from rl.actions import AdbExecutor, Controls

env = make_phone_env(
    serial="4A1B2C3D",
    executor=AdbExecutor(serial="4A1B2C3D", controls=Controls(
        move_center=(360, 820), attack_btn=(2130, 880), super_btn=(1950, 700))),
    tick_seconds=0.1,
)
```

**scrcpy mirror (faster capture)** — mirror the phone to a window on your PC with
[scrcpy](https://github.com/Genymobile/scrcpy) (`scrcpy` after `adb devices` works),
then capture that window region with `mss` via `make_screen_env`, still using the
`AdbExecutor` for input:

```python
from rl.env import make_screen_env
from rl.actions import AdbExecutor, Controls

env = make_screen_env(
    region=(left, top, width, height),        # the scrcpy window on your screen
    executor=AdbExecutor(serial="4A1B2C3D", controls=Controls(...)),
    tick_seconds=0.05,
)
```

Whatever the source, perception → GameState → reward is unchanged — that's the
point of injecting the source and executor.

---

## 8. Put it together

```python
from stable_baselines3 import PPO
from rl.env import make_phone_env
from rl.actions import AdbExecutor, Controls

serial = "4A1B2C3D"
controls = Controls(move_center=(360, 820), attack_btn=(2130, 880), super_btn=(1950, 700))
env = make_phone_env(serial=serial, executor=AdbExecutor(serial=serial, controls=controls),
                     tick_seconds=0.1)

model = PPO("MlpPolicy", env, verbose=1)   # or load a model pretrained on recordings
model.learn(total_timesteps=50_000)        # a live game runs in real time
```

A live game can't be fast-forwarded, so **pretrain offline** on recorded matches
(`scripts/train_rl.py`, which uses `VideoSource`) first, then fine-tune live.

---

## 9. Latency and timing

- The loop runs at `tick_seconds` (0.1 → 10 ticks/sec target). Each tick = 1
  capture + perception + 1–2 input commands.
- `adb screencap` (~150–400 ms) is the bottleneck on the adb-only path → realistically
  3–5 ticks/sec. Good enough to validate learning.
- Move capture to **scrcpy** (fast window grab) and, if needed, input to scrcpy's
  control channel to reach 15–30 ticks/sec.

---

## 10. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `adb devices` shows `unauthorized` | Re-accept the USB-debugging prompt on the phone. |
| Empty / black frames | Some games block `screencap`; use the scrcpy path instead. |
| Taps land in the wrong spot | Display rotation — swap/rotate X,Y in `Controls` and recalibrate. |
| Character doesn't move | Movement must be a held `swipe`, not a tap (that's what `AdbExecutor` does); raise `hold_ms` / `move_radius`. |
| Very low tick rate | `screencap`/`input` overhead — switch to scrcpy for capture (and control). |
| Super never fires in-game | `super_btn` off — recheck against the HUD overlay / `getSuper`. |

---

### TL;DR

1. USB-debug the Pixel; `adb devices` shows it.
2. Screenshot a match, read the joystick / attack / super pixel positions.
3. `Controls(move_center=..., attack_btn=..., super_btn=...)` → `AdbExecutor(serial=..., controls=...)`; run `ex.calibrate()`.
4. `make_phone_env(serial=..., executor=<that AdbExecutor>)`.
5. Movement = held joystick drag; attack/super = taps (auto-aim). Done.
