"""
rl/menus.py
===========

Getting from a finished match back into a new one.

Split out of `rl/env.py`, which had grown to cover the gym interface, menu
navigation, wall-clock profiling, gas caching, camera tracking and spatial
fusion in one 850-line file. Menu driving is the most separable of those: it
runs only between episodes, touches none of the RL machinery, and its whole
contract is "block until we are in a match again".

WHY THE POLL RATE AND THE TAP RATE ARE DIFFERENT NUMBERS
--------------------------------------------------------
They used to be one interval of 1.0-1.2s, which meant every screen transition
was noticed up to 1.2 seconds after it happened. Measured on a phone: 24.2s per
reset, 71% of wall clock, the large majority of it sleeping.

They are separate because they answer different questions. LOOK often, so a
transition is caught quickly. TAP rarely, so a button is never machine-gunned —
a second tap landing after the menu advances hits whatever is now underneath,
which is how you end up somewhere unexpected.
"""

from __future__ import annotations

import time

from perception.getGameState import get_game_state

# Menu button positions as screen fractions, measured on 2424x1080 reference
# screenshots (defeated.png / endMenu.png at the repo root).
EXIT_BUTTON_NORM = (0.538, 0.922)          # "Exit" on the defeated screen
PLAY_AGAIN_BUTTON_NORM = (0.7376, 0.9167)  # "Play Again" on the end menu
NAVIGATE_TIMEOUT = 180.0   # give up navigating after this many seconds
NAVIGATE_POLL = 0.3        # how often to LOOK at the screen
NAVIGATE_TAP_INTERVAL = 1.0  # minimum gap between taps on the SAME screen

BUTTONS = {"defeated": EXIT_BUTTON_NORM, "match_end": PLAY_AGAIN_BUTTON_NORM}


def navigate_to_match(source, executor, verbose: bool = False) -> bool:
    """Drive the menus back into a match. Returns True if we got there.

    defeated -> Exit, end menu -> Play Again, loading/unknown -> wait. Live
    sources only (a recording cannot react to taps); a no-op without a tappable
    executor.
    """
    if (source is None or not getattr(source, "live", False)
            or not hasattr(executor, "tap_norm")):
        return False

    # Pause background frame capture so the menu taps get the USB/adb channel to
    # themselves; continuous screencap otherwise queues behind big frame
    # transfers and makes restarting a match sluggish. grab() still works — it
    # captures synchronously while paused.
    pause = getattr(source, "pause", None)
    resume = getattr(source, "resume", None)
    if pause:
        pause()

    prof = {"grab": 0.0, "classify": 0.0, "sleep": 0.0, "polls": 0, "taps": 0}
    t_start = time.perf_counter()
    try:
        deadline = time.time() + NAVIGATE_TIMEOUT
        last_tap_at = 0.0
        last_screen = None
        while time.time() < deadline:
            poll_due = time.perf_counter() + NAVIGATE_POLL

            t0 = time.perf_counter()
            frame = source.grab()
            prof["grab"] += time.perf_counter() - t0
            prof["polls"] += 1

            if frame is not None:
                t0 = time.perf_counter()
                screen = get_game_state(frame)["state"]
                prof["classify"] += time.perf_counter() - t0
                if screen == "in_match":
                    return True

                # Re-tap while the screen is still showing: early taps during
                # the rank/trophy animation harmlessly miss, and a later one
                # lands once the button goes live. Rate-limited per screen so
                # fast polling does not turn into a burst of taps.
                if screen != last_screen:
                    last_tap_at = 0.0        # new screen, tap immediately
                    last_screen = screen
                target = BUTTONS.get(screen)
                if target and time.time() - last_tap_at >= NAVIGATE_TAP_INTERVAL:
                    executor.tap_norm(*target)
                    last_tap_at = time.time()
                    prof["taps"] += 1

            # Wait out the REMAINDER of the poll interval. The grab is a
            # synchronous screencap while capture is paused (a few hundred ms),
            # so sleeping a flat interval on top would roughly double the period.
            t0 = time.perf_counter()
            remaining = poll_due - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            prof["sleep"] += time.perf_counter() - t0
        print("WARNING: navigate_to_match timed out; continuing anyway")
        return False
    finally:
        if resume:
            resume()
        total = time.perf_counter() - t_start
        if verbose and total > 0.5:
            print(f"[navigate] {total:5.1f}s | {prof['polls']} polls, "
                  f"{prof['taps']} taps | grab {prof['grab']:.1f}s "
                  f"classify {prof['classify']:.1f}s sleep {prof['sleep']:.1f}s")
