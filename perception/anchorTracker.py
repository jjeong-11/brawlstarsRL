"""
perception/anchorTracker.py
===========================

Temporal validation for `getAnchor.find_player_position`.

WHY
---
The anchor is the root of the whole perception stack: HP, ammo and the cube
counter are all searched RELATIVE to it, and the planner uses it as the origin
for every waypoint. When it is wrong, everything downstream is wrong -- and it
fails silently, because a false anchor still returns a perfectly well-formed
(x, y, radius).

Measured over 159 in-match frames from the four recordings:

    HP read succeeded  ->  anchor at play-rect fraction (0.50, 0.50),
                           5% further than 0.30 from centre
    HP read failed     ->  anchor at play-rect fraction (0.54, 0.70),
                          34% further than 0.30 from centre

i.e. the failures are not "the HP search is weak", they are "the anchor was
somewhere else entirely" -- clustered low on the screen, where the detector
latches onto HUD chrome and shadows. The radius told the same story: within a
single match it ranged 25-151px, which is impossible for a fixed-size ring
under a fixed camera.

WHAT THIS DOES
--------------
`getAnchor` was changed to return NO anchor rather than a guess when it cannot
verify a readable HP number above the ring. That made it far more accurate
(29% -> 79% of returned anchors are usable) but it now declines on many frames.

This fills those gaps by COASTING: a brawler does not vanish between frames, so
the last verified anchor, a few frames old, is a much better estimate than
nothing. Coverage comes back without reintroducing wrong answers.

A deliberately light sanity check runs on top, using two physical facts:

  1. THE RING IS A CONSTANT SIZE within a match, so a detection at wildly
     different radius from the running median is suspect.
  2. THE CAMERA FOLLOWS THE PLAYER, so he cannot teleport between frames.

These are set loose on purpose. An EARLIER, STRICTER VERSION OF THIS FILE MADE
THINGS WORSE -- measured 29% -> 18% HP read rate -- because it rejected good
detections on the strength of a history built from bad ones. The detector is
now the trustworthy part; the tracker's job is to fill gaps, not to second-guess
it. Anything it rejects it must be able to recover from, so after enough
consecutive rejections it RE-ACQUIRES and accepts whatever it is given.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import hypot
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class AnchorTrackerConfig:
    # How many accepted radii to keep for the running median.
    history: int = 15
    # Trust the radius prior only once this many samples agree.
    min_samples: int = 4
    # Reject a radius this far (fractionally) from the running median. Loose:
    # the primary detector derives the radius from the HP digit-line WIDTH, so
    # it legitimately changes when the HP value gains or loses a digit
    # (9999 -> 10000 is a real, sudden step).
    radius_tol: float = 0.80
    # Reject a jump further than this fraction of the play rect's short side.
    # Generous because the perception stage runs a few times per second, not
    # per frame, so consecutive observations can be ~0.5s apart.
    max_jump: float = 0.60
    # Carry the last verified anchor forward for at most this many observations.
    # This is the tracker's main job now.
    max_coast: int = 8
    # After this many consecutive rejections, drop the history and accept the
    # next detection unconditionally, so a bad history can never lock out a
    # genuinely relocated player.
    reacquire_after: int = 4


class AnchorTracker:
    """Accept / reject / coast on raw anchors from find_player_position."""

    def __init__(self, config: Optional[AnchorTrackerConfig] = None):
        self.config = config or AnchorTrackerConfig()
        self.reset()

    def reset(self) -> None:
        self._radii = deque(maxlen=self.config.history)
        self._last: Optional[Tuple[int, int, int]] = None
        self._coast = 0
        self._rejects = 0
        self.last_reason: str = "init"

    @property
    def radius_estimate(self) -> Optional[float]:
        if len(self._radii) < self.config.min_samples:
            return None
        return float(np.median(self._radii))

    def update(self, anchor, rect=None, frame_shape=None):
        """Validate one raw anchor.

        Returns (anchor, accepted). `anchor` may be the raw detection, the
        carried-forward previous one, or None once coasting is exhausted.
        `rect` is the play rect (x, y, w, h); frame_shape is (h, w) and is used
        only when no rect is supplied.
        """
        cfg = self.config
        if rect is None:
            if frame_shape is None:
                rect = (0, 0, 1, 1)
            else:
                rect = (0, 0, frame_shape[1], frame_shape[0])
        rx, ry, rw, rh = rect
        short = max(1.0, min(rw, rh))

        if anchor is None or len(anchor) < 3 or anchor[2] <= 0:
            return self._coast_or_none("no detection")

        ax, ay, ar = int(anchor[0]), int(anchor[1]), int(anchor[2])
        forced = self._rejects >= cfg.reacquire_after

        if not forced:
            # (1) radius must agree with the running estimate
            est = self.radius_estimate
            if est is not None and abs(ar - est) / est > cfg.radius_tol:
                return self._coast_or_none(f"radius {ar} vs est {est:.0f}")

            # (2) must not have teleported since the last accepted frame
            if self._last is not None:
                moved = hypot(ax - self._last[0], ay - self._last[1])
                if moved > cfg.max_jump * short:
                    return self._coast_or_none(f"jumped {moved:.0f}px")

        # Accepted.
        self._radii.append(ar)
        self._last = (ax, ay, ar)
        self._coast = 0
        self._rejects = 0
        self.last_reason = "reacquired" if forced else "ok"
        return (ax, ay, ar), True

    def _coast_or_none(self, reason: str):
        self.last_reason = reason
        self._rejects += 1
        if self._last is not None and self._coast < self.config.max_coast:
            self._coast += 1
            return self._last, False
        return None, False
