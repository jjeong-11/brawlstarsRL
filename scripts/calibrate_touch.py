#!/usr/bin/env python3
"""
One-time touch-orientation calibration for the sendevent executor.

The digitizer reports coordinates in the panel's portrait orientation, while your
controls.json is in landscape — so one of two flips is correct. This taps the
ATTACK button under BOTH orientations (~1.5s apart). Watch the phone: whichever
tap lands on the attack button (bottom-right) is your orientation.

    python scripts/calibrate_touch.py --serial 57230DLCR000M4 --controls controls.json

Then run training with --sendevent. If orientation "B" was the correct one, tell me
and I'll flip the default (or pass flip_short=True, flip_long=False yourself).
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rl.actions import Controls  # noqa: E402
from rl.sendevent_backend import SendeventExecutor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default=None)
    ap.add_argument("--controls", default=None, help="controls.json (recommended)")
    args = ap.parse_args()

    controls = Controls.from_json(args.controls) if args.controls else None
    ex = SendeventExecutor(serial=args.serial, controls=controls)
    print("Get the phone into a match (or any screen) and WATCH it now.")
    ex.calibrate()
    ex.close()


if __name__ == "__main__":
    main()
