#!/usr/bin/env python3
"""Safely calibrate the raw-touch (``--sendevent``) control backend.

The script sends exactly two short taps at the configured attack-button
location, once for each landscape orientation.  It never starts a match and
does not send movement commands.  Watch the phone and note which tap hit the
attack button; use that orientation when configuring ``SendeventExecutor``.

Run from the project root::

    python3 scripts/calibrate_sendevent.py --serial <DEVICE> --controls controls.json
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rl.actions import Controls  # noqa: E402
from rl.sendevent_backend import SendeventExecutor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send two test taps to calibrate raw Android touch orientation.")
    parser.add_argument("--serial", default=None,
                        help="adb device serial (required if more than one device is connected)")
    parser.add_argument("--controls", default="controls.json",
                        help="calibrated controls JSON (default: controls.json)")
    parser.add_argument("--device", default=None,
                        help="raw touch event device, e.g. /dev/input/event2")
    args = parser.parse_args()

    controls_path = pathlib.Path(args.controls)
    if not controls_path.exists():
        parser.error(f"controls file not found: {controls_path}")

    kwargs = {"serial": args.serial, "controls": Controls.from_json(str(controls_path))}
    if args.device:
        kwargs["device"] = args.device
    executor = SendeventExecutor(**kwargs)
    try:
        print("Watch the phone. Two attack-button test taps will follow.")
        executor.calibrate()
    finally:
        executor.close()


if __name__ == "__main__":
    main()
