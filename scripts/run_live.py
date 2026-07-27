#!/usr/bin/env python3
"""Run the live perception loop (prints one perception status line per tick).

    python scripts/run_live.py --video media/testvideos/test_game1.mp4
    python scripts/run_live.py --screen            # needs `pip install mss`

Thin wrapper so the `perception` package resolves regardless of cwd.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from perception.liveLoop import main  # noqa: E402

if __name__ == "__main__":
    main()
