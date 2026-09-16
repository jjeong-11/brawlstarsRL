#!/usr/bin/env python3
"""
Launch the web UI: pick a brawler, press Start, watch the decision trace live.

    python scripts/run_webui.py                     # -> http://127.0.0.1:5000
    python scripts/run_webui.py --port 8080
    python scripts/run_webui.py --host 0.0.0.0      # reachable from the phone

WHAT IT SHOWS
-------------
The same annotated frame `scripts/watch_live.py --trace` writes to
debugOutput/trace/, streamed to the browser as it is produced: fused occupancy
+ gas, the A* path, the latched waypoint, the joystick vector, the decoded
action and the reward breakdown.

Default mode is **observe** — the policy is asked for an action and the planner
plans a route, but no executor is constructed, so nothing reaches the phone.
Switch to **control** in the page to let the bot actually play; calibrate
controls.json first (docs/ANDROID_CONTROL.md) or the taps will miss.

PER-BRAWLER WEIGHTS
-------------------
Choosing a brawler resolves models/<brawler>/policy.zip, falling back to the
shared brawlstars_move.zip until that brawler has been trained. The page always
says which of the two loaded — see brawlers/registry.py for why that
distinction is worth surfacing.

NO PHONE? Put a recording path in Options → "play a recording instead of the
phone" and the whole loop runs offline against the video.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (default: localhost only)")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--debug", action="store_true",
                    help="Flask auto-reload. NOTE: the reloader restarts the "
                         "process, which kills a running session mid-match.")
    args = ap.parse_args()

    try:
        # webui/app.py raises SystemExit at import with the pip line if Flask is
        # missing. Catch it here so the launcher prints one clear message
        # instead of a traceback.
        from webui.app import create_app
    except SystemExit as e:
        print(e)
        return

    from brawlers import default_registry
    reg = default_registry()
    trained = reg.trained()
    print(f"{len(reg)} brawlers loaded from {reg.roster_path.name}")
    print(f"weights: {len(trained)} brawler-specific "
          f"{trained or '(none yet — all fall back to the shared base policy)'}")
    print(f"\n  http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}\n")

    app = create_app()
    # threaded=True is required, not cosmetic: the MJPEG stream holds a worker
    # for the life of the connection, so a single-threaded server would answer
    # the stream and then never serve /api/status again.
    app.run(host=args.host, port=args.port, threaded=True, debug=args.debug)


if __name__ == "__main__":
    main()
