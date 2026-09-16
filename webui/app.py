"""
webui/app.py
============

A thin Flask front end over one :class:`SessionRunner`.

Pick a brawler, press Start, watch the annotated decision frames stream in.
Nothing in here does perception, planning or inference — every route either
reads the registry or pokes the runner, and the runner owns the loop on its own
thread. That split is what keeps a slow browser from stalling the tick.

Routes
------
GET  /                  the page
GET  /api/brawlers      roster + which weights each one would load right now
GET  /api/devices       connected adb devices
GET  /api/status        session status (polled ~2/s by the page)
POST /api/start         {brawler, mode, fps, serial, source, ...}
POST /api/stop          end the session
GET  /api/stream        MJPEG of the annotated trace frames
GET  /api/frame.jpg     the single newest frame (for a still or a screenshot)

Single session by design: the loop drives one phone over one adb channel, so a
second concurrent session would fight the first for the device. `/api/start`
returns 409 rather than silently interleaving them.
"""

from __future__ import annotations

import json
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from flask import (Flask, Response, jsonify, redirect, render_template,
                       request, send_file)
except ImportError:                                            # pragma: no cover
    raise SystemExit(
        "The web UI needs Flask:\n\n"
        "    pip install flask\n\n"
        "(it is in requirements.txt; nothing else in the project depends on it)"
    )

from brawlers import (default_abilities, default_registry,        # noqa: E402
                      profile_for)
from webui.runner import SessionConfig, SessionRunner, list_adb_devices  # noqa: E402

BOUNDARY = "frame"


def create_app() -> Flask:
    app = Flask(__name__)
    registry = default_registry()
    runner = SessionRunner(registry)
    app.config["REGISTRY"] = registry
    app.config["RUNNER"] = runner

    def _roster_payload(full: bool = True) -> dict:
        """Every brawler with its resolved weights and portrait URL.

        The page is rendered with `full=False` inlined, so picking a brawler is
        a pure client-side lookup. Selection used to depend on a per-click fetch
        of /api/brawlers/<id>, which meant one failed request left the picker
        looking dead; now the only thing that can go wrong is the initial page
        load, and that failure is visible.

        `full=False` trims to what the picker actually draws — absolute paths
        and the (currently empty) profile dicts are five times the bytes and the
        page never reads them. /api/brawlers still serves everything.
        """
        book = default_abilities()
        out = []
        for b in registry.all():
            slot = registry.slot(b.id)
            a = book.get(b.id)
            entry = {
                "id": b.id, "name": b.name, "rarity": b.rarity,
                "multi_body": b.multi_body,
                "icon": f"/api/icon/{b.id}",
                # The two facts the combat script reads, shown in the picker so
                # an unexplained "manual" is never the only thing on screen.
                "brawler_class": a.brawler_class if a else None,
                "range_tiles": a.range_tiles if a else None,
                "range_confidence": a.confidence if a else "unknown",
                "aim": a.aim if a else "auto",
                "aim_reason": a.aim_reason if a else "",
                "model": (slot.to_dict() if full else
                          {"label": slot.label,
                           "is_brawler_specific": slot.is_brawler_specific}),
            }
            if full:
                entry["scid"] = b.scid
                entry["profile"] = profile_for(b.id).to_dict()
                entry["ability"] = a.to_dict() if a else None
            out.append(entry)
        payload = {"brawlers": out,
                   "rarity_order": [r for r, _ in registry.grouped()],
                   "trained": registry.trained()}
        if full:
            payload["models_root"] = str(registry.models_root)
        return payload

    # --- page ------------------------------------------------------------- #
    @app.get("/")
    def index():
        payload = _roster_payload(full=False)
        return render_template(
            "index.html",
            groups=registry.grouped(),
            n_brawlers=len(registry),
            trained=payload["trained"],
            # </script> inside a JSON string would end the tag early; escaping
            # the slash is the standard fix and is still valid JSON.
            roster_json=json.dumps(payload).replace("</", "<\\/"))

    # --- data ------------------------------------------------------------- #
    @app.get("/api/brawlers")
    def api_brawlers():
        return jsonify(_roster_payload())

    @app.get("/api/icon/<brawler_id>")
    def api_icon(brawler_id):
        """The brawler portrait: the cached file if we have it, else the CDN.

        Going through the app rather than linking the CDN straight from the page
        means `tools/sync_brawlers.py --icons` changes nothing in the markup —
        cached and uncached installs render from the same URL.
        """
        try:
            registry.get(brawler_id)
        except KeyError:
            return ("", 404)
        local = registry.local_icon(brawler_id)
        if local is not None:
            return send_file(local, mimetype="image/png",
                             max_age=60 * 60 * 24 * 30)
        url = registry.icon_url(brawler_id)
        return redirect(url, code=302) if url else ("", 404)

    @app.get("/api/brawlers/<brawler_id>")
    def api_brawler(brawler_id):
        try:
            b = registry.get(brawler_id)
        except KeyError as e:
            return jsonify({"error": str(e)}), 404
        return jsonify({"id": b.id, "name": b.name, "rarity": b.rarity,
                        "multi_body": b.multi_body,
                        "model": registry.slot(b.id).to_dict(),
                        "profile": profile_for(b.id).to_dict()})

    @app.get("/api/devices")
    def api_devices():
        return jsonify({"devices": list_adb_devices()})

    @app.get("/api/status")
    def api_status():
        return jsonify(runner.status())

    # --- control ---------------------------------------------------------- #
    @app.post("/api/start")
    def api_start():
        if runner.is_running:
            return jsonify({"error": "a session is already running"}), 409
        body = request.get_json(silent=True) or {}
        try:
            cfg = SessionConfig(
                brawler=str(body.get("brawler") or "shelly"),
                mode=str(body.get("mode") or "observe"),
                serial=(body.get("serial") or None),
                fps=float(body.get("fps") or 4.0),
                source=str(body.get("source") or "phone"),
                video_path=(body.get("video_path") or None),
                controls_path=(body.get("controls_path") or None),
                backend=str(body.get("backend") or "adb"),
                sendevent_orientation=str(body.get("sendevent_orientation") or "A"),
                save_traces=bool(body.get("save_traces", False)),
                deterministic=bool(body.get("deterministic", False)),
            )
            return jsonify(runner.start(cfg))
        except (KeyError, ValueError) as e:
            return jsonify({"error": str(e)}), 400
        except RuntimeError as e:
            return jsonify({"error": str(e)}), 409

    @app.post("/api/stop")
    def api_stop():
        return jsonify(runner.stop())

    # --- pictures --------------------------------------------------------- #
    @app.get("/api/frame.jpg")
    def api_frame():
        jpeg = runner.latest_jpeg()
        if jpeg is None:
            return ("", 204)
        return Response(jpeg, mimetype="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.get("/api/stream")
    def api_stream():
        """MJPEG. An <img src> renders this natively, no JS decoding needed."""
        def gen():
            seq = -1
            idle = 0
            while True:
                jpeg, seq_now = runner.wait_for_frame(seq, timeout=5.0)
                if jpeg is None or seq_now == seq:
                    # No new frame. Keep the connection open while a session is
                    # running (starting up can take a few seconds); give up once
                    # it clearly is not going to produce one, so a forgotten tab
                    # does not hold a worker thread forever.
                    idle += 1
                    if not runner.is_running and idle > 2:
                        break
                    continue
                idle = 0
                seq = seq_now
                yield (b"--" + BOUNDARY.encode() + b"\r\n"
                       b"Content-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                       + jpeg + b"\r\n")

        return Response(gen(),
                        mimetype=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
                        headers={"Cache-Control": "no-store",
                                 "X-Accel-Buffering": "no"})

    return app


if __name__ == "__main__":                      # python -m webui.app
    create_app().run(host="127.0.0.1", port=5000, threaded=True, debug=False)
