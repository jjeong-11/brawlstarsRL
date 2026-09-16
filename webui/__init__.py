"""
webui/
======

A rudimentary browser front end: pick a brawler, press Start, watch the
annotated decision traces stream in live.

    python scripts/run_webui.py            # -> http://127.0.0.1:5000

`runner.py` owns the loop (the same tick as `scripts/watch_live.py --trace`),
`app.py` is a thin Flask layer over it. Weights are resolved per brawler by
`brawlers/registry.py`.
"""

from .runner import SessionConfig, SessionRunner

__all__ = ["SessionConfig", "SessionRunner", "create_app"]


def create_app(*args, **kwargs):
    """Lazy so importing `webui` does not require Flask."""
    from .app import create_app as _create_app
    return _create_app(*args, **kwargs)
