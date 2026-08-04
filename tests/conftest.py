"""Shared pytest setup.

The project is a set of top-level packages (`perception/`, `rl/`) rather than an
installed distribution, so the repo root has to be importable. Every entry-point
script does this for itself with a `sys.path.insert`; doing it once here is what
lets the tests be plain modules instead of scripts.
"""

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "media" / "fixtures"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def repo_root() -> pathlib.Path:
    return ROOT


@pytest.fixture(scope="session")
def showdown_frame():
    """The one committed in-match screenshot, or skip.

    Several tests assert against real pixels rather than synthetic grids. That
    is deliberate -- a planner that works only on a simulator is not evidence of
    much -- but it means they cannot run without the fixture.
    """
    import cv2
    img = cv2.imread(str(FIXTURES / "showdown.png"))
    if img is None:
        pytest.skip("media/fixtures/showdown.png not present")
    return img
