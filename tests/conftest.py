"""Make the project importable from the tests directory."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import Settings  # noqa: E402


@pytest.fixture
def settings() -> Settings:
    """Defaults with no environment involvement, so tests are deterministic."""
    s = Settings()
    s.validate()
    return s
