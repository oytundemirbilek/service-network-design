"""Common functions for pytest modules."""

from __future__ import annotations

import logging
from collections.abc import Generator

import pytest


@pytest.fixture
def logger_test() -> Generator[logging.Logger]:
    """Yield a logger for all tests."""
    yield logging.getLogger(__name__)
    logging.shutdown()
