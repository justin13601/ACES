"""Test set-up and fixtures code."""

import json
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import polars as pl
import pytest


@pytest.fixture(autouse=True)
def _setup_doctest_namespace(doctest_namespace: dict[str, Any]):
    doctest_namespace.update(
        {
            "MagicMock": MagicMock,
            "sys": sys,
            "Path": Path,
            "patch": patch,
            "json": json,
            "pl": pl,
            "datetime": datetime,
            "UTC": UTC,
            "tempfile": tempfile,
        }
    )
