"""Test set-up and fixtures code."""

import json
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
import yaml


@pytest.fixture(autouse=True)
def _setup_doctest_namespace(doctest_namespace: dict[str, Any], caplog: pytest.LogCaptureFixture) -> None:
    doctest_namespace.update(
        {
            "caplog": caplog,
            "MagicMock": MagicMock,
            "sys": sys,
            "Path": Path,
            "patch": patch,
            "json": json,
            "pl": pl,
            "datetime": datetime,
            "timedelta": timedelta,
            "tempfile": tempfile,
            "yaml": yaml,
        }
    )
