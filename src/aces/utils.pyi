import io
import os
import sys
from contextlib import contextmanager
from datetime import timedelta
from typing import Any, Generator, Optional

import hydra
from bigtree import Node
from loguru import logger
from pytimeparse import timeparse

def parse_timedelta(time_str: Optional[str]) -> timedelta: ...
@contextmanager
def capture_output() -> Generator[io.StringIO, None, None]: ...
def log_tree(node: Node) -> None: ...
def hydra_loguru_init(filename: str) -> None: ...
