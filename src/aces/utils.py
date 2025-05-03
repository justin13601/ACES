import io
import logging
import sys
from collections.abc import Generator
from contextlib import contextmanager
from datetime import timedelta

from bigtree import Node, print_tree
from pytimeparse import parse

logger = logging.getLogger(__name__)


def parse_timedelta(time_str: str | None = None) -> timedelta:
    """Parse a time string and return a timedelta object.

    Using time expression parser: https://github.com/wroberts/pytimeparse

    Args:
        time_str: The time string to parse.

    Returns:
        datetime.timedelta: The parsed timedelta object.

    Examples:
        >>> parse_timedelta("1 days")
        datetime.timedelta(days=1)
        >>> parse_timedelta("1 day")
        datetime.timedelta(days=1)
        >>> parse_timedelta("1 days 2 hours 3 minutes 4 seconds")
        datetime.timedelta(days=1, seconds=7384)
        >>> parse_timedelta('1 day, 14:20:16')
        datetime.timedelta(days=1, seconds=51616)
        >>> parse_timedelta('365 days')
        datetime.timedelta(days=365)
        >>> parse_timedelta()
        datetime.timedelta(0)
        >>> parse_timedelta("")
        datetime.timedelta(0)
        >>> parse_timedelta(None)
        datetime.timedelta(0)
    """
    if not time_str:
        return timedelta(days=0)

    return timedelta(seconds=parse(time_str))


@contextmanager
def capture_output() -> Generator[io.StringIO, None, None]:
    """A context manager to capture stdout output.

    This can eventually be eliminated if https://github.com/kayjan/bigtree/issues/285 is resolved.

    Examples:
        >>> with capture_output() as captured:
        ...     print("Hello, world!")
        >>> captured.getvalue().strip()
        'Hello, world!'
    """
    new_out = io.StringIO()  # Create a StringIO object to capture output
    old_out = sys.stdout  # Save the current stdout so we can restore it later
    try:
        sys.stdout = new_out  # Redirect stdout to our StringIO object
        yield new_out
    finally:
        sys.stdout = old_out  # Restore the original stdout


def log_tree(node: Node) -> None:
    """Logs the tree structure using logging.info."""
    with capture_output() as captured:
        print_tree(node, style="const_bold")  # This will print to the captured StringIO instead of stdout
    logger.info("\n" + captured.getvalue())  # Log the captured output
