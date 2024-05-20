from datetime import timedelta

from pytimeparse import parse


def parse_timedelta(time_str: str) -> timedelta:
    """Parse a time string and return a timedelta object.

    Using time expression parser: https://github.com/wroberts/pytimeparse

    Args:
        time_str: The time string to parse.

    Returns:
        timedelta: The parsed timedelta object.

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
    """
    if not time_str:
        return timedelta(days=0)

    return timedelta(seconds=parse(time_str))
