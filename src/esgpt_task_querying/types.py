"""This module contains types defined by this package.

These are all simple types using named tuples so can be safely ignored by downstream users provided data
fields are passed in the correct order.
"""

import dataclasses
from datetime import timedelta


@dataclasses.dataclass(order=True)
class TemporalWindowBounds:
    """Named tuple to represent temporal window bounds.

    Attributes:
        left_inclusive: The start of the window, inclusive.
        window_size: The size of the window.
        right_inclusive: The end of the window, inclusive.
        offset: The offset from the start of the window to the end of the window.

    Example:
        >>> bounds = TemporalWindowBounds(
        ...     left_inclusive=True,
        ...     window_size=timedelta(days=1),
        ...     right_inclusive=False,
        ...     offset=timedelta(hours=1)
        ... )
        >>> bounds # doctest: +NORMALIZE_WHITESPACE
        TemporalWindowBounds(left_inclusive=True,
                             window_size=datetime.timedelta(days=1),
                             right_inclusive=False,
                             offset=datetime.timedelta(seconds=3600))
        >>> left_inclusive, window_size, right_inclusive, offset = bounds
        >>> bounds.left_inclusive
        True
        >>> window_size
        datetime.timedelta(days=1)
        >>> right_inclusive
        False
        >>> offset
        datetime.timedelta(seconds=3600)
    """

    left_inclusive: bool
    window_size: timedelta
    right_inclusive: bool
    offset: timedelta | None

    # Needed to make it accessible like a tuple.
    def __iter__(self):
        return (getattr(self, field.name) for field in dataclasses.fields(self))

    @property
    def polars_gp_rolling_kwargs(self) -> dict[str, str | timedelta]:
        """Return the parameters for a group_by rolling operation in Polars.

        Examples:
            >>> TemporalWindowBounds(
            ...     left_inclusive=True,
            ...     window_size=timedelta(days=1),
            ...     right_inclusive=True,
            ...     offset=None
            ... ).polars_gp_rolling_kwargs # doctest: +NORMALIZE_WHITESPACE
            {'period': datetime.timedelta(days=1),
             'offset': datetime.timedelta(0),
             'closed': 'both'}
            >>> TemporalWindowBounds(
            ...     left_inclusive=True,
            ...     window_size=timedelta(days=1),
            ...     right_inclusive=True,
            ...     offset=timedelta(hours=1)
            ... ).polars_gp_rolling_kwargs # doctest: +NORMALIZE_WHITESPACE
            {'period': datetime.timedelta(days=1),
             'offset': datetime.timedelta(seconds=3600),
             'closed': 'both'}
            >>> TemporalWindowBounds(
            ...     left_inclusive=False,
            ...     window_size=timedelta(days=2),
            ...     right_inclusive=False,
            ...     offset=timedelta(minutes=1)
            ... ).polars_gp_rolling_kwargs # doctest: +NORMALIZE_WHITESPACE
            {'period': datetime.timedelta(days=2),
             'offset': datetime.timedelta(seconds=60),
             'closed': 'none'}
            >>> TemporalWindowBounds(
            ...     left_inclusive=True,
            ...     window_size=timedelta(days=2),
            ...     right_inclusive=False,
            ...     offset=timedelta(minutes=1)
            ... ).polars_gp_rolling_kwargs # doctest: +NORMALIZE_WHITESPACE
            {'period': datetime.timedelta(days=2),
             'offset': datetime.timedelta(seconds=60),
             'closed': 'left'}
            >>> TemporalWindowBounds(
            ...     left_inclusive=False,
            ...     window_size=timedelta(days=2),
            ...     right_inclusive=True,
            ...     offset=timedelta(minutes=1)
            ... ).polars_gp_rolling_kwargs # doctest: +NORMALIZE_WHITESPACE
            {'period': datetime.timedelta(days=2),
             'offset': datetime.timedelta(seconds=60),
             'closed': 'right'}
        """
        if self.offset is None:
            offset = timedelta(days=0)
        else:
            offset = self.offset

        if self.left_inclusive and self.right_inclusive:
            closed = "both"
        elif self.left_inclusive:
            closed = "left"
        elif self.right_inclusive:
            closed = "right"
        else:
            closed = "none"

        # set parameters for group_by rolling window
        if self.window_size < timedelta(days=0):
            period = -self.window_size
            offset = -period + offset
        else:
            period = self.window_size
            offset = offset

        return {"period": period, "offset": offset, "closed": closed}


@dataclasses.dataclass(order=True)
class ToEventWindowBounds:
    """Named tuple to represent temporal window bounds.

    Attributes:
        left_inclusive: The start of the window, inclusive.
        end_event: The string name of the event that bounds the end of this window. Operationally, this is
            interpreted as the string name of the column which contains a positive value if the row
            corresponds to the end event of this window and a zero otherwise.
        right_inclusive: The end of the window, inclusive.
        offset: The offset from the start of the window to the end of the window.

    Raises:
        ValueError: If `end_event` is an empty string.
        ValueError: If `offset` is negative.

    Example:
        >>> bounds = ToEventWindowBounds(
        ...     left_inclusive=True,
        ...     end_event="foo",
        ...     right_inclusive=False,
        ...     offset=timedelta(hours=1)
        ... )
        >>> bounds # doctest: +NORMALIZE_WHITESPACE
        ToEventWindowBounds(left_inclusive=True,
                            end_event='foo',
                            right_inclusive=False,
                            offset=datetime.timedelta(seconds=3600))
        >>> left_inclusive, end_event, right_inclusive, offset = bounds
        >>> left_inclusive
        True
        >>> end_event
        'foo'
        >>> right_inclusive
        False
        >>> offset
        datetime.timedelta(seconds=3600)
        >>> bounds = ToEventWindowBounds(
        ...     left_inclusive=True,
        ...     end_event="",
        ...     right_inclusive=False,
        ...     offset=timedelta(hours=1)
        ... )
        Traceback (most recent call last):
            ...
        ValueError: end_event must be a non-empty string
        >>> bounds = ToEventWindowBounds(
        ...     left_inclusive=True,
        ...     end_event="foo",
        ...     right_inclusive=False,
        ...     offset=timedelta(hours=-1)
        ... )
        Traceback (most recent call last):
            ...
        ValueError: offset must be non-negative. Got -1 day, 23:00:00
    """

    left_inclusive: bool
    end_event: str
    right_inclusive: bool
    offset: timedelta | None

    def __post_init__(self):
        if self.end_event == "":
            raise ValueError("end_event must be a non-empty string")

        if self.offset is None:
            self.offset = timedelta(0)

        if self.offset < timedelta(0):
            raise ValueError(f"offset must be non-negative. Got {self.offset}")

    # Needed to make it accessible like a tuple.
    def __iter__(self):
        return (getattr(self, field.name) for field in dataclasses.fields(self))
