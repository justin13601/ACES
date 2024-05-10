"""This module contains types defined by this package.

These are all simple types using named tuples so can be safely ignored by downstream users provided data
fields are passed in the correct order.
"""

from datetime import timedelta
from typing import NamedTuple


class TemporalWindowBounds(NamedTuple):
    """Named tuple to represent temporal window bounds.

    Attributes:
        start_inclusive: The start of the window, inclusive.
        window_size: The size of the window.
        end_inclusive: The end of the window, inclusive.
        offset: The offset from the start of the window to the end of the window.

    Example:
        >>> bounds = TemporalWindowBounds(
        ...     start_inclusive=True,
        ...     window_size=timedelta(days=1),
        ...     end_inclusive=False,
        ...     offset=timedelta(hours=1)
        ... )
        >>> bounds # doctest: +NORMALIZE_WHITESPACE
        TemporalWindowBounds(start_inclusive=True,
                             window_size=datetime.timedelta(days=1),
                             end_inclusive=False,
                             offset=datetime.timedelta(seconds=3600))
        >>> start_inclusive, window_size, end_inclusive, offset = bounds
        >>> start_inclusive
        True
        >>> window_size
        datetime.timedelta(days=1)
        >>> end_inclusive
        False
        >>> offset
        datetime.timedelta(seconds=3600)
    """

    start_inclusive: bool
    window_size: timedelta
    end_inclusive: bool
    offset: timedelta | None

    @property
    def polars_gp_rolling_kwargs(self) -> dict[str, str | timedelta]:
        """Return the parameters for a group_by rolling operation in Polars.

        Examples:
            >>> TemporalWindowBounds(
            ...     start_inclusive=True,
            ...     window_size=timedelta(days=1),
            ...     end_inclusive=True,
            ...     offset=None
            ... ).polars_gp_rolling_kwargs # doctest: +NORMALIZE_WHITESPACE
            {'period': datetime.timedelta(days=1),
             'offset': datetime.timedelta(0),
             'closed': 'both'}
            >>> TemporalWindowBounds(
            ...     start_inclusive=True,
            ...     window_size=timedelta(days=1),
            ...     end_inclusive=True,
            ...     offset=timedelta(hours=1)
            ... ).polars_gp_rolling_kwargs # doctest: +NORMALIZE_WHITESPACE
            {'period': datetime.timedelta(days=1),
             'offset': datetime.timedelta(seconds=3600),
             'closed': 'both'}
            >>> TemporalWindowBounds(
            ...     start_inclusive=False,
            ...     window_size=timedelta(days=2),
            ...     end_inclusive=False,
            ...     offset=timedelta(minutes=1)
            ... ).polars_gp_rolling_kwargs # doctest: +NORMALIZE_WHITESPACE
            {'period': datetime.timedelta(days=2),
             'offset': datetime.timedelta(seconds=60),
             'closed': 'none'}
            >>> TemporalWindowBounds(
            ...     start_inclusive=True,
            ...     window_size=timedelta(days=2),
            ...     end_inclusive=False,
            ...     offset=timedelta(minutes=1)
            ... ).polars_gp_rolling_kwargs # doctest: +NORMALIZE_WHITESPACE
            {'period': datetime.timedelta(days=2),
             'offset': datetime.timedelta(seconds=60),
             'closed': 'left'}
            >>> TemporalWindowBounds(
            ...     start_inclusive=False,
            ...     window_size=timedelta(days=2),
            ...     end_inclusive=True,
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

        if self.start_inclusive and self.end_inclusive:
            closed = "both"
        elif self.start_inclusive:
            closed = "left"
        elif self.end_inclusive:
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


class ToEventWindowBounds(NamedTuple):
    """Named tuple to represent temporal window bounds.

    Attributes:
        start_inclusive: The start of the window, inclusive.
        end_event: The string name of the event that bounds the end of this window. Operationally, this is
            interpreted as the string name of the column which contains a positive value if the row
            corresponds to the end event of this window and a zero otherwise.
        end_inclusive: The end of the window, inclusive.
        offset: The offset from the start of the window to the end of the window.

    Example:
        >>> bounds = ToEventWindowBounds(
        ...     start_inclusive=True,
        ...     end_event="foo",
        ...     end_inclusive=False,
        ...     offset=timedelta(hours=1)
        ... )
        >>> bounds # doctest: +NORMALIZE_WHITESPACE
        ToEventWindowBounds(start_inclusive=True,
                            end_event='foo',
                            end_inclusive=False,
                            offset=datetime.timedelta(seconds=3600))
        >>> start_inclusive, end_event, end_inclusive, offset = bounds
        >>> start_inclusive
        True
        >>> end_event
        'foo'
        >>> end_inclusive
        False
        >>> offset
        datetime.timedelta(seconds=3600)
    """

    start_inclusive: bool
    end_event: str
    end_inclusive: bool
    offset: timedelta | None
