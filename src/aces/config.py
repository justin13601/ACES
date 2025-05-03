"""This module contains functions for loading and parsing the configuration file and subsequently building a
tree structure from the configuration."""

from __future__ import annotations

import dataclasses
import logging
import re
from dataclasses import field
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import networkx as nx
import polars as pl
import ruamel.yaml
from bigtree import Node

from .types import (
    ANY_EVENT_COLUMN,
    END_OF_RECORD_KEY,
    START_OF_RECORD_KEY,
    TemporalWindowBounds,
    ToEventWindowBounds,
)
from .utils import parse_timedelta

if TYPE_CHECKING:
    from collections import OrderedDict

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class PlainPredicateConfig:
    code: str | dict[str, Any]
    value_min: float | None = None
    value_max: float | None = None
    value_min_inclusive: bool | None = None
    value_max_inclusive: bool | None = None
    static: bool = False
    other_cols: dict[str, str] = field(default_factory=dict)

    def MEDS_eval_expr(self) -> pl.Expr:
        """Returns a Polars expression that evaluates this predicate for a MEDS formatted dataset.

        Note: The output syntax for the following examples is dependent on the polars version used. The
        expected outputs have been validated on polars version 0.20.30.

        Examples:
            >>> print(PlainPredicateConfig("BP//systolic", 120, 140, True, False).MEDS_eval_expr())
            [(col("code")) == ("BP//systolic")].all_horizontal([[(col("numeric_value")) >=
               (dyn int: 120)], [(col("numeric_value")) < (dyn int: 140)]])
            >>> cfg = PlainPredicateConfig("BP//systolic", value_min=120, value_min_inclusive=False)
            >>> print(cfg.MEDS_eval_expr())
            [(col("code")) == ("BP//systolic")].all_horizontal([[(col("numeric_value")) >
               (dyn int: 120)]])
            >>> cfg = PlainPredicateConfig("BP//systolic", value_max=140, value_max_inclusive=True)
            >>> print(cfg.MEDS_eval_expr())
            [(col("code")) == ("BP//systolic")].all_horizontal([[(col("numeric_value")) <=
               (dyn int: 140)]])
            >>> print(PlainPredicateConfig("BP//diastolic").MEDS_eval_expr())
            [(col("code")) == ("BP//diastolic")]
            >>> cfg = PlainPredicateConfig("BP//diastolic", other_cols={"chamber": "atrial"})
            >>> print(cfg.MEDS_eval_expr())
            [(col("code")) == ("BP//diastolic")].all_horizontal([[(col("chamber")) ==
               ("atrial")]])

            >>> PlainPredicateConfig(code={'regex': None, 'any': None}).MEDS_eval_expr()
            Traceback (most recent call last):
                ...
            ValueError: Only one of 'regex' or 'any' can be specified in the code field!
            Got: ['regex', 'any'].
            >>> PlainPredicateConfig(code={'foo': None}).MEDS_eval_expr()
            Traceback (most recent call last):
                ...
            ValueError: Invalid specification in the code field! Got: {'foo': None}.
            Expected one of 'regex', 'any'.
            >>> PlainPredicateConfig(code={'regex': ''}).MEDS_eval_expr()
            Traceback (most recent call last):
                ...
            ValueError: Invalid specification in the code field! Got: {'regex': ''}.
            Expected a non-empty string for 'regex'.
            >>> PlainPredicateConfig(code={'any': []}).MEDS_eval_expr()
            Traceback (most recent call last):
                ...
            ValueError: Invalid specification in the code field! Got: {'any': []}.
            Expected a list of strings for 'any'.

            >>> print(PlainPredicateConfig(code={'regex': '^foo.*'}).MEDS_eval_expr())
            col("code").str.contains(["^foo.*"])
            >>> print(PlainPredicateConfig(code={'regex': '^foo.*'}, value_min=120).MEDS_eval_expr())
            col("code").str.contains(["^foo.*"]).all_horizontal([[(col("numeric_value")) >
            (dyn int: 120)]])
            >>> print(PlainPredicateConfig(code={'any': ['foo', 'bar']}).MEDS_eval_expr())
            col("code").is_in([Series])
        """
        criteria = []
        if isinstance(self.code, dict):
            if len(self.code) > 1:
                raise ValueError(
                    "Only one of 'regex' or 'any' can be specified in the code field! "
                    f"Got: {list(self.code.keys())}."
                )

            if "regex" in self.code:
                if not self.code["regex"] or not isinstance(self.code["regex"], str):
                    raise ValueError(
                        "Invalid specification in the code field! "
                        f"Got: {self.code}. "
                        "Expected a non-empty string for 'regex'."
                    )
                criteria.append(pl.col("code").str.contains(self.code["regex"]))
            elif "any" in self.code:
                if not self.code["any"] or not isinstance(self.code["any"], list):
                    raise ValueError(
                        "Invalid specification in the code field! "
                        f"Got: {self.code}. "
                        f"Expected a list of strings for 'any'."
                    )
                criteria.append(pl.Expr.is_in(pl.col("code"), self.code["any"]))
            else:
                raise ValueError(
                    "Invalid specification in the code field! "
                    f"Got: {self.code}. "
                    "Expected one of 'regex', 'any'."
                )
        else:
            criteria.append(pl.col("code") == self.code)

        if self.value_min is not None:
            if self.value_min_inclusive:
                criteria.append(pl.col("numeric_value") >= self.value_min)
            else:
                criteria.append(pl.col("numeric_value") > self.value_min)
        if self.value_max is not None:
            if self.value_max_inclusive:
                criteria.append(pl.col("numeric_value") <= self.value_max)
            else:
                criteria.append(pl.col("numeric_value") < self.value_max)

        if self.other_cols:
            criteria.extend([pl.col(col) == value for col, value in self.other_cols.items()])

        if len(criteria) == 1:
            return criteria[0]
        else:
            return pl.all_horizontal(criteria)

    def ESGPT_eval_expr(self, values_column: str | None = None) -> pl.Expr:
        """Returns a Polars expression that evaluates this predicate for a ESGPT formatted dataset.

        Note: The output syntax for the following examples is dependent on the polars version used. The
        expected outputs have been validated on polars version 0.20.30.

        Examples:
            >>> cfg = PlainPredicateConfig("HR", value_min=120, value_min_inclusive=False)
            >>> print(cfg.ESGPT_eval_expr("HR"))
            [(col("HR")) > (dyn int: 120)]
            >>> print(PlainPredicateConfig("BP//systolic", 120, 140, True, False).ESGPT_eval_expr("BP_value"))
            [(col("BP")) == ("systolic")].all_horizontal([[(col("BP_value")) >=
               (dyn int: 120)], [(col("BP_value")) < (dyn int: 140)]])
            >>> cfg = PlainPredicateConfig("BP//systolic", value_min=120, value_min_inclusive=False)
            >>> print(cfg.ESGPT_eval_expr("blood_pressure_value"))
            [(col("BP")) == ("systolic")].all_horizontal([[(col("blood_pressure_value")) >
               (dyn int: 120)]])
            >>> cfg = PlainPredicateConfig("BP//systolic", value_max=140, value_max_inclusive=True)
            >>> print(cfg.ESGPT_eval_expr("blood_pressure_value"))
            [(col("BP")) == ("systolic")].all_horizontal([[(col("blood_pressure_value")) <=
               (dyn int: 140)]])
            >>> print(PlainPredicateConfig("BP//diastolic").ESGPT_eval_expr())
            [(col("BP")) == ("diastolic")]
            >>> print(PlainPredicateConfig("event_type//ADMISSION").ESGPT_eval_expr())
            col("event_type").strict_cast(String).str.split(["&"]).list.contains(["ADMISSION"])
            >>> print(PlainPredicateConfig("BP//diastolic//atrial").ESGPT_eval_expr())
            [(col("BP")) == ("diastolic//atrial")]
            >>> print(PlainPredicateConfig("BP//diastolic", None, None).ESGPT_eval_expr())
            [(col("BP")) == ("diastolic")]
            >>> print(PlainPredicateConfig("BP").ESGPT_eval_expr())
            col("BP").is_not_null()
            >>> print(PlainPredicateConfig("BP//systole", other_cols={"chamber": "atrial"}).ESGPT_eval_expr())
            [(col("BP")) == ("systole")].all_horizontal([[(col("chamber")) == ("atrial")]])

            >>> PlainPredicateConfig("BP//systolic", value_min=120).ESGPT_eval_expr()
            Traceback (most recent call last):
                ...
            ValueError: Must specify a values column for ESGPT predicates with a value_min = 120
            >>> PlainPredicateConfig("BP//systolic", value_max=140).ESGPT_eval_expr()
            Traceback (most recent call last):
                ...
            ValueError: Must specify a values column for ESGPT predicates with a value_max = 140
        """
        code_is_in_parts = "//" in self.code

        if code_is_in_parts:
            codes = self.code.split("//")
            measurement_name = codes.pop(0)
            code = "//".join(codes) if len(codes) > 1 else codes[0]
            if measurement_name.lower() == "event_type":
                criteria = [pl.col("event_type").cast(pl.Utf8).str.split("&").list.contains(code)]
            else:
                criteria = [pl.col(measurement_name) == code]
        elif (self.value_min is None) and (self.value_max is None):
            return pl.col(self.code).is_not_null()
        else:
            values_column = self.code
            criteria = []

        if self.value_min is not None:
            if values_column is None:
                raise ValueError(
                    f"Must specify a values column for ESGPT predicates with a value_min = {self.value_min}"
                )
            if self.value_min_inclusive:
                criteria.append(pl.col(values_column) >= self.value_min)
            else:
                criteria.append(pl.col(values_column) > self.value_min)
        if self.value_max is not None:
            if values_column is None:
                raise ValueError(
                    f"Must specify a values column for ESGPT predicates with a value_max = {self.value_max}"
                )
            if self.value_max_inclusive:
                criteria.append(pl.col(values_column) <= self.value_max)
            else:
                criteria.append(pl.col(values_column) < self.value_max)

        if self.other_cols:
            criteria.extend([pl.col(col) == value for col, value in self.other_cols.items()])

        if len(criteria) == 1:
            return criteria[0]
        else:
            return pl.all_horizontal(criteria)

    @property
    def is_plain(self) -> bool:
        return True


@dataclasses.dataclass
class DerivedPredicateConfig:
    """A configuration object for derived predicates, which are composed of multiple input predicates.

    Args:
        expr: The expression defining the derived predicate in terms of other predicates.

    Raises:
        ValueError: If the expression is empty, does not start with 'and(' or 'or(', or does not contain at
            least two input predicates.

    Examples:
        >>> pred = DerivedPredicateConfig("and(P1, P2, P3)")
        >>> pred = DerivedPredicateConfig("and()")
        Traceback (most recent call last):
            ...
        ValueError: Derived predicate expression must have at least two input predicates (comma separated).
        Got: 'and()'
        >>> pred = DerivedPredicateConfig("or(PA, PB)")
        >>> pred = DerivedPredicateConfig("PA + PB")
        Traceback (most recent call last):
            ...
        ValueError: Derived predicate expression must start with 'and(' or 'or('. Got: 'PA + PB'
        >>> pred = DerivedPredicateConfig("")
        Traceback (most recent call last):
            ...
        ValueError: Derived predicates must have a non-empty expression field.
    """

    expr: str
    static: bool = False

    def __post_init__(self) -> None:
        if not self.expr:
            raise ValueError("Derived predicates must have a non-empty expression field.")

        self.is_and = self.expr.startswith("and(") and self.expr.endswith(")")
        self.is_or = self.expr.startswith("or(") and self.expr.endswith(")")
        if not (self.is_and or self.is_or):
            raise ValueError(
                f"Derived predicate expression must start with 'and(' or 'or('. Got: '{self.expr}'"
            )

        if self.is_and:
            self.input_predicates = [x.strip() for x in self.expr[4:-1].split(",")]
        elif self.is_or:
            self.input_predicates = [x.strip() for x in self.expr[3:-1].split(",")]

        if len(self.input_predicates) < 2:
            raise ValueError(
                "Derived predicate expression must have at least two input predicates (comma separated). "
                f"Got: '{self.expr}'"
            )

    def eval_expr(self) -> pl.Expr:
        """Returns a Polars expression that evaluates this predicate against necessary dependent predicates.

        Note: The output syntax for the following examples is dependent on the polars version used. The
        expected outputs have been validated on polars version 0.20.30.

        Examples:
            >>> print(DerivedPredicateConfig("and(P1, P2, P3)").eval_expr())
            [(col("P1")) > (dyn int: 0)].all_horizontal([[(col("P2")) >
               (dyn int: 0)], [(col("P3")) > (dyn int: 0)]])
            >>> print(DerivedPredicateConfig("or(PA, PB)").eval_expr())
            [(col("PA")) > (dyn int: 0)].any_horizontal([[(col("PB")) > (dyn int: 0)]])
        """
        if self.is_and:
            return pl.all_horizontal([pl.col(pred) > 0 for pred in self.input_predicates])
        elif self.is_or:
            return pl.any_horizontal([pl.col(pred) > 0 for pred in self.input_predicates])

    @property
    def is_plain(self) -> bool:
        return False


@dataclasses.dataclass
class WindowConfig:
    """A configuration object for defining a window in the task extraction process.

    This defines the boundary points and constraints for a window in the patient record in the task extraction
    process.

    Args:
        start: The boundary conditions for the start of the window. This (like ``end``) can either be `None`,
            in which case the window starts at the beginning of the patient record, or is expressed through a
            string language that expresses a relative startpoint to this window either in reference to (a)
            another window's start or end event, (b) this window's `end` event. In case (a), this window's end
            event must either be `None` or reference this window's start event, and in case (b), this window's
            end event must reference a different window's start or end event.
            The string language is as follows:
              - ``None``: The window starts at the beginning of the patient record.
              - ``$REFERENCED <- $PREDICATE`` or ``$REFERENCED -> $PREDICATE``: The window starts at the
                closest event satisfying the predicate ``$PREDICATE`` relative to the ``$REFERENCED`` event.
                Form ``$REFERENCED <- $PREDICATE`` means that the window starts at the closest event _prior
                to_ the ``$REFERENCED`` event that satisfies the predicate ``$PREDICATE``, and the other form
                is analogous but with the closest event _after_ the ``$REFERENCED`` event.
              - ``$REFERENCED +- timedelta``: The window starts at the ``$REFERENCED`` event plus or minus the
                specified timedelta. The timedelta is expressed through the string language specified in the
                `utils.parse_timedelta` function.
              - ``$REFERENCED``: The window starts at the ``$REFERENCED`` event.
            In all cases, the ``$REFERENCED`` event must be either
              - The name of another window's start or end event, as specified by ``$WINDOW_NAME.start`` or
                ``$WINDOW_NAME.end``.
              - This window's end event, as specified by ``end``.
            In the case that ``$REFERENCED`` is this window's end event, the window must be defined such that
            ``start`` would precede ``end`` in the order of the patient record (e.g., ``$PREDICATE -> end`` is
            invalid, and ``end - timedelta`` is invalid).
        end: The name of the event that ends the window. See the documentation for ``start`` for more details
            on the specification language.
        start_inclusive: Whether or not the start event is included in the window. Note that this term can not
            only dictate whether an event's counts are included in the summarization of the window, but also
            whether or not an event satisfying ``$PREDICATE`` can be used as the boundary of an event. E.g.,
            if we have that `start_inclusive=False` and the `end` field is equal to `start -> $PREDICATE`, and
            it so happens that the `start` event itself satisfies `$PREDICATE`, the fact that
            `start_inclusive=False` will mean that we do not consider the `start` event itself to be a valid
            start to any window that ends at the same `start` event, as its timestamp when considered as the
            prospective "window start timestamp" occurs "after" the effective timestamp of itself when
            considered as the `$PREDICATE` event that marks the window end given that `start_inclusive=False`
            and thus we will think of the window as truly starting an iota after the timestamp of the `start`
            event itself.
        end_inclusive: Whether or not the end event is included in the window.
        has: A dictionary of predicates that must be present in the window, mapped to tuples of the form
            `(min_valid, max_valid)` that define the valid range the count of observations of the named
            predicate that must be found in a window for it to be considered valid. Either `min_valid` or
            `max_valid` constraints can be `None`, in which case those endpoints are left unconstrained.
            Likewise, unreferenced predicates are also left unconstrained. Note that as predicate counts are
            always integral, this specification does not need an additional inclusive/exclusive endpoint
            field, as one can simply increment the bound by one in the appropriate direction to achieve the
            result. Instead, this bound is always interpreted to be inclusive, so a window would satisfy the
            constraint for predicate `name` with constraint `name: (1, 2)` if the count of observations of
            predicate `name` in a window was either 1 or 2. All constraints in the dictionary must be
            satisfied on a window for it to be included.
        label: A string that specifies the name of a predicate to be used as the label for the task. The
            predicate count of the window this field is specified in will be extracted as a column in the
            final result. Hence, there can only be one 'label' per TaskExtractorConfig. If more than one
            'label' is specified, an error is raised. If the specified 'label' is not a defined predicate,
            an error is also raised. If no 'label' is specified, there will be not be a 'label' column.
        index_timestamp: A string that is either 'start' or 'end' and is used to index result rows. If it is
            defined, there will be an 'index_timestamp' column in the result with its values equal to the
            'start' or 'end' timestamp of the window in which it was specified. Usually, this will be
            specified to indicate the time of prediction for the task, which is often the 'end' of the input
            window. There can only be one 'index_timestamp' per TaskExtractorConfig. If more than one
            'index_timestamp' is specified, an error is raised. If the specified 'index_timestamp' is not
            'start' or 'end', an error is also raised. If no 'index_timestamp' is defined, there will be no
            'index_timestamp' column.

    Raises:
        ValueError: If the window is misconfigured in any of a variety of ways; see below for examples.

    Examples:
        >>> input_window = WindowConfig(
        ...     start=None,
        ...     end="trigger + 2 days",
        ...     start_inclusive=True,
        ...     end_inclusive=True,
        ...     has={"_ANY_EVENT": "(5, None)"},
        ...     index_timestamp="end",
        ... )
        >>> input_window.referenced_event
        ('trigger',)
        >>> # This window does not reference any "true" external predicates, only implicit predicates like
        >>> # start, end, and * events, so this list should be empty.
        >>> sorted(input_window.referenced_predicates)
        ['_ANY_EVENT']
        >>> input_window.start_endpoint_expr
        ToEventWindowBounds(left_inclusive=True,
                            end_event='-_RECORD_START',
                            right_inclusive=True,
                            offset=datetime.timedelta(0))
        >>> input_window.end_endpoint_expr
        TemporalWindowBounds(left_inclusive=False,
                             window_size=datetime.timedelta(days=2),
                             right_inclusive=False,
                             offset=datetime.timedelta(0))
        >>> input_window.root_node
        'end'
        >>> gap_window = WindowConfig(
        ...     start="input.end",
        ...     end="start + 24h",
        ...     start_inclusive=False,
        ...     end_inclusive=True,
        ...     has={"discharge": "(None, 0)", "death": "(None, 0)"}
        ... )
        >>> gap_window.referenced_event
        ('input', 'end')
        >>> sorted(gap_window.referenced_predicates)
        ['death', 'discharge']
        >>> gap_window.start_endpoint_expr is None
        True
        >>> gap_window.end_endpoint_expr
        TemporalWindowBounds(left_inclusive=False,
                             window_size=datetime.timedelta(days=1),
                             right_inclusive=True,
                             offset=datetime.timedelta(0))
        >>> gap_window.root_node
        'start'
        >>> gap_window = WindowConfig(
        ...     start="input.end",
        ...     end="start + 0h",
        ...     start_inclusive=False,
        ...     end_inclusive=True,
        ...     has={"discharge": "(None, 0)", "death": "(None, 0)"}
        ... )
        >>> gap_window.referenced_event
        ('input', 'end')
        >>> sorted(gap_window.referenced_predicates)
        ['death', 'discharge']
        >>> gap_window.start_endpoint_expr is None
        True
        >>> gap_window.end_endpoint_expr is None
        True
        >>> gap_window.root_node
        'start'
        >>> target_window = WindowConfig(
        ...     start="gap.end",
        ...     end="start -> discharge_or_death",
        ...     start_inclusive=False,
        ...     end_inclusive=True,
        ...     has={}
        ... )
        >>> target_window.referenced_event
        ('gap', 'end')
        >>> sorted(target_window.referenced_predicates)
        ['discharge_or_death']
        >>> target_window.start_endpoint_expr is None
        True
        >>> target_window.end_endpoint_expr
        ToEventWindowBounds(left_inclusive=False,
                            end_event='discharge_or_death',
                            right_inclusive=True,
                            offset=datetime.timedelta(0))
        >>> target_window.root_node
        'start'
        >>> target_window = WindowConfig(
        ...     start="end",
        ...     end="gap.end <- discharge_or_death",
        ...     start_inclusive=False,
        ...     end_inclusive=True,
        ...     has={}
        ... )
        >>> target_window.referenced_event
        ('gap', 'end')
        >>> sorted(target_window.referenced_predicates)
        ['discharge_or_death']
        >>> target_window.start_endpoint_expr is None
        True
        >>> target_window.end_endpoint_expr
        ToEventWindowBounds(left_inclusive=False,
                            end_event='-discharge_or_death',
                            right_inclusive=False,
                            offset=datetime.timedelta(0))
        >>> target_window.root_node
        'end'

        >>> invalid_window = WindowConfig(
        ...     start="gap.end gap.start",
        ...     end="start -> discharge_or_death",
        ...     start_inclusive=False,
        ...     end_inclusive=True,
        ...     has={}
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Window boundary reference must be either a valid alphanumeric/'_' string or a reference to
            another window's start or end event, formatted as a valid alphanumeric/'_' string, followed by
            '.start' or '.end'.
            Got: 'gap.end gap.start'
        >>> invalid_window = WindowConfig(
        ...     start="input",
        ...     end="start window -> discharge_or_death",
        ...     start_inclusive=False,
        ...     end_inclusive=True,
        ...     has={"discharge": "(None, 0)", "death": "(None, 0)"}
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Window boundary reference must be either a valid alphanumeric/'_' string or a reference
        to another window's start or end event, formatted as a valid alphanumeric/'_' string, followed by
        '.start' or '.end'. Got: 'start window'
        >>> invalid_window = WindowConfig(
        ...     start="input",
        ...     end="window.foo -> discharge_or_death",
        ...     start_inclusive=False,
        ...     end_inclusive=True,
        ...     has={"discharge": "(None, 0)", "death": "(None, 0)"}
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Window boundary reference must be either a valid alphanumeric/'_' string or a reference
        to another window's start or end event, formatted as a valid alphanumeric/'_' string, followed by
        '.start' or '.end'. Got: 'window.foo'
        >>> invalid_window = WindowConfig(
        ...     start=None, end=None, start_inclusive=True, end_inclusive=True, has={}
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Window cannot progress from the start of the record to the end of the record.
        >>> invalid_window = WindowConfig(
        ...     start="input.end",
        ...     end="start - 2d",
        ...     start_inclusive=False,
        ...     end_inclusive=True,
        ...     has={"discharge": "(None, 0)", "death": "(None, 0)"}
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Window start will not occur before window end! Got: input.end -> start - 2d
        >>> invalid_window = WindowConfig(
        ...     start="end -> predicate",
        ...     end="input.end",
        ...     start_inclusive=False,
        ...     end_inclusive=True,
        ...     has={"discharge": "(None, 0)", "death": "(None, 0)"}
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Window start will not occur before window end! Got: end -> predicate -> input.end
        >>> invalid_window = WindowConfig(
        ...     start="end - 24h", end="start + 1d", start_inclusive=True, end_inclusive=True, has={}
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Exactly one of the start or end of the window must reference the other.
        Got: end - 24h -> start + 1d
        >>> invalid_window = WindowConfig(
        ...     start="input.end",
        ...     end="input.end + 2d",
        ...     start_inclusive=False,
        ...     end_inclusive=True,
        ...     has={"discharge": "(None, 0)", "death": "(None, 0)"}
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Exactly one of the start or end of the window must reference the other.
        Got: input.end -> input.end + 2d
        >>> invalid_window = WindowConfig(
        ...     start="input.end",
        ...     end="start + -24h",
        ...     start_inclusive=False,
        ...     end_inclusive=True,
        ...     has={"discharge": "(None, 0)", "death": "(None, 0)"}
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Window boundary cannot contain both '+' and '-' operators.
        >>> invalid_window = WindowConfig(
        ...     start="input.end",
        ...     end="start + invalid time string.",
        ...     start_inclusive=False,
        ...     end_inclusive=True,
        ...     has={"discharge": "(None, 0)", "death": "(None, 0)"}
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Failed to parse timedelta from window offset for 'invalid time string.'
        >>> target_window = WindowConfig(
        ...     start="gap.end",
        ...     end="start <-> discharge_or_death",
        ...     start_inclusive=False,
        ...     end_inclusive=True,
        ...     has={}
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Window boundary cannot contain both '->' and '<-' operators.
        >>> invalid_window = WindowConfig(
        ...     start="input.end",
        ...     end="input.end + 2d",
        ...     start_inclusive=False,
        ...     end_inclusive=True,
        ...     has={"discharge": "(0)", "death": "(None, 0)"}
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Invalid constraint format: discharge.
        Expected format: '(min, max)'. Got: '(0)'
    """

    start: str | None
    end: str | None
    start_inclusive: bool
    end_inclusive: bool
    has: dict[str, str] = field(default_factory=dict)
    label: str | None = None
    index_timestamp: str | None = None

    @classmethod
    def _check_reference(cls, reference: str) -> None:
        """Checks to ensure referenced events are valid."""
        err_str = (
            "Window boundary reference must be either a valid alphanumeric/'_' string "
            "or a reference to another window's start or end event, formatted as a valid "
            f"alphanumeric/'_' string, followed by '.start' or '.end'. Got: '{reference}'"
        )

        if "." in reference:
            if reference.count(".") > 1:
                raise ValueError(err_str)
            window, event = reference.split(".")
            if event not in {"start", "end"} or not re.match(r"^\w+$", window):
                raise ValueError(err_str)
        elif not re.match(r"^\w+$", reference):
            raise ValueError(err_str)

    @classmethod
    def _parse_boundary(cls, boundary: str) -> dict[str, str]:
        if "->" in boundary or "<-" in boundary:
            if "->" in boundary and "<-" in boundary:
                raise ValueError("Window boundary cannot contain both '->' and '<-' operators.")
            elif "->" in boundary:
                ref, predicate = (x.strip() for x in boundary.split("->"))
            else:
                ref, predicate = (x.strip() for x in boundary.split("<-"))
                predicate = "-" + predicate

            cls._check_reference(ref)

            return {
                "referenced": ref,
                "offset": None,
                "event_bound": predicate,
                "occurs_before": "-" in predicate,
            }
        elif "+" in boundary or "-" in boundary:
            if "+" in boundary and "-" in boundary:
                raise ValueError("Window boundary cannot contain both '+' and '-' operators.")
            elif "+" in boundary:
                ref, offset = (x.strip() for x in boundary.split("+"))
            else:
                ref, offset = (x.strip() for x in boundary.split("-"))
                offset = "-" + offset

            cls._check_reference(ref)

            try:
                parsed_offset = parse_timedelta(offset)
                if parsed_offset == timedelta(0):
                    logger.warning(f"Window offset for {boundary} is zero; this may not be intended.")
                    return {"referenced": ref, "offset": None, "event_bound": None, "occurs_before": None}
            except (ValueError, TypeError) as e:
                raise ValueError(f"Failed to parse timedelta from window offset for '{offset}'") from e

            return {"referenced": ref, "offset": offset, "event_bound": None, "occurs_before": "-" in offset}
        else:
            ref = boundary.strip()
            cls._check_reference(ref)
            return {"referenced": ref, "offset": None, "event_bound": None, "occurs_before": None}

    def __post_init__(self) -> None:
        # Parse the has constraints from the string representation to the tuple representation
        if self.has is not None:
            for each_constraint in self.has:
                elements = self.has[each_constraint].strip("()").split(",")
                elements = [element.strip() for element in elements]
                if len(elements) != 2:
                    raise ValueError(
                        f"Invalid constraint format: {each_constraint}. "
                        f"Expected format: '(min, max)'. Got: '{self.has[each_constraint]}'"
                    )
                self.has[each_constraint] = tuple(
                    int(element) if element not in ("None", "") else None for element in elements
                )

        if self.start is None and self.end is None:
            raise ValueError("Window cannot progress from the start of the record to the end of the record.")

        if self.start is None:
            self._parsed_start = {
                "referenced": "end",
                "offset": None,
                "event_bound": f"-{START_OF_RECORD_KEY}",
                "occurs_before": True,
            }
        else:
            self._parsed_start = self._parse_boundary(self.start)

        if self.end is None:
            self._parsed_end = {
                "referenced": "start",
                "offset": None,
                "event_bound": END_OF_RECORD_KEY,
                "occurs_before": False,
            }
        else:
            self._parsed_end = self._parse_boundary(self.end)

        if self._parsed_start["referenced"] == "end" and self._parsed_end["referenced"] == "start":
            raise ValueError(
                "Exactly one of the start or end of the window must reference the other. "
                f"Got: {self.start} -> {self.end}"
            )
        elif self._parsed_start["referenced"] == "end":
            self._start_references_end = True
            # We use `is False` because it may be None, which is distinct from True or False
            if self._parsed_start["occurs_before"] is False:
                raise ValueError(
                    f"Window start will not occur before window end! Got: {self.start} -> {self.end}"
                )
        elif self._parsed_end["referenced"] == "start":
            self._start_references_end = False
            # We use `is True` because it may be None, which is distinct from True or False
            if self._parsed_end["occurs_before"] is True:
                raise ValueError(
                    f"Window start will not occur before window end! Got: {self.start} -> {self.end}"
                )
        else:
            raise ValueError(
                "Exactly one of the start or end of the window must reference the other. "
                f"Got: {self.start} -> {self.end}"
            )

    @property
    def root_node(self) -> str:
        """Returns 'start' if the end of the window is defined relative to the start and 'end' otherwise."""
        return "end" if self._start_references_end else "start"

    @property
    def referenced_event(self) -> tuple[str]:
        if self._start_references_end:
            return tuple(self._parsed_end["referenced"].split("."))
        else:
            return tuple(self._parsed_start["referenced"].split("."))

    @property
    def constraint_predicates(self) -> set[str]:
        predicates = set(self.has.keys())
        return predicates

    @property
    def referenced_predicates(self) -> set[str]:
        predicates = set(self.has.keys())
        if self._parsed_start["event_bound"]:
            predicates.add(self._parsed_start["event_bound"].replace("-", ""))
        if self._parsed_end["event_bound"]:
            predicates.add(self._parsed_end["event_bound"].replace("-", ""))

        predicates -= {START_OF_RECORD_KEY, END_OF_RECORD_KEY}
        return predicates

    @property
    def start_endpoint_expr(self) -> None | ToEventWindowBounds | TemporalWindowBounds:
        if self._start_references_end:
            # If end references start, then end will occur after start, so `left_inclusive` corresponds to
            # `start_inclusive` and `right_inclusive` corresponds to `end_inclusive`.
            left_inclusive = self.start_inclusive
            right_inclusive = self.end_inclusive
        else:
            # If this window references end from start, then the end event window expression will not have
            # any constraints as it will reference an external event, and therefore the inclusive
            # parameters don't matter.
            left_inclusive = False
            right_inclusive = False

        if self._parsed_start["event_bound"]:
            return ToEventWindowBounds(
                end_event=self._parsed_start["event_bound"],
                left_inclusive=left_inclusive,
                right_inclusive=right_inclusive,
            )
        elif self._parsed_start["offset"]:
            return TemporalWindowBounds(
                window_size=parse_timedelta(self._parsed_start["offset"]),
                left_inclusive=left_inclusive,
                right_inclusive=right_inclusive,
            )
        else:
            return None

    @property
    def end_endpoint_expr(self) -> None | ToEventWindowBounds | TemporalWindowBounds:
        if self._start_references_end:
            # If this window references end from start, then the end event window expression will not have
            # any constraints as it will reference an external event, and therefore the inclusive
            # parameters don't matter.
            left_inclusive = False
            right_inclusive = False
        else:
            # If end references start, then end will occur after start, so `left_inclusive` corresponds to
            # `start_inclusive` and `right_inclusive` corresponds to `end_inclusive`.
            left_inclusive = self.start_inclusive
            right_inclusive = self.end_inclusive

        if self._parsed_end["event_bound"]:
            return ToEventWindowBounds(
                end_event=self._parsed_end["event_bound"],
                left_inclusive=left_inclusive,
                right_inclusive=right_inclusive,
            )
        elif self._parsed_end["offset"]:
            return TemporalWindowBounds(
                window_size=parse_timedelta(self._parsed_end["offset"]),
                left_inclusive=left_inclusive,
                right_inclusive=right_inclusive,
            )
        else:
            return None


@dataclasses.dataclass
class EventConfig:
    """A configuration object for defining the event that triggers the task extraction process.

    This is defined by all events that match a simple predicate. This event serves as the root of the window
    tree, and its form is dictated by the fact that we must be able to localize the tree to identify valid
    realizations of the tree.

    Examples:
        >>> event = EventConfig("event_type//ADMISSION")
        >>> event.predicate
        'event_type//ADMISSION'
    """

    predicate: str


@dataclasses.dataclass
class TaskExtractorConfig:
    """A configuration object for parsing the plain-data stored in a task extractor config.

    This class can be serialized to and deserialized from a YAML file, and is largely a collection of
    utilities to parse, validate, and leverage task extraction configuration data in practice. There is no
    state stored in this class that is not present or recoverable from the source YAML file on disk. It also
    can be read from a simplified, "user-friendly" language, which can also be stored on or read from disk,
    which is ultimately parsed into the expansive, full specification contained in the YAML file referenced
    above.

    Args:
        predicates: A dictionary of predicate configurations, stored as either plain or derived predicate
            configuration objects (which are simple dataclasses with utility functions over plain
            dictionaries).
        trigger: The event configuration that triggers the task extraction process. This is a simple
            dataclass with a single field, the name of the predicate that triggers the task extraction and
            serves as the root of the window tree.
        windows: A dictionary of window configurations. Each window configuration is a simple dataclass with
            that can be materialized to/from a simple, POD dictionary.

    Raises:
        ValueError: If any window or predicate names are not composed of alphanumeric or "_" characters.

    Examples:
        >>> from bigtree import print_tree
        >>> predicates = {
        ...     "admission": PlainPredicateConfig("admission"),
        ...     "discharge": PlainPredicateConfig("discharge"),
        ...     "death": PlainPredicateConfig("death"),
        ...     "death_or_discharge": DerivedPredicateConfig("or(death, discharge)"),
        ...     "diabetes_icd9": PlainPredicateConfig("ICD9CM//250.02"),
        ...     "diabetes_icd10": PlainPredicateConfig("ICD10CM//E11.65"),
        ...     "diabetes": DerivedPredicateConfig("or(diabetes_icd9, diabetes_icd10)"),
        ...     "diabetes_and_discharge": DerivedPredicateConfig("and(diabetes, discharge)"),
        ... }
        >>> trigger = EventConfig("admission")
        >>> windows = {
        ...     "input": WindowConfig(
        ...         start=None,
        ...         end="trigger + 24h",
        ...         start_inclusive=True,
        ...         end_inclusive=True,
        ...         has={"_ANY_EVENT": "(32, None)"},
        ...         index_timestamp="end",
        ...     ),
        ...     "gap": WindowConfig(
        ...         start="input.end",
        ...         end="start + 24h",
        ...         start_inclusive=False,
        ...         end_inclusive=True,
        ...         has={"death_or_discharge": "(None, 0)", "admission": "(None, 0)"},
        ...     ),
        ...     "target": WindowConfig(
        ...         start="gap.end",
        ...         end="start -> death_or_discharge",
        ...         start_inclusive=False,
        ...         end_inclusive=True,
        ...         has={},
        ...         label="death",
        ...     ),
        ... }
        >>> config = TaskExtractorConfig(predicates=predicates, trigger=trigger, windows=windows)
        >>> print(config.plain_predicates)
        {'admission': PlainPredicateConfig(code='admission',
                              value_min=None,
                              value_max=None,
                              value_min_inclusive=None,
                              value_max_inclusive=None,
                              static=False,
                              other_cols={}),
         'discharge': PlainPredicateConfig(code='discharge',
                              value_min=None,
                              value_max=None,
                              value_min_inclusive=None,
                              value_max_inclusive=None,
                              static=False,
                              other_cols={}),
         'death': PlainPredicateConfig(code='death',
                              value_min=None,
                              value_max=None,
                              value_min_inclusive=None,
                              value_max_inclusive=None,
                              static=False,
                              other_cols={}),
         'diabetes_icd9': PlainPredicateConfig(code='ICD9CM//250.02',
                              value_min=None,
                              value_max=None,
                              value_min_inclusive=None,
                              value_max_inclusive=None,
                              static=False,
                              other_cols={}),
         'diabetes_icd10': PlainPredicateConfig(code='ICD10CM//E11.65',
                              value_min=None,
                              value_max=None,
                              value_min_inclusive=None,
                              value_max_inclusive=None,
                              static=False,
                              other_cols={})}
        >>> print(config.label_window)
        target
        >>> print(config.index_timestamp_window)
        input
        >>> print(config.derived_predicates)
        {'death_or_discharge': DerivedPredicateConfig(expr='or(death, discharge)', static=False),
         'diabetes': DerivedPredicateConfig(expr='or(diabetes_icd9, diabetes_icd10)', static=False),
         'diabetes_and_discharge': DerivedPredicateConfig(expr='and(diabetes, discharge)', static=False)}
        >>> print(nx.write_network_text(config.predicates_DAG))
        ╟── death
        ╎   └─╼ death_or_discharge ╾ discharge
        ╟── discharge
        ╎   ├─╼ diabetes_and_discharge ╾ diabetes
        ╎   └─╼  ...
        ╟── diabetes_icd9
        ╎   └─╼ diabetes ╾ diabetes_icd10
        ╎       └─╼  ...
        ╙── diabetes_icd10
            └─╼  ...
        >>> print_tree(config.window_tree)
        trigger
        └── input.end
            ├── input.start
            └── gap.end
                └── target.end

    Configs will error out in various ways when passed inappropriate arguments:
        >>> config_path = "/foo/non_existent_file.yaml"
        >>> cfg = TaskExtractorConfig.load(config_path)
        Traceback (most recent call last):
            ...
        FileNotFoundError: Cannot load missing configuration file /foo/non_existent_file.yaml!
        >>> with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as f:
        ...     config_path = Path(f.name)
        ...     cfg = TaskExtractorConfig.load(config_path)
        Traceback (most recent call last):
            ...
        ValueError: Only supports reading from '.yaml'. Got: '.txt' in ....txt'.
        >>> predicates_path = "/foo/non_existent_predicates.yaml"
        >>> with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
        ...     config_path = Path(f.name)
        ...     cfg = TaskExtractorConfig.load(config_path, predicates_path)
        Traceback (most recent call last):
            ...
        FileNotFoundError: Cannot load missing predicates file /foo/non_existent_predicates.yaml!
        >>> with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as f:
        ...     predicates_path = Path(f.name)
        ...     with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f2:
        ...         config_path = Path(f2.name)
        ...         cfg = TaskExtractorConfig.load(config_path, predicates_path)
        Traceback (most recent call last):
            ...
        ValueError: Only supports reading from '.yaml'. Got: '.txt' in ....txt'.
        >>> data = {
        ...     'predicates': {},
        ...     'trigger': {},
        ...     'foo': {}
        ... }
        >>> with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
        ...     config_path = Path(f.name)
        ...     yaml.dump(data, f)
        ...     cfg = TaskExtractorConfig.load(config_path)
        Traceback (most recent call last):
            ...
        ValueError: Unrecognized keys in configuration file: 'foo'
        >>> with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
        ...     predicates_path = Path(f.name)
        ...     yaml.dump(data, f)
        ...     with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f2:
        ...         config_path = Path(f2.name)
        ...         cfg = TaskExtractorConfig.load(config_path, predicates_path)
        Traceback (most recent call last):
            ...
        ValueError: Unrecognized keys in configuration file: 'foo, trigger'
        >>> predicates = {"foo bar": PlainPredicateConfig("foo")}
        >>> trigger = EventConfig("foo")
        >>> TaskExtractorConfig(predicates=predicates, trigger=trigger, windows={})
        Traceback (most recent call last):
            ...
        ValueError: Predicate name 'foo bar' is invalid; must be composed of alphanumeric or '_' characters.
        >>> predicates = {"foo": str("foo")}
        >>> trigger = EventConfig("foo")
        >>> TaskExtractorConfig(predicates=predicates, trigger=trigger, windows={})
        Traceback (most recent call last):
            ...
        ValueError: Invalid predicate configuration for 'foo': foo. Must be either a PlainPredicateConfig or
        DerivedPredicateConfig object. Got: <class 'str'>
        >>> predicates = {
        ...     "foo": PlainPredicateConfig("foo"),
        ...     "foobar": DerivedPredicateConfig("or(foo, bar)"),
        ... }
        >>> trigger = EventConfig("foo")
        >>> TaskExtractorConfig(predicates=predicates, trigger=trigger, windows={})
        Traceback (most recent call last):
            ...
        KeyError: "Missing 1 relationships: Derived predicate 'foobar' references undefined predicate 'bar'"
        >>> predicates = {"foo": PlainPredicateConfig("foo")}
        >>> trigger = EventConfig("foo")
        >>> windows = {"foo bar": WindowConfig("gap.end", "start + 24h", True, True)}
        >>> TaskExtractorConfig(predicates=predicates, trigger=trigger, windows=windows)
        Traceback (most recent call last):
            ...
        ValueError: Window name 'foo bar' is invalid; must be composed of alphanumeric or '_' characters.
        >>> windows = {"foo": WindowConfig("gap.end", "start + 24h", True, True, {}, "bar")}
        >>> TaskExtractorConfig(predicates=predicates, trigger=trigger, windows=windows)
        Traceback (most recent call last):
            ...
        ValueError: Label must be one of the defined predicates. Got: bar for window 'foo'
        >>> windows = {"foo": WindowConfig("gap.end", "start + 24h", True, True, {}, "foo", "bar")}
        >>> TaskExtractorConfig(predicates=predicates, trigger=trigger, windows=windows)
        Traceback (most recent call last):
            ...
        ValueError: Index timestamp must be either 'start' or 'end'. Got: bar for window 'foo'
        >>> windows = {
        ...     "foo": WindowConfig("gap.end", "start + 24h", True, True, {}, "foo"),
        ...     "bar": WindowConfig("gap.end", "start + 24h", True, True, {}, "foo")
        ... }
        >>> TaskExtractorConfig(predicates=predicates, trigger=trigger, windows=windows)
        Traceback (most recent call last):
            ...
        ValueError: Only one window can be labeled, found 2 labeled windows: foo, bar
        >>> windows = {
        ...     "foo": WindowConfig("gap.end", "start + 24h", True, True, {}, "foo", "start"),
        ...     "bar": WindowConfig("gap.end", "start + 24h", True, True, {}, index_timestamp="start")
        ... }
        >>> TaskExtractorConfig(predicates=predicates, trigger=trigger, windows=windows)
        Traceback (most recent call last):
            ...
        ValueError: Only the 'start'/'end' of one window can be used as the index timestamp, found
        2 windows with index_timestamp: foo, bar
        >>> predicates = {"foo": PlainPredicateConfig("foo")}
        >>> TaskExtractorConfig(predicates=predicates, trigger=EventConfig("bar"), windows={})
        Traceback (most recent call last):
            ...
        KeyError: "Trigger event predicate 'bar' not found in predicates: foo"
    """

    predicates: dict[str, PlainPredicateConfig | DerivedPredicateConfig]
    trigger: EventConfig
    windows: dict[str, WindowConfig] | None
    label_window: str | None = None
    index_timestamp_window: str | None = None

    @classmethod
    def load(
        cls: TaskExtractorConfig,
        config_path: str | Path,
        predicates_path: str | Path | None = None,
    ) -> TaskExtractorConfig:
        """Load a configuration file from the given path and return it as a dict.

        Args:
            config_path: The path to which a configuration object will be read from in YAML form.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not a ".yaml" file.

        Examples:
            >>> yaml = ruamel.yaml.YAML(typ="safe", pure=True)
            >>> config_dict = {
            ...     "metadata": {'description': 'A test configuration file'},
            ...     "description": 'this is a test',
            ...     "predicates": {"admission": {"code": "admission"}},
            ...     "trigger": "admission",
            ...     "windows": {
            ...         "start": {
            ...             "start": None, "end": "trigger + 24h", "start_inclusive": True,
            ...             "end_inclusive": True,
            ...         }
            ...     },
            ... }
            >>> with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
            ...     config_path = Path(f.name)
            ...     yaml.dump(config_dict, f)
            ...     cfg = TaskExtractorConfig.load(config_path)
            >>> cfg
            TaskExtractorConfig(predicates={'admission': PlainPredicateConfig(code='admission',
                                                           value_min=None, value_max=None,
                                                           value_min_inclusive=None, value_max_inclusive=None,
                                                           static=False, other_cols={})},
                                trigger=EventConfig(predicate='admission'),
                                windows={'start': WindowConfig(start=None, end='trigger + 24h',
                                                    start_inclusive=True, end_inclusive=True, has={},
                                                    label=None, index_timestamp=None)},
                                label_window=None, index_timestamp_window=None)

            >>> predicates_dict = {
            ...     "metadata": {'description': 'A test predicates file'},
            ...     "description": 'this is a test',
            ...     "patient_demographics": {"brown_eyes": {"code": "eye_color//BR"}},
            ...     "predicates": {"admission": {"code": "admission"}},
            ... }
            >>> no_predicates_config = {k: v for k, v in config_dict.items() if k != "predicates"}
            >>> with (tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as config_fp,
            ...      tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as pred_fp):
            ...     config_path = Path(config_fp.name)
            ...     pred_path = Path(pred_fp.name)
            ...     yaml.dump(no_predicates_config, config_fp)
            ...     yaml.dump(predicates_dict, pred_fp)
            ...     cfg = TaskExtractorConfig.load(config_path, pred_path)
            >>> cfg
            TaskExtractorConfig(predicates={'admission': PlainPredicateConfig(code='admission',
                                                           value_min=None, value_max=None,
                                                           value_min_inclusive=None, value_max_inclusive=None,
                                                           static=False, other_cols={}),
                                            'brown_eyes': PlainPredicateConfig(code='eye_color//BR',
                                                            value_min=None, value_max=None,
                                                            value_min_inclusive=None,
                                                            value_max_inclusive=None, static=True,
                                                            other_cols={})},
                                trigger=EventConfig(predicate='admission'),
                                windows={'start': WindowConfig(start=None, end='trigger + 24h',
                                                    start_inclusive=True, end_inclusive=True, has={},
                                                    label=None, index_timestamp=None)},
                                label_window=None, index_timestamp_window=None)

            >>> config_dict = {
            ...     "metadata": {'description': 'A test configuration file'},
            ...     "description": 'this is a test for joining static and plain predicates',
            ...     "patient_demographics": {"male": {"code": "MALE"}, "female": {"code": "FEMALE"}},
            ...     "predicates": {"normal_male_lab_range": {"code": "LAB", "value_min": 0, "value_max": 100,
            ...                  "value_min_inclusive": True, "value_max_inclusive": True},
            ...                  "normal_female_lab_range": {"code": "LAB", "value_min": 0, "value_max": 90,
            ...                  "value_min_inclusive": True, "value_max_inclusive": True},
            ...                  "normal_lab_male": {"expr": "and(normal_male_lab_range, male)"},
            ...                  "normal_lab_female": {"expr": "and(normal_female_lab_range, female)"}},
            ...     "trigger": "_ANY_EVENT",
            ...     "windows": {
            ...         "start": {
            ...             "start": None, "end": "trigger + 24h", "start_inclusive": True,
            ...             "end_inclusive": True, "has": {"normal_lab_male": "(1, None)"},
            ...         }
            ...     },
            ... }
            >>> with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
            ...     config_path = Path(f.name)
            ...     yaml.dump(config_dict, f)
            ...     cfg = TaskExtractorConfig.load(config_path)
            >>> cfg.predicates.keys()
            dict_keys(['normal_lab_male', 'normal_male_lab_range', 'female', 'male'])

            >>> config_dict = {
            ...     "metadata": {'description': 'A test configuration file'},
            ...     "description": 'this is a test for nested derived predicates',
            ...     "patient_demographics": {"male": {"code": "MALE"}, "female": {"code": "FEMALE"}},
            ...     "predicates": {"abnormally_low_male_lab_range": {"code": "LAB", "value_max": 90,
            ...                  "value_max_inclusive": False},
            ...                  "abnormally_low_female_lab_range": {"code": "LAB", "value_max": 80,
            ...                  "value_max_inclusive": False},
            ...                  "abnormally_high_lab_range": {"code": "LAB", "value_min": 120,
            ...                  "value_min_inclusive": False},
            ...                  "abnormal_lab_male_range": {"expr":
            ...                             "or(abnormally_low_male_lab_range, abnormally_high_lab_range)"},
            ...                  "abnormal_lab_female_range": {"expr":
            ...                             "or(abnormally_low_female_lab_range, abnormally_high_lab_range)"},
            ...                  "abnormal_lab_male": {"expr": "and(abnormal_lab_male_range, male)"},
            ...                  "abnormal_lab_female": {"expr": "and(abnormal_lab_female_range, female)"},
            ...                  "abnormal_labs": {"expr": "or(abnormal_lab_male, abnormal_lab_female)"}},
            ...     "trigger": "_ANY_EVENT",
            ...     "windows": {
            ...         "start": {
            ...             "start": None, "end": "trigger + 24h", "start_inclusive": True,
            ...             "end_inclusive": True, "label": "abnormal_labs",
            ...             "has": {"abnormal_labs": "(1, None)"},
            ...         }
            ...     },
            ... }
            >>> with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
            ...     config_path = Path(f.name)
            ...     yaml.dump(config_dict, f)
            ...     cfg = TaskExtractorConfig.load(config_path)
            >>> cfg.predicates.keys()
            dict_keys(['abnormal_lab_female', 'abnormal_lab_female_range', 'abnormal_lab_male',
            'abnormal_lab_male_range', 'abnormal_labs', 'abnormally_high_lab_range',
            'abnormally_low_female_lab_range', 'abnormally_low_male_lab_range', 'female', 'male'])

            >>> predicates_dict = {
            ...     "metadata": {'description': 'A test predicates file'},
            ...     "description": 'this is a test',
            ...     "patient_demographics": {"brown_eyes": {"code": "eye_color//BR"}},
            ...     "predicates": {'admission': "invalid"},
            ... }
            >>> with (tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as config_fp,
            ...      tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as pred_fp):
            ...     config_path = Path(config_fp.name)
            ...     pred_path = Path(pred_fp.name)
            ...     yaml.dump(no_predicates_config, config_fp)
            ...     yaml.dump(predicates_dict, pred_fp)
            ...     cfg = TaskExtractorConfig.load(config_path, pred_path)
            Traceback (most recent call last):
                ...
            ValueError: Predicate 'admission' is not defined correctly in the configuration file. Currently
            defined as the string: invalid. Please refer to the documentation for the supported formats.
            >>> predicates_dict = {
            ...     "predicates": {'adm': {"code": "admission"}},
            ... }
            >>> with (tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as config_fp,
            ...      tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as pred_fp):
            ...     config_path = Path(config_fp.name)
            ...     pred_path = Path(pred_fp.name)
            ...     yaml.dump(no_predicates_config, config_fp)
            ...     yaml.dump(predicates_dict, pred_fp)
            ...     cfg = TaskExtractorConfig.load(config_path, pred_path)
            Traceback (most recent call last):
                ...
            KeyError: "Something referenced predicate 'admission' that wasn't defined in the configuration."
            >>> config_dict = {
            ...     "predicates": {"A": {"code": "A"}, "B": {"code": "B"}, "A_or_B": {"expr": "or(A, B)"},
            ...                  "A_or_B_and_C": {"expr": "and(A_or_B, C)"}},
            ...     "trigger": "_ANY_EVENT",
            ...     "windows": {"start": {"start": None, "end": "trigger + 24h", "start_inclusive": True,
            ...             "end_inclusive": True, "has": {"A_or_B_and_C": "(1, None)"}}},
            ... }
            >>> with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
            ...     config_path = Path(f.name)
            ...     yaml.dump(config_dict, f)
            ...     cfg = TaskExtractorConfig.load(config_path)
            Traceback (most recent call last):
                ...
            KeyError: "Predicate 'C' referenced in 'A_or_B_and_C' is not defined in the configuration."
        """
        if isinstance(config_path, str):
            config_path = Path(config_path)

        if not config_path.is_file():
            raise FileNotFoundError(f"Cannot load missing configuration file {config_path.resolve()!s}!")

        if config_path.suffix == ".yaml":
            yaml = ruamel.yaml.YAML(typ="safe", pure=True)
            loaded_dict = yaml.load(config_path.read_text())
        else:
            raise ValueError(
                f"Only supports reading from '.yaml'. Got: '{config_path.suffix}' in '{config_path.name}'."
            )

        overriding_predicates = {}
        overriding_demographics = {}
        if predicates_path:
            if isinstance(predicates_path, str):
                predicates_path = Path(predicates_path)

            if not predicates_path.is_file():
                raise FileNotFoundError(f"Cannot load missing predicates file {predicates_path.resolve()!s}!")

            if predicates_path.suffix == ".yaml":
                yaml = ruamel.yaml.YAML(typ="safe", pure=True)
                predicates_dict = yaml.load(predicates_path.read_text())
            else:
                raise ValueError(
                    f"Only supports reading from '.yaml'. Got: '{predicates_path.suffix}' in "
                    f"'{predicates_path.name}'."
                )

            # Remove the description or metadata keys if they exist - currently unused except for readability
            # in the YAML
            _ = predicates_dict.pop("description", None)
            _ = predicates_dict.pop("metadata", None)
            overriding_predicates = predicates_dict.pop("predicates", {})
            overriding_demographics = predicates_dict.pop("patient_demographics", {})

            if predicates_dict:
                raise ValueError(
                    f"Unrecognized keys in configuration file: '{', '.join(predicates_dict.keys())}'"
                )

        # Remove the description or metadata keys if they exist - currently unused except for readability
        # in the YAML
        _ = loaded_dict.pop("description", None)
        _ = loaded_dict.pop("metadata", None)

        trigger = loaded_dict.pop("trigger")
        windows = loaded_dict.pop("windows", None)

        predicates = loaded_dict.pop("predicates", {})
        patient_demographics = loaded_dict.pop("patient_demographics", {})

        if loaded_dict:
            raise ValueError(f"Unrecognized keys in configuration file: '{', '.join(loaded_dict.keys())}'")

        final_predicates = {**predicates, **overriding_predicates}
        final_demographics = {**patient_demographics, **overriding_demographics}
        all_predicates = {**final_predicates, **final_demographics}

        logger.info("Parsing windows...")
        if windows is None:  # pragma: no cover
            windows = {}
            logger.warning(
                "No windows specified in configuration file. Extracting only matching trigger events."
            )
        else:
            windows = {n: WindowConfig(**w) for n, w in windows.items()}

        logger.info("Parsing trigger event...")
        trigger = EventConfig(trigger)

        # add window referenced predicates
        referenced_predicates = {pred for w in windows.values() for pred in w.referenced_predicates}

        # add trigger predicate
        referenced_predicates.add(trigger.predicate)

        # add label predicate if it exists and not already added
        label_reference = [w.label for w in windows.values() if w.label]
        if label_reference:
            referenced_predicates.update(set(label_reference))

        special_predicates = {ANY_EVENT_COLUMN, START_OF_RECORD_KEY, END_OF_RECORD_KEY}
        for pred in set(referenced_predicates) - special_predicates:
            if pred not in all_predicates:
                raise KeyError(
                    f"Something referenced predicate '{pred}' that wasn't defined in the configuration."
                )

            if "expr" in all_predicates[pred]:
                stack = list(DerivedPredicateConfig(**all_predicates[pred]).input_predicates)

                while stack:
                    nested_pred = stack.pop()

                    if nested_pred not in all_predicates:
                        raise KeyError(
                            f"Predicate '{nested_pred}' referenced in '{pred}' is not defined in the "
                            "configuration."
                        )

                    # if nested_pred is a DerivedPredicateConfig, unpack input_predicates and add to stack
                    if "expr" in all_predicates[nested_pred]:
                        derived_config = DerivedPredicateConfig(**all_predicates[nested_pred])
                        stack.extend(derived_config.input_predicates)
                        referenced_predicates.add(nested_pred)  # also add itself to referenced_predicates
                    else:
                        # if nested_pred is a PlainPredicateConfig, only add it to referenced_predicates
                        referenced_predicates.add(nested_pred)

        logger.info("Parsing predicates...")
        predicates_to_parse = {k: v for k, v in final_predicates.items() if k in referenced_predicates}
        predicate_objs = {}
        for n, p in predicates_to_parse.items():
            if "expr" in p:
                predicate_objs[n] = DerivedPredicateConfig(**p)
            else:
                if isinstance(p, str):
                    raise ValueError(
                        f"Predicate '{n}' is not defined correctly in the configuration file. "
                        f"Currently defined as the string: {p}. "
                        "Please refer to the documentation for the supported formats."
                    )
                config_data = {k: v for k, v in p.items() if k in PlainPredicateConfig.__dataclass_fields__}
                other_cols = {k: v for k, v in p.items() if k not in config_data}
                predicate_objs[n] = PlainPredicateConfig(**config_data, other_cols=other_cols)

        if final_demographics:
            logger.info("Parsing patient demographics...")
            final_demographics = {
                n: PlainPredicateConfig(**p, static=True) for n, p in final_demographics.items()
            }
            predicate_objs.update(final_demographics)

        return cls(predicates=predicate_objs, trigger=trigger, windows=windows)

    def _initialize_predicates(self) -> None:
        """Initialize the predicates tree from the configuration object and check validity.

        Raises:
            ValueError: If the predicate name is not valid.

        Examples:
            >>> TaskExtractorConfig(
            ...     predicates={
            ...         "A": DerivedPredicateConfig("and(A, B)"),  # A depends on B
            ...         "B": DerivedPredicateConfig("and(B, C)"),  # B depends on C
            ...         "C": DerivedPredicateConfig("and(A, C)"),  # C depends on A (Cyclic dependency)
            ...     },
            ...     trigger=EventConfig("A"),
            ...     windows={},
            ... )
            Traceback (most recent call last):
                ...
            ValueError: Predicate graph is not a directed acyclic graph!
            Cycle found: [('A', 'A')]
            Graph: None
        """

        dag_relationships = []

        for name, predicate in self.predicates.items():
            if re.match(r"^\w+$", name) is None:
                raise ValueError(
                    f"Predicate name '{name}' is invalid; must be composed of alphanumeric or '_' characters."
                )

            match predicate:
                case PlainPredicateConfig():
                    pass
                case DerivedPredicateConfig():
                    for pred in predicate.input_predicates:
                        dag_relationships.append((pred, name))
                case _:
                    raise ValueError(
                        f"Invalid predicate configuration for '{name}': {predicate}. "
                        "Must be either a PlainPredicateConfig or DerivedPredicateConfig object. "
                        f"Got: {type(predicate)}"
                    )

        missing_predicates = []
        for parent, child in dag_relationships:
            if parent not in self.predicates:
                missing_predicates.append(
                    f"Derived predicate '{child}' references undefined predicate '{parent}'"
                )
        if missing_predicates:
            raise KeyError(
                f"Missing {len(missing_predicates)} relationships: " + "; ".join(missing_predicates)
            )

        self._predicate_dag_graph = nx.DiGraph(dag_relationships)
        if not nx.is_directed_acyclic_graph(self._predicate_dag_graph):
            raise ValueError(
                "Predicate graph is not a directed acyclic graph!\n"
                f"Cycle found: {nx.find_cycle(self._predicate_dag_graph)}\n"
                f"Graph: {nx.write_network_text(self._predicate_dag_graph)}"
            )

    def _initialize_windows(self) -> None:
        """Initialize the windows tree from the configuration object and check validity.

        Raises:
            ValueError: If the window name is not valid.

            Examples:
            >>> TaskExtractorConfig(
            ...     predicates={"A": PlainPredicateConfig("A")},
            ...     windows={
            ...         "win1": WindowConfig(None, "trigger", True, False, has={"B": "(1, 0)"}) # B undefined
            ...     },
            ...     trigger=EventConfig("_ANY_EVENT"),
            ... )
            Traceback (most recent call last):
                ...
            KeyError: "Window 'win1' references undefined predicate 'B'.
            Window predicates: B;
            Defined predicates: A"
            >>> TaskExtractorConfig(
            ...     predicates={"A": PlainPredicateConfig("A")},
            ...     windows={
            ...         "win1": WindowConfig(None, "event_not_trigger", True, False)
            ...     },
            ...     trigger=EventConfig("_ANY_EVENT"),
            ... )
            Traceback (most recent call last):
                ...
            KeyError: "Window 'win1' references undefined trigger event
            'event_not_trigger' -- must be trigger!"
            >>> TaskExtractorConfig(
            ...     predicates={"A": PlainPredicateConfig("A")},
            ...     windows={
            ...         "win1": WindowConfig("win2.end", "start -> A", True, False)
            ...     },
            ...     trigger=EventConfig("_ANY_EVENT"),
            ... )
            Traceback (most recent call last):
                ...
            KeyError: "Window 'win1' references undefined window 'win2' for event 'end'.
            Allowed windows: win1"
        """

        for name in self.windows:
            if re.match(r"^\w+$", name) is None:
                raise ValueError(
                    f"Window name '{name}' is invalid; must be composed of alphanumeric or '_' characters."
                )

        label_windows = []
        index_timestamp_windows = []
        for name, window in self.windows.items():
            if window.label:
                if window.label not in self.predicates:
                    raise ValueError(
                        f"Label must be one of the defined predicates. Got: {window.label} "
                        f"for window '{name}'"
                    )
                label_windows.append(name)
            if window.index_timestamp:
                if window.index_timestamp not in ["start", "end"]:
                    raise ValueError(
                        f"Index timestamp must be either 'start' or 'end'. Got: {window.index_timestamp} "
                        f"for window '{name}'"
                    )
                index_timestamp_windows.append(name)
        if len(label_windows) > 1:
            raise ValueError(
                f"Only one window can be labeled, found {len(label_windows)} labeled windows: "
                f"{', '.join(label_windows)}"
            )
        self.label_window = label_windows[0] if label_windows else None
        if len(index_timestamp_windows) > 1:
            raise ValueError(
                f"Only the 'start'/'end' of one window can be used as the index timestamp, "
                f"found {len(index_timestamp_windows)} windows with index_timestamp: "
                f"{', '.join(index_timestamp_windows)}"
            )
        self.index_timestamp_window = index_timestamp_windows[0] if index_timestamp_windows else None

        if self.trigger.predicate not in self.predicates and self.trigger.predicate not in [
            ANY_EVENT_COLUMN,
            START_OF_RECORD_KEY,
            END_OF_RECORD_KEY,
        ]:
            raise KeyError(
                f"Trigger event predicate '{self.trigger.predicate}' not found in predicates: "
                f"{', '.join(self.predicates.keys())}"
            )

        trigger_node = Node("trigger")

        window_nodes = {"trigger": trigger_node}
        for name, window in self.windows.items():
            start_node = Node(f"{name}.start", endpoint_expr=window.start_endpoint_expr)
            end_node = Node(f"{name}.end", endpoint_expr=window.end_endpoint_expr)

            if window.root_node == "end":
                # In this case, the end_node will bound an unconstrained window, as it is the window between
                # a prior window and the defined anchor for this window, so it has no constraints. But the
                # start_node will have the constraints corresponding to this window, as it is defined relative
                # to the end node.
                end_node.constraints = {}
                start_node.constraints = window.has
                start_node.parent = end_node
            else:
                # In this case, the start_node will bound an unconstrained window, as it is the window between
                # a prior window and the defined anchor for this window, so it has no constraints. But the
                # start_node will have the constraints corresponding to this window, as it is defined relative
                # to the end node.
                end_node.constraints = window.has
                start_node.constraints = {}
                end_node.parent = start_node

            window_nodes[f"{name}.start"] = start_node
            window_nodes[f"{name}.end"] = end_node

        for name, window in self.windows.items():
            for predicate in window.referenced_predicates - {ANY_EVENT_COLUMN}:
                if predicate not in self.predicates:
                    raise KeyError(
                        f"Window '{name}' references undefined predicate '{predicate}'. "
                        f"Window predicates: {', '.join(window.referenced_predicates)}; "
                        f"Defined predicates: {', '.join(self.predicates.keys())}"
                    )

            if len(window.referenced_event) == 1:
                event = window.referenced_event[0]
                if event != "trigger":
                    raise KeyError(
                        f"Window '{name}' references undefined trigger event '{event}' -- must be trigger!"
                    )

                window_nodes[f"{name}.{window.root_node}"].parent = window_nodes[event]

            elif len(window.referenced_event) == 2:
                referenced_window, referenced_event = window.referenced_event
                if referenced_window not in self.windows:
                    raise KeyError(
                        f"Window '{name}' references undefined window '{referenced_window}' "
                        f"for event '{referenced_event}'. Allowed windows: {', '.join(self.windows.keys())}"
                    )
                # Might not be needed as valid window event references are already checked (line 660)
                if referenced_event not in {"start", "end"}:  # pragma: no cover
                    raise KeyError(
                        f"Window '{name}' references undefined event '{referenced_event}' "
                        f"for window '{referenced_window}'. Allowed events: 'start', 'end'"
                    )

                parent_node = f"{referenced_window}.{referenced_event}"
                window_nodes[f"{name}.{window.root_node}"].parent = window_nodes[parent_node]
            # Might not be needed as valid window event references are already checked (line 660)
            else:  # pragma: no cover
                raise ValueError(
                    f"Window '{name}' references invalid event '{window.referenced_event}' "
                    "must be of length 1 or 2."
                )

        # Clean up the tree
        nodes_to_remove = []

        # First pass: identify nodes to remove, reassign children's parent
        for n, node in window_nodes.items():
            if n != "trigger" and node.endpoint_expr is None:
                nodes_to_remove.append(n)
                for child in node.children:
                    # Reassign
                    child.parent = node.parent
                    if node.parent and child not in node.parent.children:
                        node.parent.children += (child,)

        # Second pass: remove nodes from parent's children
        for node_name in nodes_to_remove:
            node = window_nodes[node_name]
            if node.parent:
                # Remove
                node.parent.children = [child for child in node.parent.children if child.name != node_name]

        # Delete nodes_to_remove
        for node_name in nodes_to_remove:
            del window_nodes[node_name]

        self.window_nodes = window_nodes

    def __post_init__(self) -> None:
        self._initialize_predicates()
        self._initialize_windows()

    @property
    def window_tree(self) -> Node:
        return self.window_nodes["trigger"]

    @property
    def predicates_DAG(self) -> nx.DiGraph:
        return self._predicate_dag_graph

    @property
    def plain_predicates(self) -> dict[str, PlainPredicateConfig]:
        """Returns a dictionary of plain predicates in {name: code} format."""
        return {p: cfg for p, cfg in self.predicates.items() if cfg.is_plain}

    @property
    def derived_predicates(self) -> OrderedDict[str, DerivedPredicateConfig]:
        """Returns an ordered dictionary mapping derived predicates to their configs in a proper order."""
        return {
            p: self.predicates[p]
            for p in nx.topological_sort(self.predicates_DAG)
            if not self.predicates[p].is_plain
        }
