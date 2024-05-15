"""This module contains functions for loading and parsing the configuration file and subsequently building a
tree structure from the configuration."""

from __future__ import annotations

import dataclasses
import re
from datetime import timedelta
from pathlib import Path
from typing import Any

import networkx as nx
import polars as pl
import ruamel.yaml
from bigtree import Node
from loguru import logger

from .types import END_OF_RECORD_KEY, START_OF_RECORD_KEY
from .utils import parse_timedelta


@dataclasses.dataclass
class PlainPredicateConfig:
    code: str
    value_min: float | None = None
    value_max: float | None = None
    value_min_inclusive: bool | None = None
    value_max_inclusive: bool | None = None

    def MEDS_eval_expr(self) -> pl.Expr:
        """Returns a Polars expression that evaluates this predicate for a MEDS formatted dataset.

        Examples:
            >>> expr = PlainPredicateConfig("BP//systolic", 120, 140, True, False).MEDS_eval_expr()
            >>> print(expr) # doctest: +NORMALIZE_WHITESPACE
            [([([(col("code")) == (String(BP//systolic))]) &
               ([(col("value")) >= (120)])]) &
               ([(col("value")) < (140)])]
            >>> cfg = PlainPredicateConfig("BP//systolic", value_min=120, value_min_inclusive=False)
            >>> expr = cfg.MEDS_eval_expr()
            >>> print(expr) # doctest: +NORMALIZE_WHITESPACE
            [([(col("code")) == (String(BP//systolic))]) & ([(col("value")) > (120)])]
            >>> cfg = PlainPredicateConfig("BP//diastolic")
            >>> expr = cfg.MEDS_eval_expr()
            >>> print(expr) # doctest: +NORMALIZE_WHITESPACE
            [(col("code")) == (String(BP//diastolic))]
        """

        criteria = [pl.col("code") == self.code]

        if self.value_min is not None:
            if self.value_min_inclusive:
                criteria.append(pl.col("value") >= self.value_min)
            else:
                criteria.append(pl.col("value") > self.value_min)
        if self.value_max is not None:
            if self.value_max_inclusive:
                criteria.append(pl.col("value") <= self.value_max)
            else:
                criteria.append(pl.col("value") < self.value_max)

        if len(criteria) == 1:
            return criteria[0]
        else:
            return pl.all_horizontal(criteria)

    def ESGPT_eval_expr(self, values_column: str | None = None) -> pl.Expr:
        """Returns a Polars expression that evaluates this predicate for a MEDS formatted dataset.

        Examples:
            >>> expr = PlainPredicateConfig("BP//systolic", 120, 140, True, False).ESGPT_eval_expr("BP_value")
            >>> print(expr) # doctest: +NORMALIZE_WHITESPACE
            [([([(col("BP")) == (String(systolic))]) &
               ([(col("BP_value")) >= (120)])]) &
               ([(col("BP_value")) < (140)])]
            >>> cfg = PlainPredicateConfig("BP//systolic", value_min=120, value_min_inclusive=False)
            >>> expr = cfg.ESGPT_eval_expr("blood_pressure_value")
            >>> print(expr) # doctest: +NORMALIZE_WHITESPACE
            [([(col("BP")) == (String(systolic))]) & ([(col("blood_pressure_value")) > (120)])]
            >>> expr = PlainPredicateConfig("BP//diastolic").ESGPT_eval_expr()
            >>> print(expr) # doctest: +NORMALIZE_WHITESPACE
            [(col("BP")) == (String(diastolic))]
            >>> expr = PlainPredicateConfig("event_type//ADMISSION").ESGPT_eval_expr()
            >>> print(expr) # doctest: +NORMALIZE_WHITESPACE
            col("event_type").strict_cast(String).str.split([String(&)]).list.contains([String(ADMISSION)])
        """
        code_is_in_parts = "//" in self.code

        if code_is_in_parts:
            measurement_name, code = self.code.split("//")
            if measurement_name.lower() == "event_type":
                criteria = [pl.col("event_type").cast(pl.Utf8).str.split("&").list.contains(code)]
            else:
                criteria = [pl.col(measurement_name) == code]

        elif (self.value_min is None) and (self.value_max is None):
            return pl.col(code).is_not_null()
        else:
            values_column = self.code

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
        >>> pred = DerivedPredicateConfig("and()") # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
            ...
        ValueError: Derived predicate expression must have at least two input predicates (comma separated),
        got and()
        >>> pred = DerivedPredicateConfig("or(PA, PB)")
        >>> pred = DerivedPredicateConfig("PA + PB")
        Traceback (most recent call last):
            ...
        ValueError: Derived predicate expression must start with 'and(' or 'or(', got PA + PB
    """

    expr: str

    def __post_init__(self):
        if not self.expr:
            raise ValueError("Derived predicates must have a non-empty expression field.")

        self.is_and = self.expr.startswith("and(") and self.expr.endswith(")")
        self.is_or = self.expr.startswith("or(") and self.expr.endswith(")")
        if not (self.is_and or self.is_or):
            raise ValueError(f"Derived predicate expression must start with 'and(' or 'or(', got {self.expr}")

        if self.is_and:
            self.input_predicates = [x.strip() for x in self.expr[4:-1].split(",")]
        elif self.is_or:
            self.input_predicates = [x.strip() for x in self.expr[3:-1].split(",")]

        if len(self.input_predicates) < 2:
            raise ValueError(
                "Derived predicate expression must have at least two input predicates (comma separated), "
                f"got {self.expr}"
            )

    def eval_expr(self) -> pl.Expr:
        """Returns a Polars expression that evaluates this predicate against necessary dependent predicates.

        Examples:
            >>> expr = DerivedPredicateConfig("and(P1, P2, P3)").eval_expr()
            >>> print(expr) # doctest: +NORMALIZE_WHITESPACE
            [([([(col("P1")) > (0)]) &
               ([(col("P2")) > (0)])]) &
               ([(col("P3")) > (0)])]
            >>> expr = DerivedPredicateConfig("or(PA, PB)").eval_expr()
            >>> print(expr)
            [([(col("PA")) > (0)]) | ([(col("PB")) > (0)])]
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

    Raises:
        TODO

    Examples: TODO
    """

    start: str | None
    end: str | None
    start_inclusive: bool
    end_inclusive: bool
    has: dict[str, str]

    @classmethod
    def _parse_boundary(cls, boundary: str) -> dict[str, str]:
        if "->" in boundary or "<-" in boundary:
            if "->" in boundary:
                ref, predicate = (x.strip() for x in boundary.split("->"))
            else:
                ref, predicate = (x.strip() for x in boundary.split("<-"))
                predicate = "-" + predicate
            return {"referenced": ref, "offset": None, "event_bound": predicate}
        elif "+" in boundary or "-" in boundary:
            if "+" in boundary:
                ref, offset = (x.strip() for x in boundary.split("+"))
            else:
                ref, offset = (x.strip() for x in boundary.split("-"))
                offset = "-" + offset
            return {"referenced": ref, "offset": offset, "event_bound": None}
        else:
            return {"referenced": boundary, "offset": None, "event_bound": None}

    def __post_init__(self):
        if self.start is None and self.end is None:
            raise ValueError("Window cannot progress from the start of the record to the end of the record.")

        if self.start is None:
            self._parsed_start = {
                "referenced": "end",
                "offset": None,
                "event_bound": f"-{START_OF_RECORD_KEY}",
            }
        else:
            self._parsed_start = self._parse_boundary(self.start)

        if self.end is None:
            self._parsed_end = {"referenced": "start", "offset": None, "event_bound": END_OF_RECORD_KEY}
        else:
            self._parsed_end = self._parse_boundary(self.end)

        if self._parsed_start["referenced"] == "end":
            self._start_references_end = True
        elif self._parsed_end["referenced"] == "start":
            self._start_references_end = False
        else:
            raise ValueError(
                "Window must have either the start reference the end or the end reference the start. "
                f"Got: {self.start} -> {self.end}"
            )

        raise NotImplementedError

    @property
    def referenced_event(self) -> tuple[str]:
        if self._start_references_end:
            return tuple(self._parsed_end["referenced"].split("."))
        else:
            return tuple(self._parsed_start["referenced"].split("."))

    @property
    def referenced_predicates(self) -> set[str]:
        predicates = set()
        if self._parsed_start["event_bound"]:
            predicates.add(self._parsed_start["event_bound"].replace("-", ""))
        if self._parsed_end["event_bound"]:
            predicates.add(self._parsed_end["event_bound"].replace("-", ""))
        return predicates


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
        trigger_event: The event configuration that triggers the task extraction process. This is a simple
            dataclass with a single field, the name of the predicate that triggers the task extraction and
            serves as the root of the window tree.
        windows: A dictionary of window configurations. Each window configuration is a simple dataclass with
            that can be materialized to/from a simple, POD dictionary.

    Raises:
        ValueError: If any window or predicate names are not composed of alphanumeric or "_" characters.

    Examples: TODO
    """

    predicates: dict[str, PlainPredicateConfig | DerivedPredicateConfig]
    trigger_event: EventConfig
    windows: dict[str, WindowConfig]

    @classmethod
    def load(cls, config_path: str | Path) -> TaskExtractorConfig:
        """Load a configuration file from the given path and return it as a dict.

        Args:
            config_path: The path to which a configuration object will be read from in YAML form.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not a ".yaml" file.
        """
        if isinstance(config_path, str):
            config_path = Path(config_path)

        if not config_path.is_file():
            raise FileNotFoundError(f"Cannot load missing configuration file {str(config_path.resolve())}!")

        if config_path.suffix == ".yaml":
            yaml = ruamel.yaml.YAML(typ="safe", pure=True)
            loaded_dict = yaml.load(config_path.read_text())
        else:
            raise ValueError(f"Only supports reading from '.yaml' files currently. Got {config_path.suffix}")

        predicates = loaded_dict.pop("predicates")
        trigger_event = loaded_dict.pop("trigger_event")
        windows = loaded_dict.pop("windows")

        if loaded_dict:
            raise ValueError(f"Unrecognized keys in configuration file: {', '.join(loaded_dict.keys())}")

        predicates = {
            n: DerivedPredicateConfig(**p) if "expr" in p else PlainPredicateConfig(**p)
            for n, p in predicates.items()
        }
        trigger_event = EventConfig(**trigger_event)
        windows = {n: WindowConfig(**w) for n, w in windows.items()}

        return cls(predicates=predicates, trigger_event=trigger_event, windows=windows)

    def save(self, config_path: str | Path, do_overwrite: bool = False):
        """Load a configuration file from the given path and return it as a dict.

        Args:
            config_path: The path to which the calling object will be saved in YAML form.
            do_overwrite: Whether or not to overwrite any existing saved configuration file at that filepath.

        Raises:
            FileExistsError: If there exists a file at the given location and ``do_overwrite`` is not `True`.
            ValueError: If the filepath is not a ".yaml" file.
        """
        if isinstance(config_path, str):
            config_path = Path(config_path)

        if config_path.is_file() and not do_overwrite:
            raise FileExistsError(
                f"Can't overwrite extant {str(config_path.resolve())} as do_overwrite={do_overwrite}"
            )

        if config_path.suffix == ".yaml":
            yaml = ruamel.yaml.YAML(typ="safe", pure=True)
            config_path.write_text(yaml.dump(dataclasses.asdict(self)))
        else:
            raise ValueError(f"Only supports writing to '.yaml' files currently. Got {config_path.suffix}")

    def _initialize_predicates(self):
        """Initialize the predicates tree from the configuration object and check validity.

        Raises:
            ValueError: TODO
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
                f"Missing {len(missing_predicates)} relationships:\n" + "\n".join(missing_predicates)
            )

        self._predicate_dag_graph = nx.DiGraph(dag_relationships)
        if not nx.is_directed_acyclic_graph(self._predicate_dag_graph):
            raise ValueError(
                "Predicate graph is not a directed acyclic graph!\n"
                f"Cycle found: {nx.find_cycle(self._predicate_dag_graph)}\n"
                f"Graph: {nx.generate_network_text(self._predicate_dag_graph)}"
            )

    def _initialize_windows(self):
        """Initialize the windows tree from the configuration object and check validity.

        Raises:
            ValueError: TODO
        """

        for name in self.windows:
            if re.match(r"^\w+$", name) is None:
                raise ValueError(
                    f"Predicate name '{name}' is invalid; must be composed of alphanumeric or '_' characters."
                )

        if self.trigger_event.predicate not in self.predicates:
            raise KeyError(
                f"Trigger event predicate '{self.trigger_event.predicate}' not found in predicates: "
                f"{', '.join(self.predicates.keys())}"
            )

        raise NotImplementedError

    def __post_init__(self):
        self._initialize_predicates()
        self._initialize_windows()

    @property
    def window_tree(self) -> Node:
        raise NotImplementedError

    @property
    def predicates_DAG(self) -> nx.DiGraph:
        return self._predicate_dag_graph

    @property
    def plain_predicates(self) -> list[PlainPredicateConfig]:
        """Returns a list of plain predicates."""
        return [cfg for p, cfg in self.predicates.items() if cfg.is_plain]

    @property
    def derived_predicates(self) -> list[DerivedPredicateConfig]:
        """Returns a list of derived predicates in an order that permits iterative evaluation."""
        return [
            self.predicates[p]
            for p in nx.topological_sort(self.predicates_DAG)
            if not self.predicates[p].is_plain
        ]

    # Simple Form Parsing Methods

    @classmethod
    def parse(cls, raw_input: str | dict) -> TaskExtractorConfig:
        """Parses a simple form suitable for user input into a full configuration object.

        TODO: docstring.
        """
        logger.info("Parsing raw input:")
        logger.info(str(raw_input))

        if isinstance(raw_input, str):
            logger.info("Parsing string input via YAML")
            yaml = ruamel.yaml.YAML(typ="safe", pure=True)
            raw_input = yaml.load(raw_input)

        if set(raw_input.keys()) != {"predicates", "windows"}:
            raise ValueError(
                "Raw input is malformed; must have keys 'predicates' and 'windows' (and no others). "
                f"Got: {', '.join(raw_input.keys())}"
            )

        predicates = cls._parse_predicates(raw_input.pop("predicates"))
        windows, trigger_event = cls._parse_windows(raw_input.pop("windows"))

        return cls(predicates=predicates, windows=windows, trigger_event=trigger_event)

    @classmethod
    def _parse_predicate(cls, predicate_dict: dict) -> PlainPredicateConfig | DerivedPredicateConfig:
        raise NotImplementedError

    @classmethod
    def _parse_predicates(cls, predicates: dict) -> dict[str, PlainPredicateConfig | DerivedPredicateConfig]:
        return {n: cls._parse_predicate(p) for n, p in predicates.items()}

    @classmethod
    def _parse_window(cls, window: dict) -> WindowConfig | EventConfig:
        raise NotImplementedError

    @classmethod
    def _parse_windows(cls, windows: dict) -> tuple[dict[str, WindowConfig], EventConfig]:
        windows = {}
        trigger_event = None
        for name, window in windows.items():
            window = cls._parse_window(window)
            match window:
                case WindowConfig():
                    windows[name] = window
                case EventConfig():
                    trigger_event = window
                case _:
                    raise ValueError(f"Invalid window configuration for '{name}': {window}")
        if trigger_event is None:
            raise ValueError("No trigger event specified in configuration.")
        return windows, trigger_event


def get_config(cfg: dict, key: str, default: Any) -> Any:
    if key not in cfg or cfg[key] is None:
        return default
    return cfg[key]


def build_tree_from_config(cfg: dict[str, Any]) -> Node:
    """Build a tree structure from the given configuration. Note: the parse_timedelta function handles
    negative durations already if duration is specified with "-".

    TODO: Fix up API, configuration language, and tests.

    Args:
        cfg: The configuration object.

    Returns:
        Node: The root node of the built tree.

    Examples:
        >>> cfg = {
        ...     "windows": {
        ...         "window1": {
        ...             "start": 'event1',
        ...             "duration": "24 hours",
        ...             "offset": None,
        ...             "excludes": [],
        ...             "includes": [],
        ...             "st_inclusive": False,
        ...             "end_inclusive": True,
        ...         },
        ...         "window2": {
        ...             "start": "window1.end",
        ...             "duration": "24 hours",
        ...             "end": None,
        ...             "excludes": [],
        ...             "st_inclusive": False,
        ...             "end_inclusive": True,
        ...         },
        ...     }
        ... }
        >>> build_tree_from_config(cfg) # doctest: +NORMALIZE_WHITESPACE
        Node(/window1,
             constraints={},
             endpoint_expr=(False, datetime.timedelta(days=1), True, datetime.timedelta(0)))
    """
    nodes = {}
    windows = [name for name, _ in cfg["windows"].items()]
    for window_name, window_info in cfg["windows"].items():
        defined_keys = [
            key for key in ["start", "end", "duration"] if get_config(window_info, key, None) is not None
        ]
        if len(defined_keys) != 2:
            error_keys = [f"{key}: {window_info[key]}" for key in defined_keys]

            error_lines = [
                f"Invalid window specification for '{window_name}'",
                (
                    "Must specify non-None values for exactly two fields of ['start', 'end', 'duration'], "
                    "and if 'start' is not one of those two, then 'end' must be specified. Got:"
                ),
                ", ".join(error_keys),
            ]
            raise ValueError("\n".join(error_lines))

        node = Node(window_name)

        # set node end_event
        duration = get_config(window_info, "duration", None)
        match duration:
            case timedelta():
                end_event = window_info["duration"]
            case str():
                end_event = parse_timedelta(window_info["duration"])
            case False | None:
                end_event = f"is_{window_info['end']}"
            case _:
                raise ValueError(f"Invalid duration in '{window_name}': {window_info['duration']}")

        # set node st_inclusive and end_inclusive
        st_inclusive = get_config(window_info, "st_inclusive", False)
        end_inclusive = get_config(window_info, "end_inclusive", False)

        # set node offset
        offset = get_config(window_info, "offset", None)
        offset = parse_timedelta(offset)

        node.endpoint_expr = (st_inclusive, end_event, end_inclusive, offset)

        # set node exclude constraints
        constraints = {}
        for each_exclusion in get_config(window_info, "excludes", []):
            if each_exclusion["predicate"]:
                constraints[f"is_{each_exclusion['predicate']}"] = (None, 0)

        # set node include constraints, using min and max values and None if not specified
        for each_inclusion in get_config(window_info, "includes", []):
            if each_inclusion["predicate"]:
                constraints[f"is_{each_inclusion['predicate']}"] = (
                    (
                        int(each_inclusion["min"])
                        if "min" in each_inclusion and each_inclusion["min"] is not None
                        else None
                    ),
                    (
                        int(each_inclusion["max"])
                        if "max" in each_inclusion and each_inclusion["max"] is not None
                        else None
                    ),
                )

        node.constraints = constraints

        # search for the parent node in tree
        if get_config(window_info, "start", None):
            root_name = window_info["start"].split(".")[0]
            node_root = next(
                (substring for substring in windows if substring == root_name),
                None,
            )
        elif get_config(window_info, "end", None):
            root_name = window_info["end"].split(".")[0]
            node_root = next(
                (substring for substring in windows if substring == root_name),
                None,
            )

        if node_root:
            node.parent = nodes[node_root]

        nodes[window_name] = node

    for node in nodes.values():
        if node.parent:
            node.parent = nodes[node.parent.name]

    root = next(iter(nodes.values())).root

    return root
