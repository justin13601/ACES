"""This module contains functions for loading and parsing the configuration file and subsequently building a
tree structure from the configuration."""

from __future__ import annotations

import dataclasses
from datetime import timedelta
from pathlib import Path
from typing import Any

import polars as pl
import ruamel.yaml
from bigtree import Node
from loguru import logger

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


@dataclasses.dataclass
class DerivedPredicateConfig:
    expr: str


@dataclasses.dataclass
class WindowConfig:
    name: str
    start: str
    end: str
    start_inclusive: bool
    end_inclusive: bool
    has: dict[str, str]


@dataclasses.dataclass
class EventConfig:
    predicate: str

    @property
    def name(self) -> str:
        return self.predicate


@dataclasses.dataclass
class TaskExtractorConfig:
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
            return cls(**yaml.load(config_path.read_text()))
        else:
            raise ValueError(f"Only supports reading from '.yaml' files currently. Got {config_path.suffix}")

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
            config_path.write_text(yaml.dump(self.to_dict()))
        else:
            raise ValueError(f"Only supports writing to '.yaml' files currently. Got {config_path.suffix}")

    def __post_init__(self):
        raise NotImplementedError

    @property
    def tree(self) -> Node:
        raise NotImplementedError

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
        windows, trigger_event = cls._parse_windows(raw_input.pop("windows"), predicates)

        return cls(predicates=predicates, windows=windows, trigger_event=trigger_event)

    @classmethod
    def _parse_predicate(cls, predicate_dict: dict) -> PlainPredicateConfig | DerivedPredicateConfig:
        raise NotImplementedError

    @classmethod
    def _parse_predicates(cls, predicates: dict) -> dict[str, PlainPredicateConfig | DerivedPredicateConfig]:
        # base_predicates = {n: cls._parse_predicate(p) for n, p in predicates.items()}

        raise NotImplementedError

    @classmethod
    def _parse_window(cls, window: dict) -> WindowConfig:
        raise NotImplementedError

    @classmethod
    def _parse_windows(
        cls, windows: dict, predicates: dict[str, PlainPredicateConfig | DerivedPredicateConfig]
    ) -> dict[str, WindowConfig] | EventConfig:
        # base_windows = {n: cls._parse_window(w) for n, w in windows.items()}

        raise NotImplementedError


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
