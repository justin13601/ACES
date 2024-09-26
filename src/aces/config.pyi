from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set, Tuple, Union

import polars as pl

from .types import TemporalWindowBounds, ToEventWindowBounds

@dataclass
class PlainPredicateConfig:
    code: Union[str, dict[str, Any]]
    value_min: Optional[float] = None
    value_max: Optional[float] = None
    value_min_inclusive: Optional[bool] = None
    value_max_inclusive: Optional[bool] = None
    static: bool = False
    other_cols: dict[str, str] = field(default_factory=dict)

    def MEDS_eval_expr(self) -> pl.Expr: ...
    def ESGPT_eval_expr(self, values_column: Optional[str] = None) -> pl.Expr: ...
    @property
    def is_plain(self) -> bool: ...

@dataclass
class DerivedPredicateConfig:
    expr: str
    static: bool = False

    def __post_init__(self): ...
    def eval_expr(self) -> pl.Expr: ...
    @property
    def is_plain(self) -> bool: ...

@dataclass
class WindowConfig:
    start: Optional[str]
    end: Optional[str]
    start_inclusive: bool
    end_inclusive: bool
    has: dict[str, str] = field(default_factory=dict)
    label: Optional[str] = None
    index_timestamp: Optional[str] = None

    @classmethod
    def _check_reference(cls, reference: str) -> None: ...
    @classmethod
    def _parse_boundary(cls, boundary: str) -> dict[str, str]: ...
    def __post_init__(self) -> None: ...
    @property
    def root_node(self) -> str: ...
    @property
    def referenced_event(self) -> Tuple[str]: ...
    @property
    def constraint_predicates(self) -> Set[str]: ...
    @property
    def referenced_predicates(self) -> Set[str]: ...
    @property
    def start_endpoint_expr(self) -> Optional[Union[ToEventWindowBounds, TemporalWindowBounds]]: ...
    @property
    def end_endpoint_expr(self) -> Optional[Union[ToEventWindowBounds, TemporalWindowBounds]]: ...

# Assuming these classes are defined elsewhere
class EventConfig:
    predicate: str

class PlainPredicateConfig:
    pass

class DerivedPredicateConfig:
    input_predicates: Any

class WindowConfig:
    referenced_predicates: set[str]
    label: Optional[str]
    index_timestamp: Optional[str]
    start_endpoint_expr: Optional[str]
    end_endpoint_expr: Optional[str]
    has: dict[str, Any]
    root_node: str
    referenced_event: list[str]

class TaskExtractorConfig:
    predicates: Dict[str, Union[PlainPredicateConfig, DerivedPredicateConfig]]
    trigger: EventConfig
    windows: Optional[Dict[str, WindowConfig]]
    label_window: Optional[str]
    index_timestamp_window: Optional[str]

    @classmethod
    def load(
        cls: Type["TaskExtractorConfig"],
        config_path: Union[str, Path],
        predicates_path: Optional[Union[str, Path]] = None,
    ) -> "TaskExtractorConfig": ...
    def _initialize_predicates(self) -> None: ...
    def _initialize_windows(self) -> None: ...
    def __post_init__(self) -> None: ...
    @property
    def window_tree(self) -> Any: ...
    @property
    def predicates_DAG(self) -> Any: ...
    @property
    def plain_predicates(self) -> Dict[str, PlainPredicateConfig]: ...
    @property
    def derived_predicates(self) -> Dict[str, DerivedPredicateConfig]: ...
