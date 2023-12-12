"""TODO(justin): Add a module docstring."""


import polars as pl


def has_event_type(type_str: str) -> pl.Expr:
    has_event_type = pl.col("event_type").cast(pl.Utf8).str.contains(type_str)
    # has_event_type = event_types.str.contains(type_str)
    return has_event_type


def generate_predicate_columns(cfg, ESD):
    for predicate_name, predicate_info in cfg.predicates.items():
        if "value" in predicate_info:
            if isinstance(predicate_info["value"], list):
                ESD = ESD.with_columns(
                    pl.when(
                        (
                            pl.col(predicate_info.column)
                            >= (
                                float(predicate_info["value"][0]["min"] or -float("inf"))
                                if "min" in predicate_info["value"][0]
                                else -float("inf")
                            )
                        )
                        & (
                            pl.col(predicate_info.column)
                            <= (
                                float(predicate_info["value"][0]["max"] or float("inf"))
                                if "max" in predicate_info["value"][0]
                                else float("inf")
                            )
                        )
                    )
                    .then(1)
                    .otherwise(0)
                    .alias(f"is_{predicate_name}")
                    .cast(pl.Int32)
                )
                print(f"Added predicate column is_{predicate_name}.")
            else:
                if predicate_info.column == "event_type":
                    ESD = ESD.with_columns(
                        has_event_type(predicate_info["value"]).alias(f"is_{predicate_name}").cast(pl.Int32)
                    )
                else:
                    ESD = ESD.with_columns(
                        pl.when(pl.col(predicate_info.column) == predicate_info["value"])
                        .then(1)
                        .otherwise(0)
                        .alias(f"is_{predicate_name}")
                        .cast(pl.Int32)
                    )
                print(f"Added predicate column is_{predicate_name}.")
        elif "type" in predicate_info:
            if predicate_info.type == "ANY":
                any_expr = pl.col(f"is_{predicate_info.predicates[0]}")
                for predicate in predicate_info.predicates[1:]:
                    any_expr = any_expr | pl.col(f"is_{predicate}")
                ESD = ESD.with_columns(any_expr.alias(f"is_{'_or_'.join(predicate_info.predicates)}"))
                print(f"Added predicate column is_{'_or_'.join(predicate_info.predicates)}.")
            elif predicate_info.type == "ALL":
                all_expr = pl.col(predicate_info.predicates[0])
                for predicate in predicate_info.predicates[1:]:
                    all_expr = all_expr & pl.col(predicate)
                ESD = ESD.with_column(all_expr.alias(f"is_{'_and_'.join(predicate_info.predicates)}"))
                print(f"Added predicate column is_{'_and_'.join(predicate_info.predicates)}.")
            else:
                raise ValueError(f"Invalid predicate type {predicate_info.type}.")

    ESD = ESD.with_columns(pl.when(pl.col("event_type").is_not_null()).then(1).otherwise(0).alias("is_any"))
    return ESD
