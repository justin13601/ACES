# Usage Guide

## Installation

______________________________________________________________________

To use ACES, first clone the repository and install it using pip:

```bash
(.venv) $ pip install es-aces
```

## Querying Tasks

______________________________________________________________________

To extract a cohort for a particular task, you can use the `aces.query.query()` function:

```{eval-rst}
.. autofunction:: aces.query.query
```

The `cfg` parameter must be of type `dict()`, and the `df_predicates` parameter must be of type `polars.DataFrame()`.
Otherwise, `aces.query.query()` will raise a `TypeError` exception.

Details about the configuration language used to define the `cfg` parameter can be found in {doc}`/configuration`.

For example, assuming `cfg` and `df_predicates` are defined properly, a query can be run using:

```python
>>> from aces import query
>>> query.query(cfg, df_predicates)
```

```plaintext
shape: (1, 7)
┌────────────┬───────────────┬───────────────┬───────────────┬──────────────┬──────────────┬───────┐
│ subject_id ┆ input.start_s ┆ target.end_su ┆ gap.end_summa ┆ subtree_anch ┆ input.end_su ┆ label │
│ ---        ┆ ummary        ┆ mmary         ┆ ry            ┆ or_timestamp ┆ mmary        ┆ ---   │
│ i64        ┆ ---           ┆ ---           ┆ ---           ┆ ---          ┆ ---          ┆ u16   │
│            ┆ struct[8]     ┆ struct[8]     ┆ struct[8]     ┆ datetime[μs] ┆ struct[8]    ┆       │
╞════════════╪═══════════════╪═══════════════╪═══════════════╪══════════════╪══════════════╪═══════╡
│ 1          ┆ {"input.start ┆ {"target.end" ┆ {"gap.end",19 ┆ 1991-01-27   ┆ {"input.end" ┆ 0     │
│            ┆ ",1989-12-01  ┆ ,1991-01-29   ┆ 91-01-28      ┆ 23:32:00     ┆ ,1991-01-27  ┆       │
│            ┆ 12:03:…       ┆ 23:32:0…      ┆ 23:32:00,1…   ┆              ┆ 23:32:00…    ┆       │
└────────────┴───────────────┴───────────────┴───────────────┴──────────────┴──────────────┴───────┘
```
