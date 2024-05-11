# Usage Guide

## Installation

______________________________________________________________________

To use ESGPTTaskQuerying, first clone the repository and install it using pip:

```bash
(.venv) $ pip install .
```

## Querying Tasks

______________________________________________________________________

To extract a cohort for a particular task, you can use the `esgpt_task_querying.query.query()` function:

```{eval-rst}
.. autofunction:: esgpt_task_querying.query.query
```

The `cfg` parameter must be of type `dict()`, and the `df_predicates` parameter must be of type `polars.DataFrame()`.
Otherwise, `esgpt_task_querying.query.query()` will raise a `TypeError` exception.

Details about the configuration language used to define the `cfg` parameter can be found in {doc}`/configuration`.

For example, assuming `cfg` and `df_predicates` are defined properly, a query can be run using:

```python
>>> from esgpt_task_querying import query
>>> query.query(cfg, df_predicates)
```

```bash
shape: (4, 9)
┌───────────┬───────────┬───────────┬──────────┬──────────┬──────────┬──────────┬──────────┬───────┐
│ subject_i ┆ trigger/t ┆ gap/times ┆ target/t ┆ input/ti ┆ gap/wind ┆ target/w ┆ input/wi ┆ label │
│ d         ┆ imestamp  ┆ tamp      ┆ imestamp ┆ mestamp  ┆ ow_summa ┆ indow_su ┆ ndow_sum ┆ ---   │
│ ---       ┆ ---       ┆ ---       ┆ ---      ┆ ---      ┆ ry       ┆ mmary    ┆ mary     ┆ i64   │
│ i64       ┆ datetime[ ┆ datetime[ ┆ datetime ┆ datetime ┆ ---      ┆ ---      ┆ ---      ┆       │
│           ┆ μs]       ┆ μs]       ┆ [μs]     ┆ [μs]     ┆ struct[5 ┆ struct[5 ┆ struct[5 ┆       │
│           ┆           ┆           ┆          ┆          ┆ ]        ┆ ]        ┆ ]        ┆       │
╞═══════════╪═══════════╪═══════════╪══════════╪══════════╪══════════╪══════════╪══════════╪═══════╡
│ 1         ┆ 1989-12-0 ┆ 1989-12-0 ┆ 1989-12- ┆ 1989-12- ┆ {0,0,0,0 ┆ {0,1,0,1 ┆ {1,0,0,0 ┆ 0     │
│           ┆ 1         ┆ 2         ┆ 02       ┆ 01       ┆ ,7}      ┆ ,2}      ┆ ,1}      ┆       │
│           ┆ 12:03:00  ┆ 12:03:00  ┆ 15:00:00 ┆ 12:03:00 ┆          ┆          ┆          ┆       │
│ 1         ┆ 1991-01-2 ┆ 1991-01-2 ┆ 1991-01- ┆ 1989-12- ┆ {0,0,0,0 ┆ {0,1,0,1 ┆ {2,1,0,1 ┆ 0     │
│           ┆ 7         ┆ 8         ┆ 31       ┆ 01       ┆ ,4}      ┆ ,8}      ┆ ,12}     ┆       │
│           ┆ 23:32:00  ┆ 23:32:00  ┆ 02:15:00 ┆ 12:03:00 ┆          ┆          ┆          ┆       │
│ 2         ┆ 1996-06-0 ┆ 1996-06-0 ┆ 1996-06- ┆ 1996-03- ┆ {0,0,0,0 ┆ {0,0,1,1 ┆ {2,1,0,1 ┆ 1     │
│           ┆ 5         ┆ 6         ┆ 08       ┆ 08       ┆ ,2}      ┆ ,5}      ┆ ,6}      ┆       │
│           ┆ 00:32:00  ┆ 00:32:00  ┆ 03:00:00 ┆ 02:24:00 ┆          ┆          ┆          ┆       │
│ 3         ┆ 1996-03-0 ┆ 1996-03-0 ┆ 1996-03- ┆ 1996-03- ┆ {0,0,0,0 ┆ {0,0,1,1 ┆ {1,0,0,0 ┆ 1     │
│           ┆ 8         ┆ 9         ┆ 12       ┆ 08       ┆ ,1}      ┆ ,6}      ┆ ,2}      ┆       │
│           ┆ 02:24:00  ┆ 02:24:00  ┆ 00:00:00 ┆ 02:22:00 ┆          ┆          ┆          ┆       │
└───────────┴───────────┴───────────┴──────────┴──────────┴──────────┴──────────┴──────────┴───────┘
```
