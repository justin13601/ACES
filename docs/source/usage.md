# Usage Guide

## Installation

______________________________________________________________________

To use ACES, first determine which data standard you'd like to use. Currently, ACES supports the [MEDS](https://github.com/Medical-Event-Data-Standard/meds) standard and the [EventStreamGPT (ESGPT)](https://github.com/mmcdermott/EventStreamGPT) standard. Please first follow instructions on their respective repositories to install and/or transform your data into one of these standards. Alternatively, you may choose to transform your data into a predicates dataframe `.csv` format - details are provided [here](https://eventstreamaces.readthedocs.io/en/latest/predicates.html).

**Note:** If you choose to use the ESGPT standard, please install ESGPT first before installing ACES. This ensures compatibility with the `polars` version required by ACES.

To install ACES:

```bash
pip install es-aces
```

## Querying Tasks

______________________________________________________________________

To extract a cohort for a particular task, you may use `aces-cli` in your terminal:

```bash
aces-cli --config-dir='/path/to/hydra/config/' --config-name='config.yaml'
```

Alternatively, you can use the `aces.query.query()` function:

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
│ i64        ┆ ---           ┆ ---           ┆ ---           ┆ ---          ┆ ---          ┆ i64   │
│            ┆ struct[8]     ┆ struct[8]     ┆ struct[8]     ┆ datetime[μs] ┆ struct[8]    ┆       │
╞════════════╪═══════════════╪═══════════════╪═══════════════╪══════════════╪══════════════╪═══════╡
│ 1          ┆ {"input.start ┆ {"target.end" ┆ {"gap.end",19 ┆ 1991-01-27   ┆ {"input.end" ┆ 0     │
│            ┆ ",1989-12-01  ┆ ,1991-01-29   ┆ 91-01-28      ┆ 23:32:00     ┆ ,1991-01-27  ┆       │
│            ┆ 12:03:…       ┆ 23:32:0…      ┆ 23:32:00,1…   ┆              ┆ 23:32:00…    ┆       │
└────────────┴───────────────┴───────────────┴───────────────┴──────────────┴──────────────┴───────┘
```
