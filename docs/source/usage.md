# Usage Guide

## Installation

______________________________________________________________________

To use ACES, first determine which data standard you'd like to use. Currently, ACES supports the [MEDS](https://github.com/Medical-Event-Data-Standard/meds) standard and the [EventStreamGPT (ESGPT)](https://github.com/mmcdermott/EventStreamGPT) standard. Please first follow instructions on their respective repositories to install and/or transform your data into one of these standards. Alternatively, you may choose to transform your data into a predicates dataframe `.csv` format - details are provided [here](https://eventstreamaces.readthedocs.io/en/latest/predicates.html).

**Note:** If you choose to use the ESGPT standard, please install ESGPT first before installing ACES. This ensures compatibility with the `polars` version required by ACES.

To install ACES:

```bash
pip install es-aces
```

Additionally, you may enable shell completion for configuration files. For Bash, please run:

```bash
eval "$(aces-cli -sc install=bash)"
```

Please visit [Hydra's Documentation](https://hydra.cc/docs/tutorials/basic/running_your_app/tab_completion/) for more details.

## Querying Tasks

______________________________________________________________________

To extract a cohort for a particular task, you may use `aces-cli` in your terminal:

```bash
aces-cli data.path='/path/to/data/file/or/directory' data.standard='<esgpt/meds/direct>' cohort_dir='/directory/to/task/config/' cohort_name='<task_config_name>'
```

Alternatively, you can use the `aces.query.query()` function:

```{eval-rst}
.. autofunction:: aces.query.query
```

The `cfg` parameter must be of type `TaskExtractorConfig`, and the `df_predicates` parameter must be of type `polars.DataFrame`.

Details about the configuration language used to define the `cfg` parameter can be found in {doc}`/configuration`.

For example, to query a in-hospital mortality task on the sample data (both configuration file and data are provided in the repository) using the 'direct' predicates method:

```python
>>> from aces import query, predicates, config
>>> from omegaconf import DictConfig

>>> cfg = config.TaskExtractorConfig.load(config_path="/home/justinxu/esgpt/ESGPTTaskQuerying/sample_configs/inhospital-mortality.yaml")

>>> data_config = DictConfig({"path": "sample_data.csv", "standard": "direct", "ts_format": "%m/%d/%Y %H:%M"})
>>> predicates_df = predicates.get_predicates_df(cfg=cfg, data_config=data_config)

>>> query.query(cfg=cfg, predicates_df=predicates_df)
```

```plaintext
shape: (1, 8)
┌────────────┬────────────┬───────┬────────────┬────────────┬────────────┬────────────┬────────────┐
│ subject_id ┆ index_time ┆ label ┆ trigger    ┆ input.star ┆ target.end ┆ gap.end_su ┆ input.end_ │
│ ---        ┆ stamp      ┆ ---   ┆ ---        ┆ t_summary  ┆ _summary   ┆ mmary      ┆ summary    │
│ i64        ┆ ---        ┆ i64   ┆ datetime[μ ┆ ---        ┆ ---        ┆ ---        ┆ ---        │
│            ┆ datetime[μ ┆       ┆ s]         ┆ struct[8]  ┆ struct[8]  ┆ struct[8]  ┆ struct[8]  │
│            ┆ s]         ┆       ┆            ┆            ┆            ┆            ┆            │
╞════════════╪════════════╪═══════╪════════════╪════════════╪════════════╪════════════╪════════════╡
│ 1          ┆ 1991-01-28 ┆ 0     ┆ 1991-01-27 ┆ {"input.st ┆ {"target.e ┆ {"gap.end" ┆ {"input.en │
│            ┆ 23:32:00   ┆       ┆ 23:32:00   ┆ art",1989- ┆ nd",1991-0 ┆ ,1991-01-2 ┆ d",1991-01 │
│            ┆            ┆       ┆            ┆ 12-01      ┆ 1-29       ┆ 8          ┆ -27        │
│            ┆            ┆       ┆            ┆ 12:0…      ┆ 23:32…     ┆ 23:32:00…  ┆ 23:32:…    │
└────────────┴────────────┴───────┴────────────┴────────────┴────────────┴────────────┴────────────┘
```
