# Usage Guide

## Quick Start

### Installation

To use ACES, first determine which data standard you'd like to use. Currently, ACES can be automatically applied to the [Medical Event Data Standard (MEDS)](https://github.com/Medical-Event-Data-Standard/meds) and [EventStreamGPT (ESGPT)](https://github.com/mmcdermott/EventStreamGPT). Please first follow instructions on their respective repositories to install and/or transform your data into one of these standards. Alternatively, ACES also supports ***any*** arbitrary dataset schema, provided you extract the necessary dataset-specific plain predicates and format it **directly** as an event-stream - details are provided [here](https://eventstreamaces.readthedocs.io/en/latest/predicates.html).

**Note:** If you choose to use the ESGPT standard, please install ESGPT first before installing ACES. This ensures compatibility with the `polars` version required by ACES.

**To install ACES:**

```bash
pip install es-aces
```

### Task Configuration Example

**Example: `inhospital_mortality.yaml`**

Please see the [Task Configuration File Overview](https://eventstreamaces.readthedocs.io/en/latest/overview.html#task-configuration-file) for details on how to create this configuration for your own task!

This particular task configuration defines a cohort for the binary prediction of in-hospital mortality 48 hours after admission. Patients with 5 or more records between the start of their record and 24 hours after the admission will be included. The cohort includes both those that have been discharged (label=`0`)  and those that have died (label=`1`).

```yaml
predicates:
  admission:
    code: code//ADMISSION
  discharge:
    code: code//DISCHARGE
  death:
    code: code//DEATH
  discharge_or_death:
    expr: or(discharge, death)

trigger: admission

windows:
  input:
    start:
    end: trigger + 24h
    start_inclusive: true
    end_inclusive: true
    has:
      _ANY_EVENT: (5, None)
    index_timestamp: end
  gap:
    start: trigger
    end: start + 24h
    start_inclusive: false
    end_inclusive: true
    has:
      admission: (None, 0)
      discharge: (None, 0)
      death: (None, 0)
  target:
    start: gap.end
    end: start -> discharge_or_death
    start_inclusive: false
    end_inclusive: true
    label: death
```

### Run the CLI

You can now run `aces-cli` in your terminal. Suppose we have a directory structure like the following:

```
ACES/
├── sample_data/
│   ├── esgpt_sample/
│   │   ├── ...
│   │   ├── events_df.parquet
│   │   └── dynamic_measurements_df.parquet
│   ├── meds_sample/
│   │   ├── shards/
│   │   │   ├── 0.parquet
│   │   │   └── 1.parquet
│   │   └── sample_shard.parquet
│   └── sample_data.csv
├── sample_configs/
│   └── inhospital_mortality.yaml
└── ...
```

**To query from a single MEDS shard**:

```bash
aces-cli cohort_name="inhospital_mortality" cohort_dir="sample_configs/" data.standard=meds data.path="sample_data/meds_sample/sample_shard.parquet"
```

**To query from multiple MEDS shards**:

```bash
aces-cli cohort_name="inhospital_mortality" cohort_dir="sample_configs/" data.standard=meds data=sharded data.root="sample_data/meds_sample/" "data.shard=$(expand_shards shards/5)" -m
```

**To query from ESGPT**:

```bash
aces-cli cohort_name="inhospital_mortality" cohort_dir="sample_configs/" data.standard=esgpt data.path="sample_data/esgpt_sample/"
```

**To query from a direct predicates dataframe (`.csv` | `.parquet`)**:

```bash
aces-cli cohort_name="inhospital_mortality" cohort_dir="sample_configs/" data.standard=direct data.path="sample_data/sample_data.csv"
```

**For help using `aces-cli`:**

```bash
aces-cli --help
```

### Results

By default, results from the above examples would be saved to `sample_configs/inhospital_mortality/shards/0.parquet` for MEDS with multiple shards, and `sample_configs/inhospital_mortality.parquet` otherwise. However, these can be overridden using `output_filepath='/path/to/output.parquet'`.

```plaintext
shape: (2, 8)
┌────────────┬────────────┬───────┬────────────┬────────────┬────────────┬────────────┬────────────┐
│ subject_id ┆ index_time ┆ label ┆ trigger    ┆ input.end_ ┆ input.star ┆ gap.end_su ┆ target.end │
│ ---        ┆ stamp      ┆ ---   ┆ ---        ┆ summary    ┆ t_summary  ┆ mmary      ┆ _summary   │
│ i64        ┆ ---        ┆ i64   ┆ datetime[μ ┆ ---        ┆ ---        ┆ ---        ┆ ---        │
│            ┆ datetime[μ ┆       ┆ s]         ┆ struct[8]  ┆ struct[8]  ┆ struct[8]  ┆ struct[8]  │
│            ┆ s]         ┆       ┆            ┆            ┆            ┆            ┆            │
╞════════════╪════════════╪═══════╪════════════╪════════════╪════════════╪════════════╪════════════╡
│ 1          ┆ 1991-01-28 ┆ 0     ┆ 1991-01-27 ┆ {"input.en ┆ {"input.st ┆ {"gap.end" ┆ {"target.e │
│            ┆ 23:32:00   ┆       ┆ 23:32:00   ┆ d",1991-01 ┆ art",1989- ┆ ,1991-01-2 ┆ nd",1991-0 │
│            ┆            ┆       ┆            ┆ -27        ┆ 12-01      ┆ 7          ┆ 1-29       │
│            ┆            ┆       ┆            ┆ 23:32:…    ┆ 12:0…      ┆ 23:32:00…  ┆ 23:32…     │
│ 2          ┆ 1996-06-06 ┆ 1     ┆ 1996-06-05 ┆ {"input.en ┆ {"input.st ┆ {"gap.end" ┆ {"target.e │
│            ┆ 00:32:00   ┆       ┆ 00:32:00   ┆ d",1996-06 ┆ art",1996- ┆ ,1996-06-0 ┆ nd",1996-0 │
│            ┆            ┆       ┆            ┆ -05        ┆ 03-08      ┆ 5          ┆ 6-07       │
│            ┆            ┆       ┆            ┆ 00:32:…    ┆ 02:2…      ┆ 00:32:00…  ┆ 00:32…     │
└────────────┴────────────┴───────┴────────────┴────────────┴────────────┴────────────┴────────────┘
```

______________________________________________________________________

## Detailed Instructions

### Hydra

Hydra configuration files are leveraged for cohort extraction runs. All fields can be overridden by specifying their values in the command-line.

#### Data Configuration

To set a data standard:
`data.standard`: String specifying the data standard, must be 'meds' OR 'esgpt' OR 'direct'

To query from a single MEDS shard:
`data.path`: Path to the `.parquet`shard file

To query from multiple MEDS shards, you must set `data=sharded`. Additionally:

`data.root`: Root directory of MEDS dataset containing shard directories
`data.shard`: Expression specifying MEDS shards (`$(expand_shards <str>/<int>)`)

To query from an ESGPT dataset:
`data.path`: Directory of the full ESGPT dataset

To query from a direct predicates dataframe:
`data.path` Path to the `.csv` or `.parquet` file containing the predicates dataframe
`data.ts_format`: Timestamp format for predicates. Defaults to "%m/%d/%Y %H:%M"

#### Task Configuration

`cohort_dir`: Directory the your task configuration file
`cohort_name`: Name of the task configuration file

The above two fields are used for automatically loading task configurations, saving results, and logging:

`config_path`: Path to the task configuration file. Defaults to `${cohort_dir}/${cohort_name}.yaml`

`output_filepath`: Path to store the outputs. Defaults to `${cohort_dir}/${cohort_name}/${data.shard}.parquet` for MEDS with multiple shards, and `${cohort_dir}/${cohort_name}.parquet` otherwise.

#### Tab Completion

Shell completion can be enabled for the Hydra configuration fields. For Bash, please run:

```bash
eval "$(aces-cli -sc install=bash)"
```

**Note**: you may have to run this command for every terminal - please visit [Hydra's Documentation](https://hydra.cc/docs/tutorials/basic/running_your_app/tab_completion/) for more details.

### MEDS

#### Single Shard

Shards are stored as `.parquet` files in MEDS. As such, the data can be loading by providing a path pointing to the `.parquet` file directly.

```bash
aces-cli cohort_name="foo" cohort_dir="bar/" data.standard=meds data.path="baz.parquet"
```

#### Multiple Shards

A MEDS dataset can have multiple shards, each stored as a `.parquet` file containing subsets of the full dataset. We can make use of Hydra's launchers and multi-run (`-m`) capabilities to start an extraction job for each shard (`data=sharded`), either in series or in parallel (which can be useful with Slurm). To load data with multiple shards, a data root needs to be provided, along with an expression containing a comma-delimited list of files for each shard. We provide a function `expand_shards` to do this, which accepts a sequence representing `<shards_location>/<number_of_shards>`.

```bash
aces-cli cohort_name="foo" cohort_dir="bar/" data.standard=meds data=sharded data.root="baz/" "data.shard=$(expand_shards qux/#)" -m
```

### ESGPT

A ESGPT dataset will be encapsulated in a directory with two key files, `events_df.parquet` and `dynamic_measurements_df.parquet`. To load data formatting using the ESGPT standard, a directory of a valid ESGPT dataset containing these two tables is needed.

```bash
aces-cli cohort_name="foo" cohort_dir="bar/" data.standard=esgpt data.path="baz/"
```

### Direct

A direct predicates dataset could also be used instead of MEDS or ESGPT to support ***any*** dataset schema. You will need to handle the transformation of your dataset into a predicates dataframe (see [Predicates Dataframe](https://eventstreamaces.readthedocs.io/en/latest/notebooks/predicates.html)), and save it either to a `.csv` or `.parquet` file. It can then be loaded by passing this file into ACES.

```bash
aces-cli cohort_name="foo" cohort_dir="bar/" data.standard=direct data.path="baz.csv | baz.parquet"
```

### Python

You can also use the `aces.query.query()` function to extract a cohort in Python directly. Please see the [Module API Reference](https://eventstreamaces.readthedocs.io/en/latest/api/modules.html) for specifics.

```{eval-rst}
.. autofunction:: aces.query.query
```

The `cfg` parameter must be of type `config.TaskExtractorConfig`, and the `predicates_df` parameter must be of type `polars.DataFrame`.

Details about the configuration language used to define the `cfg` parameter can be found in {doc}`/configuration`.

For example, to query an in-hospital mortality task on the sample data (both the configuration file and data are provided in the repository) using the `'direct'` predicates method:

```python
>>> from aces import query, predicates, config
>>> from omegaconf import DictConfig

>>> cfg = config.TaskExtractorConfig.load(config_path="sample_configs/inhospital_mortality.yaml")

>>> data_config = DictConfig({"path": "sample_data.csv", "standard": "direct", "ts_format": "%m/%d/%Y %H:%M"})
>>> predicates_df = predicates.get_predicates_df(cfg=cfg, data_config=data_config)

>>> query.query(cfg=cfg, predicates_df=predicates_df)
```

______________________________________________________________________
