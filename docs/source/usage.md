# Usage Guide

## Quick Start

### Installation

To use ACES, first determine which data standard you'd like to use. Currently, ACES can be automatically applied to the [Medical Event Data Standard (MEDS)](https://github.com/Medical-Event-Data-Standard/meds) and [EventStreamGPT (ESGPT)](https://github.com/mmcdermott/EventStreamGPT). Please first follow instructions on their respective repositories to install and/or transform your data into one of these standards. Alternatively, ACES also supports ***any*** arbitrary dataset schema, provided you extract the necessary dataset-specific plain predicates and format it **directly** as an event-stream - details are provided [here](https://eventstreamaces.readthedocs.io/en/latest/notebooks/predicates.html).

**Note:** If you choose to use the ESGPT standard, please install ESGPT first before installing ACES. This ensures compatibility with the `polars` version required by ACES.

**To install ACES:**

```bash
pip install es-aces
```

### Task Configuration Example

**Example: `inhospital_mortality.yaml`**

Please see the [Task Configuration File Overview](https://eventstreamaces.readthedocs.io/en/latest/readme.html#task-configuration-file) for details on how to create this configuration for your own task! More examples are available [here](https://eventstreamaces.readthedocs.io/en/latest/notebooks/examples.html) and in the [GitHub repository](https://github.com/justin13601/ACES/tree/main/sample_configs).

This particular task configuration defines a cohort for the binary prediction of in-hospital mortality 48 hours after admission. Patients with 5 or more records between the start of their record and 24 hours after the admission will be included. The cohort includes both those that have been discharged (label=`0`) and those that have died (label=`1`).

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

> [!NOTE]
> Each configuration file contains
> [`predicates`](https://eventstreamaces.readthedocs.io/en/latest/readme.html#predicates), a
> [`trigger`](https://eventstreamaces.readthedocs.io/en/latest/readme.html#trigger-event), and
> [`windows`](https://eventstreamaces.readthedocs.io/en/latest/readme.html#windows). Additionally, the `label`
> field is used to extract the predicate count from the window it was defined in, which acts as the task
> label. This has been set to the `death` predicate from the `target` window in this example. The
> `index_timestamp` is used to specify the timestamp at which a prediction is made and can be set to `start`
> or `end` of a particular window. In most tasks, including this one, it can be set to `end` in the window
> containing input data (`input` in this example).

### Run the CLI

You can now run `aces-cli` in your terminal!

#### MEDS

With MEDS, ACES supports the simultaneous extraction of tasks over multiple shards with just a single command. Suppose we have a directory structure like the following:

```
ACES/
├── sample_data/
│   ├── meds_sample/
│   │   ├── held_out/
│   │   │   └── 0.parquet
│   │   ├── train/
│   │   │   ├── 0.parquet
│   │   │   └── 1.parquet
│   │   └── tuning/
│   │       └── 0.parquet
├── sample_configs/
│   └── inhospital_mortality.yaml
└── ...
```

You can run the following to execute Hydra jobs in series or parallel to extract over all MEDS shards:

```bash
aces-cli cohort_name="inhospital_mortality" cohort_dir="sample_configs/" data.standard=meds data=sharded data.root="sample_data/meds_sample/" "data.shard=$(expand_shards train/2 tuning/1)" -m
```

If you'd like to just extract a cohort from a singular shard, you can also use the following:

```bash
aces-cli cohort_name="inhospital_mortality" cohort_dir="sample_configs/" data.standard=meds data.path="sample_data/meds_sample/train/0.parquet"
```

#### ESGPT

Given the following directory structure containing an appropriate formatted ESGPT dataset with `events_df` and `dynamic_measurements_df`:

```
ACES/
├── sample_data/
│   ├── esgpt_sample/
│   │   ├── ...
│   │   ├── events_df.parquet
│   │   └── dynamic_measurements_df.parquet
├── sample_configs/
│   └── inhospital_mortality.yaml
└── ...
```

You can extract a cohort using the following:

```bash
aces-cli cohort_name="inhospital_mortality" cohort_dir="sample_configs/" data.standard=esgpt data.path="sample_data/esgpt_sample/"
```

#### Direct Predicates

To extract from a direct predicates dataframe (`.csv` | `.parquet`) from the following directory structure:

```
ACES/
├── sample_data/
│   └── sample_data.csv
├── sample_configs/
│   └── inhospital_mortality.yaml
└── ...
```

You can use the following:

```bash
aces-cli cohort_name="inhospital_mortality" cohort_dir="sample_configs/" data.standard=direct data.path="sample_data/sample_data.csv"
```

**For help using `aces-cli`:**

```bash
aces-cli --help
```

### Results

By default, results from the above examples would be saved to `sample_configs/inhospital_mortality/` containing `[train/0.parquet, train/1.parquet, test/0.parquet]` for MEDS with multiple shards, and `sample_configs/inhospital_mortality.parquet` otherwise. However, these can be overridden using `output_filepath='/path/to/output.parquet'`.

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

**To set a data standard**:

***`data.standard`***: String specifying the data standard, must be 'meds' OR 'esgpt' OR 'direct'

**To query from multiple MEDS shards**, you must set `data=sharded`. Additionally:

***`data.root`***: Root directory of MEDS dataset containing shard directories

***`data.shard`***: Expression specifying MEDS shards using [expand_shards](https://github.com/justin13601/ACES/blob/main/src/aces/expand_shards.py) (`$(expand_shards <str>/<int>)`)

**To query from a single MEDS shard**, you must set `data=single_file`. Additionally:

***`data.path`***: Path to the `.parquet` shard file

**To query from an ESGPT dataset**:

***`data.path`***: Directory of the full ESGPT dataset

**To query from a direct predicates dataframe**:

***`data.path`*** Path to the `.csv` or `.parquet` file containing the predicates dataframe

***`data.ts_format`***: Timestamp format for predicates. Defaults to "%m/%d/%Y %H:%M"

#### Task Configuration

***`cohort_dir`***: Directory of your task configuration file

***`cohort_name`***: Name of the task configuration file

The above two fields are used below for automatically loading task configurations, saving results, and logging:

***`config_path`***: Path to the task configuration file. Defaults to `${cohort_dir}/${cohort_name}.yaml`

***`output_filepath`***: Path to store the outputs. Defaults to `${cohort_dir}/${cohort_name}/${data.shard}.parquet` for MEDS with multiple shards, and `${cohort_**dir}/${cohort_name}.parquet` otherwise

***`log_dir`***: Path to store logs. Defaults to `${cohort_dir}/${cohort_name}/.logs`

Additionally, predicates may be specified in a separate predicates configuration file and loaded for overrides:

***`predicates_path`***: Path to the [separate predicates-only file](https://eventstreamaces.readthedocs.io/en/latest/usage.html#separate-predicates-only-file). Defaults to null

#### Tab Completion

Shell completion can be enabled for the Hydra configuration fields. For Bash, please run:

```bash
eval "$(aces-cli -sc install=bash)"
```

> [!NOTE]
> you may have to run this command for every terminal - please visit [Hydra's
> Documentation](https://hydra.cc/docs/tutorials/basic/running_your_app/tab_completion/) for more details.

### MEDS

#### Multiple Shards

A MEDS dataset can have multiple shards, each stored as a `.parquet` file containing subsets of the full dataset. We can make use of Hydra's launchers and multi-run (`-m`) capabilities to start an extraction job for each shard (`data=sharded`), either in series or in parallel (e.g., using `joblib`, or `submitit` for Slurm). To load data with multiple shards, a data root needs to be provided, along with an expression containing a comma-delimited list of files for each shard. We provide a function `expand_shards` to do this, which accepts a sequence representing `<shards_location>/<number_of_shards>`. It also accepts a file directory, where all `.parquet` files in its directory and subdirectories will be included.

```bash
aces-cli cohort_name="foo" cohort_dir="bar/" data.standard=meds data=sharded data.root="baz/" "data.shard=$(expand_shards qux/#)" -m
```

#### Single Shard

Shards are stored as `.parquet` files in MEDS. As such, the data can be loading by providing a path pointing to the `.parquet` file directly, and specifying `data=single_file`.

```bash
aces-cli cohort_name="foo" cohort_dir="bar/" data.standard=meds data.path="baz.parquet"
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

The `cfg` parameter must be of type {py:class}`aces.config.TaskExtractorConfig`, and the `predicates_df` parameter must be of type `polars.DataFrame`.

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

### Separate Predicates-Only File

For more complex tasks involving a large number of predicates, a separate predicates-only "database" file can
be created and passed into `TaskExtractorConfig.load()`. Only referenced predicates will have a predicate
column computed and evaluated, so one could create a dataset-specific deposit file with many predicates and
reference as needed to ensure the cleanliness of the dataset-agnostic task criteria file.

```python
>>> cfg = config.TaskExtractorConfig.load(config_path="criteria.yaml", predicates_path="predicates.yaml")
```

If the same predicates are defined in both the task configuration file and the predicates-only file, the
predicates-only definition takes precedent and will be used to override previous definitions. As such, one may
create a predicates-only "database" file for a particular dataset, and override accordingly for various tasks.

______________________________________________________________________
