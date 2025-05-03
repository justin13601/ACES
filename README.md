<p align="center">
  <a href="https://eventstreamaces.readthedocs.io/en/latest/index.html"><img alt="ACES" src="https://raw.githubusercontent.com/justin13601/ACES/bbde3d2047d30f2203cc09a288a8e3565a0d7d62/docs/source/assets/aces_logo_text.svg" width=35%></a>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/release/python-3100/"><img alt="Python" src="https://img.shields.io/badge/-Python_3.10+-blue?logo=python&logoColor=white"></a>
  <a href="https://pypi.org/project/es-aces/"><img alt="PyPI" src="https://img.shields.io/pypi/v/es-aces"></a>
  <a href="https://hydra.cc/"><img alt="Hydra" src="https://img.shields.io/badge/Config-Hydra_1.3-89b8cd"></a>
  <a href="https://codecov.io/gh/justin13601/ACES"><img alt="Codecov" src="https://codecov.io/gh/justin13601/ACES/graph/badge.svg?token=6EA84VFXOV"></a>
  <a href="https://github.com/justin13601/ACES/actions/workflows/tests.yaml"><img alt="Tests" src="https://github.com/justin13601/ACES/actions/workflows/tests.yaml/badge.svg"></a>
  <a href="https://github.com/justin13601/ACES/actions/workflows/code-quality-main.yaml"><img alt="Code Quality" src="https://github.com/justin13601/ACES/actions/workflows/code-quality-main.yaml/badge.svg"></a>
  <a href="https://eventstreamaces.readthedocs.io/en/latest/?badge=latest"><img alt="Documentation" src="https://readthedocs.org/projects/eventstreamaces/badge/?version=latest"/></a>
  <a href="https://github.com/justin13601/ACES/graphs/contributors"><img alt="Contributors" src="https://img.shields.io/github/contributors/justin13601/ACES.svg"></a>
  <a href="https://github.com/justin13601/ACES/pulls"><img alt="Pull Requests" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg"></a>
  <a href="https://github.com/justin13601/ACES#license"><img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray"></a>
</p>

# Automatic Cohort Extraction System for Event-Streams

**Updates**

- **[2025-01-22]** ACES accepted to ICLR'25!
- **[2024-12-10]** Latest `polars` version (`1.17.1`) is now supported.
- **[2024-10-28]** Nested derived predicates and derived predicates between static variables and plain predicates can now be defined.
- **[2024-09-01]** Predicates can now be defined in a configuration file separate to task criteria files.
- **[2024-08-29]** Latest `MEDS` version (`0.3.3`) is now supported.
- **[2024-08-10]** Expanded predicates configuration language to support regular expressions, multi-column constraints, and multi-value constraints.
- **[2024-07-30]** Added ability to place constraints on static variables, such as patient demographics.
- **[2024-06-28]** Paper available at [arXiv:2406.19653](https://arxiv.org/abs/2406.19653).

Automatic Cohort Extraction System (ACES) is a library that streamlines the extraction of task-specific cohorts from time series datasets formatted as event-streams, such as Electronic Health Records (EHR). ACES is designed to query these EHR datasets for valid subjects, guided by various constraints and requirements defined in a YAML task configuration file. This offers a powerful and user-friendly solution to researchers and developers. The use of a human-readable YAML configuration file also eliminates the need for users to be proficient in complex dataframe querying, making the extraction process accessible to a broader audience.

There are diverse applications in healthcare and beyond. For instance, researchers can effortlessly define subsets of EHR datasets for training of foundation models. Retrospective analyses can also become more accessible to clinicians as it enables the extraction of tailored cohorts for studying specific medical conditions or population demographics. Finally, ACES can help realize a new era of benchmarking over tasks instead of data - please check out [MEDS-DEV](https://github.com/mmcdermott/MEDS-DEV/tree/main)!

Currently, two data standards are directly supported: the [Medical Event Data Standard (MEDS)](https://github.com/Medical-Event-Data-Standard/meds) standard and the [EventStreamGPT (ESGPT)](https://github.com/mmcdermott/EventStreamGPT) standard. You must format your data in one of these two formats by following instructions in their respective repositories. ACES also supports ***any*** arbitrary dataset schema, provided you extract the necessary dataset-specific plain predicates and format it as an event-stream. More information about this is available below and [here](https://eventstreamaces.readthedocs.io/en/latest/notebooks/predicates.html).

This README provides a brief overview of this tool, instructions for use, and a description of the fields in the task configuration file (see representative configs in `sample_configs/`). Please refer to the [ACES Documentation](https://eventstreamaces.readthedocs.io/en/latest/) for more detailed information.

## Installation

### For MEDS v0.3.3

```bash
pip install es-aces
```

### For ESGPT

1. Install [EventStreamGPT (ESGPT)](https://github.com/mmcdermott/EventStreamGPT):

Clone EventStreamGPT:

```bash
git clone https://github.com/mmcdermott/EventStreamGPT.git
```

Install with dependencies from the root directory of the cloned repo:

```bash
pip install -e .
```

> [!NOTE]
> To avoid potential dependency conflicts, please install ESGPT first before installing ACES. This ensures
> compatibility with the `polars` version required by ACES.

## Instructions for Use

1. **Prepare a Task Configuration File**: Define your predicates and task windows according to your research needs. Please see below or [here](https://eventstreamaces.readthedocs.io/en/latest/configuration.html) for details regarding the configuration language.
2. **Prepare Dataset & Predicates DataFrame**: Process your dataset according to instructions for the [MEDS](https://github.com/Medical-Event-Data-Standard/meds) or [ESGPT](https://github.com/mmcdermott/EventStreamGPT) standard so you can leverage ACES to automatically create the predicates dataframe. Alternatively, you can also create your own predicates dataframe directly (more information below and [here](https://eventstreamaces.readthedocs.io/en/latest/notebooks/predicates.html)).
3. **Execute Query**: A query may be executed using either the command-line interface or by importing the package in Python:

### Command-Line Interface:

```bash
aces-cli data.path='/path/to/data/directory/or/file' data.standard='<meds|esgpt|direct>' cohort_dir='/directory/to/task/config/' cohort_name='<task_config_name>'
```

For help using `aces-cli`:

```bash
aces-cli --help
```

### Python Code:

```python
from aces import config, predicates, query
from omegaconf import DictConfig

# create task configuration object
cfg = config.TaskExtractorConfig.load(config_path="/path/to/task/config.yaml")

# get predicates dataframe
data_config = DictConfig(
    {
        "path": "/path/to/data/directory/or/file",
        "standard": "<meds|esgpt|direct>",
        "ts_format": "%m/%d/%Y %H:%M",
    }
)
predicates_df = predicates.get_predicates_df(cfg=cfg, data_config=data_config)

# execute query and get results
df_result = query.query(cfg=cfg, predicates_df=predicates_df)
```

4. **Results**: The output will be a dataframe of subjects who satisfy the conditions defined in your task configuration file. Timestamps for the start/end boundaries of each window specified in the task configuration, as well as predicate counts for each window, are also provided. Below are sample logs for the successful extraction of an in-hospital mortality cohort:

```log
aces-cli cohort_name="inhospital_mortality" cohort_dir="sample_configs" data.standard="meds" data.path="MEDS_DATA"
2024-09-24 02:06:57.362 | INFO     | aces.__main__:main:153 - Loading config from 'sample_configs/inhospital_mortality.yaml'
2024-09-24 02:06:57.369 | INFO     | aces.config:load:1258 - Parsing windows...
2024-09-24 02:06:57.369 | INFO     | aces.config:load:1267 - Parsing trigger event...
2024-09-24 02:06:57.369 | INFO     | aces.config:load:1282 - Parsing predicates...
2024-09-24 02:06:57.380 | INFO     | aces.__main__:main:156 - Attempting to get predicates dataframe given:
standard: meds
ts_format: '%m/%d/%Y %H:%M'
path: MEDS_DATA/
_prefix: ''

2024-09-24 02:07:58.176 | INFO     | aces.predicates:generate_plain_predicates_from_meds:268 - Loading MEDS data...
2024-09-24 02:07:01.405 | INFO     | aces.predicates:generate_plain_predicates_from_esgpt:272 - Generating plain predicate columns...
2024-09-24 02:07:01.579 | INFO     | aces.predicates:generate_plain_predicates_from_esgpt:276 - Added predicate column 'admission'.
2024-09-24 02:07:01.770 | INFO     | aces.predicates:generate_plain_predicates_from_esgpt:276 - Added predicate column 'discharge'.
2024-09-24 02:07:01.925 | INFO     | aces.predicates:generate_plain_predicates_from_esgpt:276 - Added predicate column 'death'.
2024-09-24 02:07:07.155 | INFO     | aces.predicates:generate_plain_predicates_from_esgpt:279 - Cleaning up predicates dataframe...
2024-09-24 02:07:07.156 | INFO     | aces.predicates:get_predicates_df:642 - Loaded plain predicates. Generating derived predicate columns...
2024-09-24 02:07:07.167 | INFO     | aces.predicates:get_predicates_df:645 - Added predicate column 'discharge_or_death'.
2024-09-24 02:07:07.772 | INFO     | aces.predicates:get_predicates_df:654 - Generating special predicate columns...
2024-09-24 02:07:07.841 | INFO     | aces.predicates:get_predicates_df:681 - Added predicate column '_ANY_EVENT'.
2024-09-24 02:07:07.841 | INFO     | aces.query:query:76 - Checking if '(subject_id, timestamp)' columns are unique...
2024-09-24 02:07:08.221 | INFO     | aces.utils:log_tree:57 -

trigger
┣━━ input.end
┃   ┗━━ input.start
┗━━ gap.end
    ┗━━ target.end

2024-09-24 02:07:08.221 | INFO     | aces.query:query:85 - Beginning query...
2024-09-24 02:07:08.221 | INFO     | aces.query:query:89 - Static variable criteria specified, filtering patient demographics...
2024-09-24 02:07:08.221 | INFO     | aces.query:query:99 - Identifying possible trigger nodes based on the specified trigger event...
2024-09-24 02:07:08.233 | INFO     | aces.constraints:check_constraints:110 - Excluding 14,623,763 rows as they failed to satisfy '1 <= admission <= None'.
2024-09-24 02:07:08.249 | INFO     | aces.extract_subtree:extract_subtree:252 - Summarizing subtree rooted at 'input.end'...
2024-09-24 02:07:13.259 | INFO     | aces.extract_subtree:extract_subtree:252 - Summarizing subtree rooted at 'input.start'...
2024-09-24 02:07:26.011 | INFO     | aces.constraints:check_constraints:176 - Excluding 12,212 rows as they failed to satisfy '5 <= _ANY_EVENT <= None'.
2024-09-24 02:07:26.052 | INFO     | aces.extract_subtree:extract_subtree:252 - Summarizing subtree rooted at 'gap.end'...
2024-09-24 02:07:30.223 | INFO     | aces.constraints:check_constraints:176 - Excluding 631 rows as they failed to satisfy 'None <= admission <= 0'.
2024-09-24 02:07:30.224 | INFO     | aces.constraints:check_constraints:176 - Excluding 18,165 rows as they failed to satisfy 'None <= discharge <= 0'.
2024-09-24 02:07:30.224 | INFO     | aces.constraints:check_constraints:176 - Excluding 221 rows as they failed to satisfy 'None <= death <= 0'.
2024-09-24 02:07:30.226 | INFO     | aces.extract_subtree:extract_subtree:252 - Summarizing subtree rooted at 'target.end'...
2024-09-24 02:07:41.512 | INFO     | aces.query:query:113 - Done. 44,318 valid rows returned corresponding to 11,606 subjects.
2024-09-24 02:07:41.513 | INFO     | aces.query:query:129 - Extracting label 'death' from window 'target'...
2024-09-24 02:07:41.514 | INFO     | aces.query:query:142 - Setting index timestamp as 'end' of window 'input'...
2024-09-24 02:07:41.606 | INFO     | aces.__main__:main:188 - Completed in 0:00:44.243514. Results saved to 'sample_configs/inhospital_mortality.parquet'.
```

## Task Configuration File

The task configuration file allows users to define specific predicates and windows to query your dataset. Below is a sample generic configuration file in its most basic form:

```yaml
predicates:
  predicate_1:
    code: ???
  ...

trigger: ???

windows:
  window_1:
    start: ???
    end: ???
    start_inclusive: ???
    end_inclusive: ???
    has:
      predicate_1: (???, ???)

    label: ???
    index_timestamp: ???
  ...
```

Sample task configuration files for 6 common tasks are provided in `sample_configs/`. All task configurations can be directly extracted using `'direct'` mode on `sample_data/sample_data.csv` as this predicates dataframe was designed specifically to capture concepts needed for all tasks. However, only `inhospital_mortality.yaml` and `imminent-mortality.yaml` would be able to be extracted on `sample_data/esgpt_sample` and `sample_data/meds_sample` due to a lack of required concepts in the datasets (predicates are defined as per the MEDS sample data by default; modifications will be needed for ESGPT).

### Predicates

Predicates describe the event at a timestamp. Predicate columns are created to contain predicate counts for each row of your dataset. If the MEDS or ESGPT data standard is used, ACES automatically computes the predicates dataframe needed for the query from the `predicates` fields in your task configuration file. However, you may also choose to construct your own predicates dataframe should you not wish to use the MEDS or ESGPT data standard.

Example predicates dataframe `.csv`:

```
subject_id,timestamp,death,admission,discharge,covid,death_or_discharge,_ANY_EVENT
1,12/1/1989 12:03,0,1,0,0,0,1
1,12/1/1989 13:14,0,0,0,0,0,1
1,12/1/1989 15:17,0,0,0,0,0,1
1,12/1/1989 16:17,0,0,0,0,0,1
1,12/1/1989 20:17,0,0,0,0,0,1
1,12/2/1989 3:00,0,0,0,0,0,1
1,12/2/1989 9:00,0,0,0,0,0,1
1,12/2/1989 15:00,0,0,1,0,1,1
```

There are two types of predicates that can be defined in the configuration file, "plain" predicates, and "derived" predicates.

#### Plain Predicates

"Plain" predicates represent explicit values (either `str` or `int`) in your dataset at a particular timestamp and has 1 required `code` field (for string categorical variables) and 4 optional fields (for integer or float continuous variables). For instance, the following defines a predicate representing normal SpO2 levels (a range of 90-100 corresponding to rows where the `lab` column is `O2 saturation pulseoxymetry (%)`):

```yaml
normal_spo2:
  code: lab//O2 saturation pulseoxymetry (%)     # required <str>//<str>
  value_min: 90                                  # optional <float/int>
  value_max: 100                                 # optional <float/int>
  value_min_inclusive: true                      # optional <bool>
  value_max_inclusive: true                      # optional <bool>
  other_cols: {}                                 # optional <dict>
```

Fields for a "plain" predicate:

- `code` (required): Must be one of the following:
  - a string matching values in a column named `code` (for `MEDS` only).
  - a string with a `//` sequence separating the column name and the matching column value (for `ESGPT` only).
  - a list of strings as above in the form of `{any: \[???, ???, ...\]}` (or the corresponding expanded indented `YAML` format), which will match any of the listed codes.
  - a regex in the form of `{regex: "???"}` (or the corresponding expanded indented `YAML` format), which will match any code that matches that regular expression.
- `value_min` (optional): Must be float or integer specifying the minimum value of the predicate, if the variable is presented as numerical values.
- `value_max` (optional): Must be float or integer specifying the maximum value of the predicate, if the variable is presented as numerical values.
- `value_min_inclusive` (optional): Must be a boolean specifying whether `value_min` is inclusive or not.
- `value_max_inclusive` (optional): Must be a boolean specifying whether `value_max` is inclusive or not.
- `other_cols` (optional): Must be a 1-to-1 dictionary of column name and column value, which places additional constraints on further columns.

> [!NOTE]
> For memory optimization, we strongly recommend using either the List of Values or Regular Expression formats
> whenever possible, especially when needing to match multiple values. Defining each code as an individual
> string will increase memory usage significantly, as each code generates a separate predicate column. Using a
> list or regex consolidates multiple matching codes under a single column, reducing the overall memory
> footprint.

#### Derived Predicates

"Derived" predicates combine existing "plain" predicates using `and` / `or` keywords and have exactly 1 required `expr` field: For instance, the following defines a predicate representing either death or discharge (by combining "plain" predicates of `death` and `discharge`):

```yaml
# plain predicates
discharge:
  code: event_type//DISCHARGE
death:
  code: event_type//DEATH

# derived predicates
discharge_or_death:
  expr: or(discharge, death)
```

Field for a "derived" predicate:

- `expr`: Must be a string with the 'and()' / 'or()' key sequences, with "plain" predicates as its constituents.

A special predicate `_ANY_EVENT` is always defined, which simply represents any event, as the name suggests. This predicate can be used like any other predicate manually defined (ie., setting a constraint on its occurrence or using it as a trigger - more information below!).

#### Special Predicates

There are also a few special predicates that you can use. These *do not* need to be defined explicitly in the configuration file, and can be directly used:

`_ANY_EVENT`: specifies any event in the data (ie., effectively set to `1` for every single row in your predicates dataframe)

`_RECORD_START`: specifies the beginning of a patient's record (ie., effectively set to `1` in the first chronological row for every `subject_id`)

`_RECORD_END`: specifies the end of a patient's record (ie., effectively set to `1` in the last chronological row for every `subject_id`)

### Trigger Event

The trigger event is a simple field with a value of a predicate name. For each trigger event, a prediction by a model can be made. For instance, in the following example, the trigger event is an admission. Therefore, in your task, a prediction by a model can be made for each valid admission (ie., samples remaining after extraction according to other task specifications are considered valid). You can also simply filter to a cohort of one event (ie., just a trigger event) should you not have any further criteria in your task.

```yaml
predicates:
  admission:
    code: event_type//ADMISSION

trigger: admission                    # trigger event <predicate>
```

### Windows

Windows can be of two types, a temporally-bounded window or an event-bounded window. Below is a sample temporally-bounded window configuration:

```
trigger: admission

input:
  start: NULL
  end: trigger + 24h
  start_inclusive: True
  end_inclusive: True
  has:
    _ANY_EVENT: (5, None)
```

In this example, the window `input` begins at `NULL` (ie., the first event or the start of the time series record), and ends at 24 hours after the `trigger` event, which is specified to be a hospital admission. The window is inclusive on both ends (ie., both the first event and the event at 24 hours after the admission, if any, is included in this window). Finally, a constraint of 5 events of any kind is placed so any valid window would include sufficient data.

Two fields (`start` and `end`) are required to define the size of a window. Both fields must be a string referencing a predicate name, or a string referencing the `start` or `end` field of another window. In addition, it may express a temporal relationship by including a positive or negative time period expressed as a string (ie., `+ 2 days`, `- 365 days`, `+ 12h`, `- 30 minutes`, `+ 60s`). It may also express an event relationship by including a sequence with a directional arrow and a predicate name (ie., `-> predicate_1` indicating the period until the next occurrence of the predicate, or `<- predicate_1` indicating the period following the previous occurrence of the predicate). Finally, it may also contain `NULL`, indicating the first/last event for the `start`/`end` field, respectively.

`start_inclusive` and `end_inclusive` are required booleans specifying whether the events, if present, at the `start` and `end` points of the window are included in the window.

The `has` field specifies constraints relating to predicates within the window. For each predicate defined previously, a constraint for occurrences can be set using a string in the format of `(<min>, <max>)`. Unbounded conditions can be specified by using `None` or leaving it empty (ie., `(5, None)`, `(8,)`, `(None, 32)`, `(,10)`).

`label` is an optional field and can only exist in ONE window in the task configuration file if defined (an error is thrown otherwise). It must be a string matching a defined predicate name, and is used to extract the label for the task.

`index_timestamp` is an optional field and can only exist in ONE window in the task configuration file if defined (an error is thrown otherwise). It must be either `start` or `end`, and is used to create an index column used to easily manipulate the results output. Usually, one would set it to be the time at which the prediction would be made (ie., set to `end` in your window containing input data). Please ensure that you are validating your interpretation of `index_timestamp` for your task. For instance, if `index_timestamp` is set to the `end` of a particular window, the timestamp would be the event at the window boundary. However, in some cases, your task may want to exclude this boundary event, so ensure you are correctly interpreting the timestamp during extraction.

## FAQs

### Static Data

In MEDS, static variables are simply stored in rows with `null` timestamps. In ESGPT, static variables are stored in a separate `subjects_df` table. In either case, it is feasible to express static variables as a predicate and apply the associated criteria normally using the `patient_demographics` heading of a configuration file. Please see [here](https://eventstreamaces.readthedocs.io/en/latest/notebooks/examples.html) and [here](https://eventstreamaces.readthedocs.io/en/latest/notebooks/predicates.html) for examples and details.

### Complementary Tools

ACES is an integral part of the MEDS ecosystem. To fully leverage its capabilities, you can utilize it alongside other complementary MEDS tools, such as:

- [MEDS-ETL](https://github.com/Medical-Event-Data-Standard/meds_etl), which can be used to transform various data schemas, including some common data models, into the MEDS format.
- [MEDS-TAB](https://github.com/Medical-Event-Data-Standard/meds_etl), which can be used to generate automated tabular baseline methods (ie., XGBoost over ACES-defined tasks).
- [MEDS-Polars](https://github.com/Medical-Event-Data-Standard/meds_etl), which contains polars-based ETL scripts.

### Alternative Tools

There are existing alternatives for cohort extraction that focus on specific common data models, such as [i2b2 PIC-SURE](https://pic-sure.org/) and [OHDSI ATLAS](https://atlas.ohdsi.org/).

ACES serves as a middle ground between PIC-SURE and ATLAS. While it may offer less capability than PIC-SURE, it compensates with greater ease of use and improved communication value. Compared to ATLAS, ACES provides greater capability, though with slightly lower ease of use, yet it still maintains a higher communication value.

Finally, ACES is not tied to a particular common data model. Built on a flexible event-stream format, ACES is a no-code solution with a descriptive input format, permitting easy and wide iteration over task definitions. It can be applied to a variety of schemas, making it a versatile tool suitable for diverse research needs.

## Future Roadmap

### Usability

- Extract indexing information for easier setup of downstream tasks ([#37](https://github.com/justin13601/ACES/issues/37))

### Coverage

- Directly support nested configuration files ([#43](https://github.com/justin13601/ACES/issues/43))
- Support timestamp binning for use in predicates or as qualifiers ([#44](https://github.com/justin13601/ACES/issues/44))
- Support additional label types ([#45](https://github.com/justin13601/ACES/issues/45))
- Allow chaining of multiple task configurations ([#49](https://github.com/justin13601/ACES/issues/49))
- Additional predicates expansions ([#66](https://github.com/justin13601/ACES/issues/66))

### Generalizability

- Promote generalizability across other common data models ([#50](https://github.com/justin13601/ACES/issues/50))

### Causal Usage

- Directly support case-control matching ([#51](https://github.com/justin13601/ACES/issues/51))

### Additional Tasks

- Support for additional task types and outputs ([#53](https://github.com/justin13601/ACES/issues/53))
- Directly support tasks with multiple endpoints ([#54](https://github.com/justin13601/ACES/issues/54))

### Natural Language Interface

- LLM integration for extraction ([#55](https://github.com/justin13601/ACES/issues/55))

## Video Demonstration

<div align="left">
  <a href="https://www.youtube.com/watch?v=i_hCaHDydqA"><img src="https://img.youtube.com/vi/i_hCaHDydqA/0.jpg" alt="ACES MEDS Demo"></a>
</div>

## Acknowledgements

**Matthew McDermott**, PhD | *Harvard Medical School*

**Alistair Johnson**, DPhil | *Independent*

**Jack Gallifant**, MD | *Massachusetts Institute of Technology*

**Tom Pollard**, PhD | *Massachusetts Institute of Technology*

**Curtis Langlotz**, MD, PhD | *Stanford University*

**David Eyre**, BM BCh, DPhil | *University of Oxford*

For any questions, enhancements, or issues, please file a GitHub issue. For inquiries regarding MEDS or ESGPT, please refer to their respective repositories. Contributions are welcome via pull requests.
