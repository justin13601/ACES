# Event Stream Automatic Cohort Extraction System (ACES)

[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![codecov](https://codecov.io/gh/justin13601/ACES/graph/badge.svg?token=6EA84VFXOV)](https://codecov.io/gh/justin13601/ACES)
[![tests](https://github.com/justin13601/ACES/actions/workflows/tests.yml/badge.svg)](https://github.com/justin13601/ACES/actions/workflows/test.yml)
[![code-quality](https://github.com/justin13601/ACES/actions/workflows/code-quality-master.yaml/badge.svg)](https://github.com/justin13601/ACES/actions/workflows/code-quality-master.yaml)
[![Documentation Status](https://readthedocs.org/projects/eventstreamaces/badge/?version=latest)](https://eventstreamaces.readthedocs.io/en/latest/?badge=latest)
[![contributors](https://img.shields.io/github/contributors/justin13601/ACES.svg)](https://github.com/justin13601/ACES/graphs/contributors)
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/justin13601/ACES/pulls)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/justin13601/ACES#license)

## Background

EventStreamGPT (ESGPT) is a library that streamlines the development of generative, pre-trained transformers (i.e., foundation models) over event stream datasets, such as Electronic Health Records (EHR). ESGPT is designed to extract, preprocess, and manage these datasets efficiently, providing a Huggingface-compatible modeling API and introducing critical capabilities for representing complex intra-event causal dependencies and measuring zero-shot performance. For more detailed information, please refer to the ESGPT GitHub repository: [ESGPT GitHub Repo](https://github.com/esgpt).

A feature of ESGPT is the ability to query EHR datasets for valid subjects, guided by various constraints and requirements defined in a YAML configuration file. This streamlines the process of extracting task-specific cohorts from large time-series datasets, offering a powerful and user-friendly solution to researchers and developers. The use of a human-readable YAML configuration file also eliminates the need for users to be proficient in complex dataframe querying, making the querying process accessible to a broader audience.

There are diverse applications in healthcare and beyond. For instance, researchers can effortlessly define subsets of EHR datasets for training of foundational models. Retrospective analyses can also become more accessible to clinicians as it enables the extraction of tailored cohorts for studying specific medical conditions or population demographics.

This README provides an overview of this feature, including a description of the YAML configuration file's fields (see `sample_config.yaml`), an outline of the algorithm, and instructions for use.

Please also refer to the [documentation](https://eventstreamaces.readthedocs.io/en/latest/) for more information.

## Dependencies

- polars == 0.20.18
- bigtree == 0.17.0
- ruamel.yaml == 0.18.6,
- pytimeparse == 1.1.8,
- loguru == 0.7.2,
- hydra-core == 1.3.2,
- networkx == 3.3,

## Installation

1. If using the ESGPT data standard, install [EventStreamGPT](https://github.com/mmcdermott/EventStreamGPT):

Clone EventStreamGPT:

```bash
git clone https://github.com/mmcdermott/EventStreamGPT.git
```

Install package with dependencies from the root directory of the cloned repo:

```bash
pip install -e .
```

Note: please install ESGPT first and then install ACES. This will update the polars version to the one used by ACES. This may not necessarily create an environment suitable for ESGPT modelling; however, the environment will be able to load your data in ESGPT format to enable ACES querying with the ESGPT data standard.

2. Install ACES:

```bash
pip install es-aces
```

## Instructions for Use

1. **Prepare a Task Configuration File**: Define your predicates and task windows according to your research needs. Please see below or the [documentation](https://eventstreamaces.readthedocs.io/en/latest/) for details regarding the configuration language.
2. **Prepare Dataset into Supported Standards**: Process your dataset according to instructions for the [MEDS](https://github.com/Medical-Event-Data-Standard/meds) or [ESGPT](https://github.com/mmcdermott/EventStreamGPT) standard. You could also create a `.csv` in the same format as `sample_data/sample.csv`.
3. **Prepare a Hydra Configuration File**: Define `config_path`, `data_path`, and `output_path` to specify the location of your task configuration file, data file/directory, and results output location, respectively.

Command line:

```bash
aces-cli --config-dir='/path/to/hydra/config/' --config-name='config.yaml'
```

Code:

```python
from aces import config, predicates, query

# create task configuration object
cfg = config.TaskExtractorConfig.load(config_path="/path/to/task/config/task.yaml")

# one of the following
predicates_df = predicates.generate_predicates_df(cfg, "/path/to/data.parquet", "meds")
predicates_df = predicates.generate_predicates_df(
    cfg, "/path/to/esgpt/folder/", "esgpt"
)
predicates_df = predicates.generate_predicates_df(cfg, "/path/to/data.csv", "csv")

# execute query and display results
df_result = query.query(cfg, predicates_df)
display(df_result)
```

**Results**: The output will be a dataframe of subjects who satisfy the conditions defined in your task configuration file. Timestamps for an edge of each window specified in the YAML, as well as predicate counts for each window, are also provided.

```
aces-cli --config-path=/n/data1/hms/dbmi/zaklab/mmd/ESACES_tests/outputs/inovalon_tests/ESGPT/readmission_test/ --config-name="config"
2024-05-24 16:08:04.071 | INFO     | aces.__main__:main:42 - Loading config...
2024-05-24 16:08:04.095 | INFO     | aces.config:load:775 - Parsing predicates...
2024-05-24 16:08:04.095 | INFO     | aces.config:load:781 - Parsing trigger event...
2024-05-24 16:08:04.095 | INFO     | aces.config:load:784 - Parsing windows...
2024-05-24 16:08:04.156 | INFO     | aces.__main__:main:47 - Loading data...
2024-05-24 16:08:04.161 | INFO     | aces.__main__:main:55 - Directory provided, checking directory...
Loading events from /n/data1/hms/dbmi/zaklab/inovalon_mbm47/processed/12-19-23_InovalonSample1M/events_df.parquet...
Loading dynamic_measurements from /n/data1/hms/dbmi/zaklab/inovalon_mbm47/processed/12-19-23_InovalonSample1M/dynamic_measurements_df.parquet...
2024-05-24 16:08:28.855 | INFO     | aces.predicates:generate_plain_predicates_from_esgpt:151 - Generating plain predicate columns...
2024-05-24 16:08:30.337 | INFO     | aces.predicates:generate_plain_predicates_from_esgpt:160 - Added predicate column 'admission'.
2024-05-24 16:08:31.608 | INFO     | aces.predicates:generate_plain_predicates_from_esgpt:160 - Added predicate column 'discharge'.
2024-05-24 16:09:02.738 | INFO     | aces.predicates:generate_plain_predicates_from_esgpt:177 - Cleaning up predicates DataFrame...
2024-05-24 16:09:02.740 | INFO     | aces.predicates:generate_predicates_df:277 - Loaded plain predicates. Generating derived predicate columns...
2024-05-24 16:09:02.740 | INFO     | aces.predicates:generate_predicates_df:284 - Generating '_ANY_EVENT' predicate column...
2024-05-24 16:09:02.747 | INFO     | aces.predicates:generate_predicates_df:286 - Added predicate column '_ANY_EVENT'.
2024-05-24 16:09:05.074 | INFO     | aces.utils:log_tree:56 - trigger
┗━━ input.end
    ┗━━ target.end

2024-05-24 16:09:05.075 | INFO     | aces.query:query:31 - Beginning query...
2024-05-24 16:09:05.089 | INFO     | aces.constraints:check_constraints:95 - Excluding 45,970,524 rows as they failed to satisfy 1 <= admission <= None.
2024-05-24 16:09:05.153 | INFO     | aces.extract_subtree:extract_subtree:249 - Summarizing subtree rooted at 'input.end'...
2024-05-24 16:09:22.494 | INFO     | aces.extract_subtree:extract_subtree:249 - Summarizing subtree rooted at 'target.end'...
2024-05-24 16:09:29.664 | INFO     | aces.query:query:37 - Done. 293608 rows returned.
```

## Task Configuration File

The task configuration file allows users to define specific predicates and windows to query your dataset. Below is a description of each field:

### Predicates

Predicates describe the event at a timestamp. Predicate columns begin with `is_` and are initialized as binary counts for each row of your ESD. Here is an example .csv file with predicate columns generated.

```
subject_id,timestamp,event_type,dx,lab_test,lab_value,is_death,is_admission,is_discharge,is_covid,is_death_or_discharge,is_any
1,12/1/1989 12:03,ADMISSION,,,,0,1,0,0,0,1
1,12/1/1989 13:14,LAB,,SpO2,99,0,0,0,0,0,1
1,12/1/1989 15:17,LAB,,SpO2,98,0,0,0,0,0,1
1,12/1/1989 16:17,LAB,,SpO2,99,0,0,0,0,0,1
1,12/1/1989 20:17,LAB,,SpO2,98,0,0,0,0,0,1
1,12/2/1989 3:00,LAB,,SpO2,99,0,0,0,0,0,1
1,12/2/1989 9:00,DIAGNOSIS,FLU,,,0,0,0,0,0,1
1,12/2/1989 15:00,DISCHARGE,,,,0,0,1,0,1,1
```

There are two types of predicates that can be defined in the configuration file. They can represent explicit ESD events and be defined by (`column`, `value`) pairs:

- `column`: Specifies the column in the dataset to apply the predicate. Must be a string matching an ESD column name.
- `value`: The value to match in the specified `column`.

OR, they can combine existing predicates using `ANY` or `ALL` keywords in the (`type`, `predicates`) pairs:

- `type`: Must be `ANY` or `ALL`.
- `predicates`: Must be list of existing predicate names defined using the above configuration.

### Windows

Windows can be of two types. It can be a temporally-bound window defined by a `duration` and one of `start`/`end`. It can also be an event-bound window defined by a `start` and an `end`.

- `start`: Must be a string matching a predicate name or containing a window name to express window relationship.
- `duration`: Must be a positive or negative time period expressed as a string (ie. 2 days, -365 days, 12 hours, 30 minutes, 60 seconds).
- `offset`: Not yet available.
- `end`: Must be a string matching a predicate name or containing a window name to express window relationship.
- `excludes`: Listed `predicate` fields matching a predicate name. Used to exclude a predicate in the window.
- `includes`: Listed `predicate` fields matching a predicate name. Used to include a predicate in the window, with `min` and `max` specifying the constraints for occurrences (`None` is set where `min`/`max` is left blank).
- `st_inclusive`, `end_inclusive`: Boolean flags to indicate if events at the start and end of the window timestamps are included in the defined window.
- `label`: Must be a string matching a predicate name.

Each window uses these fields to define specific time frames and criteria within the dataset.

A sample YAML configuration file is provided in `sample_config.yaml`.

## Acknowledgements

**Matthew McDermott**, PhD | *Harvard Medical School*
**Jack Gallifant**, MD | *Massachusetts Institute of Technology*
**Tom Pollard**, PhD | *Massachusetts Institute of Technology*
**Alistair Johnson**, DPhil

For any questions, enhancements, or issues, please file a GitHub issue. For inquiries regarding MEDS or EventStreamGPT, please refer to their respective repositories. Contributions are welcome via pull requests.
