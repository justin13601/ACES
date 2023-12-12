# ESGPT Task Querying with YAML Configuration

Made with Python 3.10 <Important because of switch statements>

## Introduction

Event Stream GPT (ESGPT) is a library that streamlines the development of generative, pre-trained transformers (i.e., foundation models) over event stream datasets, such as Electronic Health Records (EHR). ESGPT is designed to extract, preprocess, and manage these datasets efficiently, providing a Huggingface-compatible modeling API and introducing critical capabilities for representing complex intra-event causal dependencies and measuring zero-shot performance. For more detailed information, please refer to the ESGPT GitHub repository: [ESGPT GitHub Repo](https://github.com/esgpt).

A feature of ESGPT is its ability to query EHR datasets for valid subjects, guided by various constraints and requirements defined in a YAML configuration file. This README provides an overview of this feature, including a description of the YAML configuration file's fields, an outline of the algorithm, and instructions for use.


## Task Example: Mortality Prediction
<Task querying schema colourful timelines>


## Dependencies
- numpy
- polars
- bigtree
- ruamel.yaml


## Instructions for Use

1. **Prepare the YAML Configuration File**: Define your predicates and windows according to your research needs. Please see below for details regarding the configuration language.
2. **Load the ESGPT Library**: Set-up and import ESGPT into your environment.
3. **Run the Query**: Use ESGPT Task Querying with your YAML file to query the dataset.
4. **Results**: The output will be a dataframe of subjects who satisfy the conditions defined in your YAML file. Timestamps for the end of each window specified in the YAML, as well as predicate counts for each window, are also provided.


## YAML Configuration File

The YAML configuration file allows users to define specific predicates and windows to query the dataset. Below is a description of each field:

### Predicates
Predicates describe the event at a timestamp. <something about naming them is_cols and also initialized as binary counts.> There are two types of predicates. They can represent explicit ESD events and be defined by (`column`, `value`) pairs:
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

Each window uses these fields to define specific time frames and criteria within the dataset.

A sample YAML configuration file is provided in `sample_config.yaml`.


## Recursive Algorithm Description

A tree structure is constructed based on the windows defined in the configuration file. This tree represents the hierarchical relationship between different time windows, where each node represents a window with its specific constraints.

`summarize_temporal_window()`: Creates a summary of predicate counts within a specified temporally-bound window.

`summarize_event_bound_window()` Creates a summary of predicate counts within a specified event-bound window.

`summarize_window()`: Combines the functionalities of the above two functions.

`check_constraints()`: Checks if the predicate counts in a window satisfy the inclusion and exclusion constraints of the window.

A function is called recursively to query each subtree in the tree structure. The function checks constraints, summarizes windows, and joins results from nodes in the subtree.


## Acknowledgement

Appeared at NeurIPS 2023 on the Event Stream GPT Poster: ???

For any questions, enhancements, or issues, please file a GitHub issue. For inquiries regarding Event Stream GPT, please refer to the ESGPT repository. Contributions to the codebase are also welcome via pull requests.

