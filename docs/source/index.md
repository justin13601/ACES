# Welcome to ESGPTTaskQuerying's Documentation!

EventStreamGPT (ESGPT) is a library that streamlines the development of generative, pre-trained transformers (i.e., foundation models) over event stream datasets, such as Electronic Health Records (EHR). ESGPT is designed to extract, preprocess, and manage these datasets efficiently, providing a Huggingface-compatible modeling API and introducing critical capabilities for representing complex intra-event causal dependencies and measuring zero-shot performance. For more detailed information, please refer to the ESGPT GitHub repository: ESGPT GitHub Repo.

A feature of ESGPT is the ability to query EHR datasets for valid subjects, guided by various constraints and requirements defined in a YAML configuration file. This streamlines the process of extracting task-specific cohorts from large time-series datasets, offering a powerful and user-friendly solution to researchers and developers. The use of a human-readable YAML configuration file also eliminates the need for users to be proficient in complex dataframe querying, making the querying process accessible to a broader audience.

There are diverse applications in healthcare and beyond. For instance, researchers can effortlessly define subsets of EHR datasets for training of foundational models. Retrospective analyses can also become more accessible to clinicians as it enables the extraction of tailored cohorts for studying specific medical conditions or population demographics.

Check out {doc}`/usage` for further information,
and {ref}`Installation` for installation instructions.

```{warning}
   This library is under development.
```

## Contents

```{toctree}
---
glob:
maxdepth: 2
---
Overview <overview>
Usage Guide <usage>
Sample Data Tutorial <tutorial/index>
Algorithm & Terminology Reference <terminology>
Module Reference <api/modules>
License <license>
```
