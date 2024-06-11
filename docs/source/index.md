# Welcome!

<p align="center">
  <a href="https://eventstreamaces.readthedocs.io/en/latest/index.html"><img alt="ACES" src="https://raw.githubusercontent.com/justin13601/ACES/bbde3d2047d30f2203cc09a288a8e3565a0d7d62/docs/source/assets/aces_logo_text.svg" width=35%></a>
</p>

ACES is a library designed for the automatic extraction of cohorts from event-stream datasets for downstream machine learning tasks. Check out below for an overview of ACES and how it could be useful in your workflows!

## Why ACES?

Why should use this ecosystem? If you have a dataset you want to do something with, you can do the following:

______________________________________________________________________

```{toctree}
---
glob:
maxdepth: 2
---
GitHub README <readme>
Usage Guide <usage>
Task Examples <notebooks/examples>
Sample Data Tutorial <notebooks/tutorial>
Predicates DataFrame <notebooks/predicates>
Configuration Language <configuration>
Algorithm & Terminology <terminology>
Module API Reference <api/modules>
License <license>
```

______________________________________________________________________

### 1. Transform to MEDS

Put your dataset in MEDS form (Link to ETL) -- this will take some human effort, but is simple, easy (relative to other CDMs), and won’t introduce significant biases in your data relative to its raw form.

### 2. Identify Predicates

Identify the predicates necessary for the tasks of interest either for new tasks or in the pre-defined criteria we have across these N tasks and M clinical areas you can find here: <LINK>

### 3. Set Dataset-Agnostic Criteria

Merge those predicates into the dataset-agnostic criteria files at the prior link -- see <HERE> and <HERE> for examples of these on other public (MIMIC-IV, eICU) or private (...) datasets

### 4. Run ACES

Run ACES to extract your tasks (show command)

### 5. Run MEDS-Tab

Run MEDS-Tab to produce comparable, reproducible, and well-tuned XGBoost results for each of these tasks over your dataset-specific feature space (see here)

______________________________________________________________________

## Examples

You can see this in action for MIMIC-IV <HERE>, eICU <HERE>, and it reliably takes no more than a week of full time human effort to perform steps i - v on new datasets in reasonable raw formulations.

______________________________________________________________________
