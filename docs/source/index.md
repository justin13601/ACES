# Welcome!

<p align="center">
  <a href="https://eventstreamaces.readthedocs.io/en/latest/index.html"><img alt="ACES" src="https://raw.githubusercontent.com/justin13601/ACES/bbde3d2047d30f2203cc09a288a8e3565a0d7d62/docs/source/assets/aces_logo_text.svg" width=35%></a>
</p>

ACES is a library designed for the automatic extraction of cohorts from event-stream datasets for downstream machine learning tasks. Check out below for an overview of ACES and how it could be useful in your workflows!

```{toctree}
---
glob:
maxdepth: 2
---
README <readme>
Usage Guide <usage>
Task Examples <notebooks/examples>
Predicates DataFrame <notebooks/predicates>
MEDS Data Tutorial <notebooks/tutorial_meds>
Technical Details <technical>
Computational Profile <profiling>
Module API Reference <api/modules>
License <license>
```

______________________________________________________________________

## Why ACES?

If you have a dataset and want to leverage it for machine learning tasks, the ACES ecosystem offers a streamlined and user-friendly approach. Here's how you can easily transform, prepare, and utilize your dataset with MEDS and ACES for efficient and effective machine learning:

### I. Transform to MEDS

- Simplicity: Converting your dataset to the Medical Event Data Standard (MEDS) is straightforward and user-friendly compared to other Common Data Models (CDMs).
- Minimal Bias: This conversion process ensures that your data remains as close to its raw form as possible, minimizing the introduction of biases.
- [MEDS-ETL](https://github.com/Medical-Event-Data-Standard/meds_etl): Follow this link for detailed instructions and ETLs to transform your dataset into the MEDS format!

### II. Identify Predicates

- Task-Specific Concepts: Identify the predicates (data concepts) required for your specific machine learning tasks.
- Pre-Defined Criteria: Utilize our pre-defined criteria across various tasks and clinical areas to expedite this process.
- [MEDS-DEV](https://github.com/mmcdermott/MEDS-DEV/tree/main): Access our benchmark of tasks to find relevant predicates!

### III. Set Dataset-Agnostic Criteria

- Standardization: Combine the identified predicates with standardized, dataset-agnostic criteria files.
- Examples: Refer to the [MEDS-DEV](https://github.com/mmcdermott/MEDS-DEV/tree/main/src/MEDS_DEV/tasks/criteria) examples for guidance on how to structure your criteria files for your private datasets!

### IV. Run ACES

- Run the ACES Command-Line Interface tool (`aces-cli`) to extract cohorts based on your task - check out the [Usage Guide](https://eventstreamaces.readthedocs.io/en/latest/usage.html) for more information!

### V. Run MEDS-Tab

- Painless Reproducibility: Use [MEDS-Tab](https://github.com/mmcdermott/MEDS_TAB_MIMIC_IV/tree/main) to obtain comparable, reproducible, and well-tuned XGBoost results tailored to your dataset-specific feature space!

By following these steps, you can seamlessly transform your dataset, define necessary criteria, and leverage powerful machine learning tools within the ACES and MEDS ecosystem. This approach not only simplifies the process but also ensures high-quality, reproducible results for your machine learning for health projects. It can reliably take no more than a week of full-time human effort to perform Steps I-V on new datasets in reasonable raw formulations!
