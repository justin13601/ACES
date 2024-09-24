"""This subpackage contains the Hydra configuration groups for ACES, which can be used for `aces-cli`.

Configuration Group File Structure:

.. code-block:: text

    config/
    ├─ data/
    │  ├─ single_file.yaml
    │  ├─ defaults.yaml
    │  ├─ sharded.yaml
    ├─ aces.yaml


`aces-cli` help message:

.. code-block:: text

    ================== aces-cli ===================
    Welcome to the command-line interface for ACES!

    This end-to-end tool extracts a cohort from the external dataset based on a defined task configuration
    file and saves the output file(s). Several data standards are supported, including `meds` (requires a
    dataset in the MEDS format, either with a single shard or multiple shards), `esgpt` (requires a dataset
    in the ESGPT format), and `direct` (requires a pre-computed predicates dataframe as well as a timestamp
    format). Hydra multi-run (`-m`) and sweep capabilities are supported, and launchers can be configured.

    ------------- Configuration Groups ------------
    $APP_CONFIG_GROUPS
    `data` is defaulted to `data=single_file`. Use `data=sharded` to enable extraction with multiple shards
    on MEDS.

    ------------------ Arguments ------------------
    data.*:
        - path (required): path to the data directory if using MEDS with multiple shards or ESGPT, or path to
        the data `.parquet` if using MEDS with a single shard, or path to the predicates dataframe
        (`.csv` or `.parquet`) if using `direct`
        - standard (required): data standard, one of  'meds', 'esgpt', or 'direct'
        - ts_format (required if data.standard is 'direct'): timestamp format for the data
        - root (required, applicable when data=sharded): root directory for the data shards
        - shard (required, applicable when data=sharded): shard number of specific shard from a MEDS dataset.

        Note: data.shard can be expanded using the `expand_shards` function. Please refer to
        https://eventstreamaces.readthedocs.io/en/latest/usage.html#multiple-shards and
        https://github.com/justin13601/ACES/blob/main/src/aces/expand_shards.py for more information.

    cohort_dir (required): cohort directory, used to automatically load configs, saving results, and logging
    cohort_name (required): cohort name, used to automatically load configs, saving results, and logging
    config_path (optional): path to the task configuration file, defaults to '<cohort_dir>/<cohort_name>.yaml'
    predicates_path (optional): path to a separate predicates-only configuration file for overriding
    output_filepath (optional): path to the output file, defaults to '<cohort_dir>/<cohort_name>.parquet'

    ---------------- Default Config ----------------
    $CONFIG
    ------------------------------------------------
    All fields may be overridden via the command-line interface. For example:

        aces-cli cohort_name="..." cohort_dir="..." data.standard="..." data="..." data.root="..."
                "data.shard=$$(expand_shards .../...)" ...

    For more information, visit: https://eventstreamaces.readthedocs.io/en/latest/usage.html

    Powered by Hydra (https://hydra.cc)
    Use --hydra-help to view Hydra specific help
    ===============================================
"""
