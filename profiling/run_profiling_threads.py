try:
    import stackprinter

    stackprinter.set_excepthook(style="darkbg2")
except ImportError:
    pass  # no need to fail because of missing dev dependency

import cProfile
import os
import pickle
import platform
import pstats
import sys
from pathlib import Path

import polars as pl
import psutil
from EventStream.data.dataset_polars import Dataset
from loguru import logger

from esgpt_task_querying import main


def get_machine_details():
    machine_details = {
        "polars_threads": pl.thread_pool_size(),
        "platform": platform.platform(),
        "memory": psutil.virtual_memory(),
        "cpu": platform.processor(),
        "cpu_freq": psutil.cpu_freq(),
        "cores": psutil.cpu_count(logical=False),
    }
    return machine_details


def profile_based_on_num_threads(DATA_DIR, config):
    logger.info(f"{pl.thread_pool_size()} Threads")
    pr = cProfile.Profile()
    pr.enable()
    ESD = Dataset.load(DATA_DIR)
    events_df = ESD.events_df
    dynamic_measurements_df = ESD.dynamic_measurements_df
    pr.disable()
    ps = pstats.Stats(pr, stream=sys.stdout)
    load_time = ps.total_tt
    logger.info(f"Load time: {load_time}")

    pr.enable()
    events_df = events_df.filter(~pl.all_horizontal(pl.all().is_null()))
    dynamic_measurements_df = dynamic_measurements_df.filter(~pl.all_horizontal(pl.all().is_null()))
    df_data = (
        events_df.join(dynamic_measurements_df, on="event_id", how="left")
        .drop(["event_id"])
        .sort(by=["subject_id", "timestamp", "event_type"])
    )
    pr.disable()
    ps = pstats.Stats(pr, stream=sys.stdout)
    preprocess_time = ps.total_tt - load_time
    logger.info(f"Preprocess time: {preprocess_time}")

    df_temp = df_data
    logger.info(f"Number of rows: {df_temp.shape[0]}")
    logger.info(f"Number of patients: {df_temp['subject_id'].n_unique()}")

    pr = cProfile.Profile()
    pr.enable()
    df_result = main.query_task(config, df_temp)
    pr.disable()
    ps = pstats.Stats(pr, stream=sys.stdout)
    query_time = ps.total_tt
    logger.info(f"Query time: {query_time}")

    logger.info(f"Number of rows: {df_result.shape[0]}")
    logger.info(f"Number of patients: {df_result['subject_id'].n_unique()}")

    profiling_result = {
        "num_threads": pl.thread_pool_size(),
        "load_time": load_time,
        "preprocess_time": preprocess_time,
        "query_time": query_time,
        "cumulative_time": load_time + preprocess_time + query_time,
        "original_rows": df_temp.shape[0],
        "original_patients": df_temp["subject_id"].n_unique(),
        "result_rows": df_result.shape[0],
        "result_patients": df_result["subject_id"].n_unique(),
        "notes": get_machine_details(),
    }

    return profiling_result


if __name__ == "__main__":
    DATA_DIR = Path("../../MIMIC_ESD_new_schema_08-31-23-1")
    output_dir = Path("profiling_output")

    os.makedirs(output_dir, exist_ok=True)

    config = "profiling_configs/profile_based_on_num_threads.yaml"
    profiling_result = profile_based_on_num_threads(DATA_DIR, config)

    with open(output_dir / "profiling_result.pkl", "wb") as f:
        pickle.dump(profiling_result, f)
