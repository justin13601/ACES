import os
import pandas as pd
import polars as pl
from loguru import logger

import subprocess
import pickle
from pathlib import Path
import platform
import psutil
import cProfile, pstats, sys

from esgpt_task_querying import main
from EventStream.data.dataset_polars import Dataset


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


def profile_based_on_num_original_rows(DATA_DIR, output_dir, original_rows):
    pr = cProfile.Profile()
    pr.enable()
    ESD = Dataset.load(DATA_DIR)
    events_df = ESD.events_df
    dynamic_measurements_df = ESD.dynamic_measurements_df
    pr.disable()
    ps = pstats.Stats(pr, stream=sys.stdout)
    load_time = ps.total_tt
    logger.debug(f"Load time: {load_time}")

    pr.enable()
    events_df = events_df.filter(~pl.all_horizontal(pl.all().is_null()))
    dynamic_measurements_df = dynamic_measurements_df.filter(
        ~pl.all_horizontal(pl.all().is_null())
    )
    df_data = (
        events_df.join(dynamic_measurements_df, on="event_id", how="left")
        .drop(["event_id"])
        .sort(by=["subject_id", "timestamp", "event_type"])
    )
    pr.disable()
    ps = pstats.Stats(pr, stream=sys.stdout)
    preprocess_time = ps.total_tt - load_time
    logger.debug(f"Preprocess time: {preprocess_time}")

    config = "profile_based_on_num_original_rows.yaml"

    profiling_results = []
    for i in original_rows:
        logger.debug(
            f"====================================={i} Rows====================================="
        )
        df_temp = df_data.head(i)
        logger.debug(f"Number of rows: {df_temp.shape[0]}")
        logger.debug(f"Number of patients: {df_temp['subject_id'].n_unique()}")

        pr = cProfile.Profile()
        pr.enable()
        df_result = main.query_task(config, df_temp, verbose=False)
        pr.disable()
        ps = pstats.Stats(pr, stream=sys.stdout)
        query_time = ps.total_tt
        logger.debug(f"Query time: {query_time}")

        logger.debug(f"Number of rows: {df_result.shape[0]}")
        logger.debug(f"Number of patients: {df_result['subject_id'].n_unique()}")

        profiling_result = {
            "num_rows": i,
            "load_time": load_time,
            "preprocess_time": preprocess_time,
            "query_time": query_time,
            "cumulative_time": load_time + preprocess_time + query_time,
            "original_rows": i,
            "original_patients": df_temp["subject_id"].n_unique(),
            "result_rows": df_result.shape[0],
            "result_patients": df_result["subject_id"].n_unique(),
            "notes": get_machine_details(),
        }
        profiling_results.append(profiling_result)

    df_profiling_results = pd.DataFrame(profiling_results)
    df_profiling_results.to_csv(
        os.path.join(output_dir, "profile_based_on_original_rows_results.csv"),
        index=False,
    )


def profile_based_on_num_threads(output_dir, num_threads):
    profiling_results = []
    for i in num_threads:
        os.environ["POLARS_MAX_THREADS"] = str(i)
        subprocess.run(["python", "run_profiling_threads.py"])
        with open(os.path.join(output_dir, 'profiling_result.pkl'), "rb") as f:
            profiling_result = pickle.load(f)
        profiling_results.append(profiling_result)

    df_profiling_results = pd.DataFrame(profiling_results)
    df_profiling_results.to_csv(
        os.path.join(output_dir, "profile_based_on_num_threads_results.csv"),
        index=False,
    )


def profile_based_on_num_predicates(DATA_DIR, output_dir, num_predicates, num_rows=None):
    pr = cProfile.Profile()
    pr.enable()
    ESD = Dataset.load(DATA_DIR)
    events_df = ESD.events_df
    dynamic_measurements_df = ESD.dynamic_measurements_df
    pr.disable()
    ps = pstats.Stats(pr, stream=sys.stdout)
    load_time = ps.total_tt
    logger.debug(f"Load time: {load_time}")

    pr.enable()
    events_df = events_df.filter(~pl.all_horizontal(pl.all().is_null()))
    dynamic_measurements_df = dynamic_measurements_df.filter(
        ~pl.all_horizontal(pl.all().is_null())
    )
    df_data = (
        events_df.join(dynamic_measurements_df, on="event_id", how="left")
        .drop(["event_id"])
        .sort(by=["subject_id", "timestamp", "event_type"])
    )
    pr.disable()
    ps = pstats.Stats(pr, stream=sys.stdout)
    preprocess_time = ps.total_tt - load_time
    logger.debug(f"Preprocess time: {preprocess_time}")

    if num_rows:
        df_temp = df_data.head(num_rows)
    else:
        df_temp = df_data

    profiling_results = []
    for i in num_predicates:
        logger.debug(
            f"====================================={i} Extra Predicates====================================="
        )
        logger.debug(f"Number of rows: {df_temp.shape[0]}")
        logger.debug(f"Number of patients: {df_temp['subject_id'].n_unique()}")

        config = f"profile_based_on_num_predicates_{i}.yaml"

        pr = cProfile.Profile()
        pr.enable()
        df_result = main.query_task(
            config, df_temp, verbose=False
        )
        pr.disable()
        ps = pstats.Stats(pr, stream=sys.stdout)
        query_time = ps.total_tt
        logger.debug(f"Query time: {query_time}")

        logger.debug(f"Number of rows: {df_result.shape[0]}")
        logger.debug(f"Number of patients: {df_result['subject_id'].n_unique()}")

        profiling_result = {
            "num_extra_predicates": i,
            "load_time": load_time,
            "preprocess_time": preprocess_time,
            "query_time": query_time,
            "cumulative_time": load_time + preprocess_time + query_time,
            "original_rows": df_temp.shape[0],
            "original_patients": df_temp["subject_id"].n_unique(),
            "result_rows": df_result.shape[0],
            "result_patients: ": df_result["subject_id"].n_unique(),
            "notes": get_machine_details(),
        }
        profiling_results.append(profiling_result)

    df_profiling_results = pd.DataFrame(profiling_results)
    df_profiling_results.to_csv(
        os.path.join(output_dir, "profile_based_on_num_predicates_results.csv"),
        index=False,
    )


def profile_based_on_num_criteria(DATA_DIR, output_dir, num_criteria, num_rows=None):
    pr = cProfile.Profile()
    pr.enable()
    ESD = Dataset.load(DATA_DIR)
    events_df = ESD.events_df
    dynamic_measurements_df = ESD.dynamic_measurements_df
    pr.disable()
    ps = pstats.Stats(pr, stream=sys.stdout)
    load_time = ps.total_tt
    logger.debug(f"Load time: {load_time}")

    pr.enable()
    events_df = events_df.filter(~pl.all_horizontal(pl.all().is_null()))
    dynamic_measurements_df = dynamic_measurements_df.filter(
        ~pl.all_horizontal(pl.all().is_null())
    )
    df_data = (
        events_df.join(dynamic_measurements_df, on="event_id", how="left")
        .drop(["event_id"])
        .sort(by=["subject_id", "timestamp", "event_type"])
    )
    pr.disable()
    ps = pstats.Stats(pr, stream=sys.stdout)
    preprocess_time = ps.total_tt - load_time
    logger.debug(f"Preprocess time: {preprocess_time}")

    if num_rows:
        df_temp = df_data.head(num_rows)
    else:
        df_temp = df_data

    profiling_results = []
    for i in num_criteria:
        logger.debug(
            f"====================================={i} Extra Criteria ====================================="
        )
        logger.debug(f"Number of rows: {df_temp.shape[0]}")
        logger.debug(f"Number of patients: {df_temp['subject_id'].n_unique()}")

        config = f"profile_based_on_num_criteria_{i}.yaml"

        pr = cProfile.Profile()
        pr.enable()
        df_result = main.query_task(
            config, df_temp, verbose=False
        )
        pr.disable()
        ps = pstats.Stats(pr, stream=sys.stdout)
        query_time = ps.total_tt
        logger.debug(f"Query time: {query_time}")

        logger.debug(f"Number of rows: {df_result.shape[0]}")
        logger.debug(f"Number of patients: {df_result['subject_id'].n_unique()}")

        profiling_result = {
            "num_extra_criteria": i,
            "load_time": load_time,
            "preprocess_time": preprocess_time,
            "query_time": query_time,
            "cumulative_time": load_time + preprocess_time + query_time,
            "original_rows": df_temp.shape[0],
            "original_patients": df_temp["subject_id"].n_unique(),
            "result_rows": df_result.shape[0],
            "result_patients: ": df_result["subject_id"].n_unique(),
            "notes": get_machine_details(),
        }
        profiling_results.append(profiling_result)

    df_profiling_results = pd.DataFrame(profiling_results)
    df_profiling_results.to_csv(
        os.path.join(output_dir, "profile_based_on_num_criteria_results.csv"),
        index=False,
    )


def profile_based_on_num_windows_in_series(DATA_DIR, output_dir, num_criteria, num_rows=None):
    pr = cProfile.Profile()
    pr.enable()
    ESD = Dataset.load(DATA_DIR)
    events_df = ESD.events_df
    dynamic_measurements_df = ESD.dynamic_measurements_df
    pr.disable()
    ps = pstats.Stats(pr, stream=sys.stdout)
    load_time = ps.total_tt
    logger.debug(f"Load time: {load_time}")

    pr.enable()
    events_df = events_df.filter(~pl.all_horizontal(pl.all().is_null()))
    dynamic_measurements_df = dynamic_measurements_df.filter(
        ~pl.all_horizontal(pl.all().is_null())
    )
    df_data = (
        events_df.join(dynamic_measurements_df, on="event_id", how="left")
        .drop(["event_id"])
        .sort(by=["subject_id", "timestamp", "event_type"])
    )
    pr.disable()
    ps = pstats.Stats(pr, stream=sys.stdout)
    preprocess_time = ps.total_tt - load_time
    logger.debug(f"Preprocess time: {preprocess_time}")

    if num_rows:
        df_temp = df_data.head(num_rows)
    else:
        df_temp = df_data

    profiling_results = []
    for i in num_criteria:
        logger.debug(
            f"====================================={i} Extra Windows in Series ====================================="
        )
        logger.debug(f"Number of rows: {df_temp.shape[0]}")
        logger.debug(f"Number of patients: {df_temp['subject_id'].n_unique()}")

        config = f"profile_based_on_num_windows_in_series_{i}.yaml"

        pr = cProfile.Profile()
        pr.enable()
        df_result = main.query_task(
            config, df_temp, verbose=False
        )
        pr.disable()
        ps = pstats.Stats(pr, stream=sys.stdout)
        query_time = ps.total_tt
        logger.debug(f"Query time: {query_time}")

        logger.debug(f"Number of rows: {df_result.shape[0]}")
        logger.debug(f"Number of patients: {df_result['subject_id'].n_unique()}")

        profiling_result = {
            "num_extra_windows_in_series": i,
            "load_time": load_time,
            "preprocess_time": preprocess_time,
            "query_time": query_time,
            "cumulative_time": load_time + preprocess_time + query_time,
            "original_rows": df_temp.shape[0],
            "original_patients": df_temp["subject_id"].n_unique(),
            "result_rows": df_result.shape[0],
            "result_patients: ": df_result["subject_id"].n_unique(),
            "notes": get_machine_details(),
        }
        profiling_results.append(profiling_result)

    df_profiling_results = pd.DataFrame(profiling_results)
    df_profiling_results.to_csv(
        os.path.join(output_dir, "profile_based_on_num_windows_in_series_results.csv"),
        index=False,
    )


def profile_based_on_num_windows_in_parallel(DATA_DIR, output_dir, num_criteria, num_rows=None):
    pr = cProfile.Profile()
    pr.enable()
    ESD = Dataset.load(DATA_DIR)
    events_df = ESD.events_df
    dynamic_measurements_df = ESD.dynamic_measurements_df
    pr.disable()
    ps = pstats.Stats(pr, stream=sys.stdout)
    load_time = ps.total_tt
    logger.debug(f"Load time: {load_time}")

    pr.enable()
    events_df = events_df.filter(~pl.all_horizontal(pl.all().is_null()))
    dynamic_measurements_df = dynamic_measurements_df.filter(
        ~pl.all_horizontal(pl.all().is_null())
    )
    df_data = (
        events_df.join(dynamic_measurements_df, on="event_id", how="left")
        .drop(["event_id"])
        .sort(by=["subject_id", "timestamp", "event_type"])
    )
    pr.disable()
    ps = pstats.Stats(pr, stream=sys.stdout)
    preprocess_time = ps.total_tt - load_time
    logger.debug(f"Preprocess time: {preprocess_time}")

    if num_rows:
        df_temp = df_data.head(num_rows)
    else:
        df_temp = df_data

    profiling_results = []
    for i in num_criteria:
        logger.debug(
            f"====================================={i} Extra Windows in Parallel ====================================="
        )
        logger.debug(f"Number of rows: {df_temp.shape[0]}")
        logger.debug(f"Number of patients: {df_temp['subject_id'].n_unique()}")

        config = f"profile_based_on_num_windows_in_parallel_{i}.yaml"

        pr = cProfile.Profile()
        pr.enable()
        df_result = main.query_task(
            config, df_temp, verbose=False
        )
        pr.disable()
        ps = pstats.Stats(pr, stream=sys.stdout)
        query_time = ps.total_tt
        logger.debug(f"Query time: {query_time}")

        logger.debug(f"Number of rows: {df_result.shape[0]}")
        logger.debug(f"Number of patients: {df_result['subject_id'].n_unique()}")

        profiling_result = {
            "num_extra_windows_in_parallel": i,
            "load_time": load_time,
            "preprocess_time": preprocess_time,
            "query_time": query_time,
            "cumulative_time": load_time + preprocess_time + query_time,
            "original_rows": df_temp.shape[0],
            "original_patients": df_temp["subject_id"].n_unique(),
            "result_rows": df_result.shape[0],
            "result_patients: ": df_result["subject_id"].n_unique(),
            "notes": get_machine_details(),
        }
        profiling_results.append(profiling_result)

    df_profiling_results = pd.DataFrame(profiling_results)
    df_profiling_results.to_csv(
        os.path.join(output_dir, "profile_based_on_num_windows_in_parallel_results.csv"),
        index=False,
    )


def profile_based_on_task(DATA_DIR, output_dir, tasks, num_rows=None):
    pr = cProfile.Profile()
    pr.enable()
    ESD = Dataset.load(DATA_DIR)
    events_df = ESD.events_df
    dynamic_measurements_df = ESD.dynamic_measurements_df
    pr.disable()
    ps = pstats.Stats(pr, stream=sys.stdout)
    load_time = ps.total_tt
    logger.debug(f"Load time: {load_time}")

    pr.enable()
    events_df = events_df.filter(~pl.all_horizontal(pl.all().is_null()))
    dynamic_measurements_df = dynamic_measurements_df.filter(
        ~pl.all_horizontal(pl.all().is_null())
    )
    df_data = (
        events_df.join(dynamic_measurements_df, on="event_id", how="left")
        .drop(["event_id"])
        .sort(by=["subject_id", "timestamp", "event_type"])
    )
    pr.disable()
    ps = pstats.Stats(pr, stream=sys.stdout)
    preprocess_time = ps.total_tt - load_time
    logger.debug(f"Preprocess time: {preprocess_time}")

    if num_rows:
        df_temp = df_data.head(num_rows)
    else:
        df_temp = df_data

    profiling_results = []
    for i in tasks:
        logger.debug(
            f"=====================================Task: {i}====================================="
        )
        logger.debug(f"Number of rows: {df_temp.shape[0]}")
        logger.debug(f"Number of patients: {df_temp['subject_id'].n_unique()}")

        config = f"{i}.yaml"

        pr = cProfile.Profile()
        pr.enable()
        df_result = main.query_task(
            config, df_temp, verbose=False
        )
        pr.disable()
        ps = pstats.Stats(pr, stream=sys.stdout)
        query_time = ps.total_tt
        logger.debug(f"Query time: {query_time}")

        logger.debug(f"Number of rows: {df_result.shape[0]}")
        logger.debug(f"Number of patients: {df_result['subject_id'].n_unique()}")

        profiling_result = {
            "num_extra_windows_in_parallel": i,
            "load_time": load_time,
            "preprocess_time": preprocess_time,
            "query_time": query_time,
            "cumulative_time": load_time + preprocess_time + query_time,
            "original_rows": df_temp.shape[0],
            "original_patients": df_temp["subject_id"].n_unique(),
            "result_rows": df_result.shape[0],
            "result_patients: ": df_result["subject_id"].n_unique(),
            "notes": get_machine_details(),
        }
        profiling_results.append(profiling_result)

    df_profiling_results = pd.DataFrame(profiling_results)
    df_profiling_results.to_csv(
        os.path.join(output_dir, "profile_based_on_task_results.csv"),
        index=False,
    )


if __name__ == "__main__":
    ############ DIRECTORIES ############
    DATA_DIR = Path("/n/data1/hms/dbmi/zaklab/inovalon_mbm47/processed/12-19-23_InovalonSample1M")
    output_dir = Path("inovalon_profiling_output")
    ############ DIRECTORIES ############

    ############ VARIABLES TO CHANGE ############
    #Number of original rows
    num_original_rows = [
        10,
        50,
        100,
        500,
        1000,
        5000,
        10000,
        50000,
        100000,
        500000,
        1000000,
        5000000,
        10000000,
        50000000,
        100000000,
        150000000,
    ]

    #Number of rows to test other profiling on, leave as None to test it on full Inovalon, else integer
    num_rows = None

    #Number of polars threads, will run on full Inovalon regardless of num_rows
    num_threads = [16, 12, 8, 4, 2, 1]
    ############ VARIABLES TO CHANGE ############

    os.makedirs(output_dir, exist_ok=True)

    profile_based_on_num_original_rows(DATA_DIR, output_dir, num_original_rows)

    num_predicates = [0, 1, 2, 4, 8, 16]
    profile_based_on_num_predicates(DATA_DIR, output_dir, num_predicates, num_rows=num_rows)
    
    num_critera = [0, 1, 2, 4, 8, 16]
    profile_based_on_num_criteria(DATA_DIR, output_dir, num_critera, num_rows=num_rows)

    num_windows_series = [0, 1, 2, 4]
    profile_based_on_num_windows_in_series(DATA_DIR, output_dir, num_windows_series, num_rows=num_rows)

    num_windows_parallel = [0, 1, 2, 4]
    profile_based_on_num_windows_in_parallel(DATA_DIR, output_dir, num_windows_parallel, num_rows=num_rows)

    tasks = [
        'readmission_risk',
        'long_term_incidence',
    ]
    profile_based_on_task(DATA_DIR, output_dir, tasks)

    profile_based_on_num_threads(output_dir, num_threads)
