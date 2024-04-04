import os
import pandas as pd
import polars as pl

import subprocess
import pickle
from pathlib import Path
import platform
import psutil
import cProfile, pstats, sys

# from esgpt_task_querying import main
import task_querying_v2 as esgpt_task_querying
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
    print(f"Load time: {load_time}")

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
    print(f"Preprocess time: {preprocess_time}")

    config = "test_configs/profile_based_on_num_original_rows.yaml"

    profiling_results = []
    for i in original_rows:
        print(
            f"====================================={i} Rows====================================="
        )
        df_temp = df_data.head(i)
        print(f"Number of rows: {df_temp.shape[0]}")
        print(f"Number of patients: {df_temp['subject_id'].n_unique()}")

        pr = cProfile.Profile()
        pr.enable()
        df_result = esgpt_task_querying.query_task(config, df_temp, verbose=False)
        pr.disable()
        ps = pstats.Stats(pr, stream=sys.stdout)
        query_time = ps.total_tt
        print(f"Query time: {query_time}")

        print(f"Number of rows: {df_result.shape[0]}")
        print(f"Number of patients: {df_result['subject_id'].n_unique()}")

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
    print(f"Load time: {load_time}")

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
    print(f"Preprocess time: {preprocess_time}")

    if num_rows:
        df_temp = df_data.head(num_rows)
    else:
        df_temp = df_data

    profiling_results = []
    for i in num_predicates:
        print(
            f"====================================={i} Extra Predicates====================================="
        )
        print(f"Number of rows: {df_temp.shape[0]}")
        print(f"Number of patients: {df_temp['subject_id'].n_unique()}")

        config = f"test_configs/profile_based_on_num_predicates_{i}.yaml"

        pr = cProfile.Profile()
        pr.enable()
        df_result = esgpt_task_querying.query_task(
            config, df_temp, verbose=False
        )
        pr.disable()
        ps = pstats.Stats(pr, stream=sys.stdout)
        query_time = ps.total_tt
        print(f"Query time: {query_time}")

        print(f"Number of rows: {df_result.shape[0]}")
        print(f"Number of patients: {df_result['subject_id'].n_unique()}")

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
    print(f"Load time: {load_time}")

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
    print(f"Preprocess time: {preprocess_time}")

    if num_rows:
        df_temp = df_data.head(num_rows)
    else:
        df_temp = df_data

    profiling_results = []
    for i in num_criteria:
        print(
            f"====================================={i} Extra Criteria ====================================="
        )
        print(f"Number of rows: {df_temp.shape[0]}")
        print(f"Number of patients: {df_temp['subject_id'].n_unique()}")

        config = f"test_configs/profile_based_on_num_criteria_{i}.yaml"

        pr = cProfile.Profile()
        pr.enable()
        df_result = esgpt_task_querying.query_task(
            config, df_temp, verbose=False
        )
        pr.disable()
        ps = pstats.Stats(pr, stream=sys.stdout)
        query_time = ps.total_tt
        print(f"Query time: {query_time}")

        print(f"Number of rows: {df_result.shape[0]}")
        print(f"Number of patients: {df_result['subject_id'].n_unique()}")

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
    print(f"Load time: {load_time}")

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
    print(f"Preprocess time: {preprocess_time}")

    if num_rows:
        df_temp = df_data.head(num_rows)
    else:
        df_temp = df_data

    profiling_results = []
    for i in num_criteria:
        print(
            f"====================================={i} Extra Windows in Series ====================================="
        )
        print(f"Number of rows: {df_temp.shape[0]}")
        print(f"Number of patients: {df_temp['subject_id'].n_unique()}")

        config = f"test_configs/profile_based_on_num_windows_in_series_{i}.yaml"

        pr = cProfile.Profile()
        pr.enable()
        df_result = esgpt_task_querying.query_task(
            config, df_temp, verbose=False
        )
        pr.disable()
        ps = pstats.Stats(pr, stream=sys.stdout)
        query_time = ps.total_tt
        print(f"Query time: {query_time}")

        print(f"Number of rows: {df_result.shape[0]}")
        print(f"Number of patients: {df_result['subject_id'].n_unique()}")

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
    print(f"Load time: {load_time}")

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
    print(f"Preprocess time: {preprocess_time}")

    if num_rows:
        df_temp = df_data.head(num_rows)
    else:
        df_temp = df_data

    profiling_results = []
    for i in num_criteria:
        print(
            f"====================================={i} Extra Windows in Parallel ====================================="
        )
        print(f"Number of rows: {df_temp.shape[0]}")
        print(f"Number of patients: {df_temp['subject_id'].n_unique()}")

        config = f"test_configs/profile_based_on_num_windows_in_parallel_{i}.yaml"

        pr = cProfile.Profile()
        pr.enable()
        df_result = esgpt_task_querying.query_task(
            config, df_temp, verbose=False
        )
        pr.disable()
        ps = pstats.Stats(pr, stream=sys.stdout)
        query_time = ps.total_tt
        print(f"Query time: {query_time}")

        print(f"Number of rows: {df_result.shape[0]}")
        print(f"Number of patients: {df_result['subject_id'].n_unique()}")

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
    print(f"Load time: {load_time}")

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
    print(f"Preprocess time: {preprocess_time}")

    if num_rows:
        df_temp = df_data.head(num_rows)
    else:
        df_temp = df_data

    profiling_results = []
    for i in tasks:
        print(
            f"=====================================Task: {i}====================================="
        )
        print(f"Number of rows: {df_temp.shape[0]}")
        print(f"Number of patients: {df_temp['subject_id'].n_unique()}")

        config = f"../sample_configs/{i}.yaml"

        pr = cProfile.Profile()
        pr.enable()
        df_result = esgpt_task_querying.query_task(
            config, df_temp, verbose=False
        )
        pr.disable()
        ps = pstats.Stats(pr, stream=sys.stdout)
        query_time = ps.total_tt
        print(f"Query time: {query_time}")

        print(f"Number of rows: {df_result.shape[0]}")
        print(f"Number of patients: {df_result['subject_id'].n_unique()}")

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
    DATA_DIR = Path("../MIMIC_ESD_new_schema_08-31-23-1")
    output_dir = Path("profiling_output")
    ############ DIRECTORIES ############

    os.makedirs(output_dir, exist_ok=True)

    ############ Number of original rows ############
    num_rows = [
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
    # profile_based_on_num_original_rows(DATA_DIR, config, output_dir, num_rows)

    ############ Number of predicates ############
    num_predicates = [0, 1, 2, 4, 8, 16]
    # profile_based_on_num_predicates(DATA_DIR, output_dir, num_predicates, num_rows=10000000)
    
    ############ Number of criteria ############
    num_critera = [0, 1, 2, 4, 8, 16]
    # profile_based_on_num_criteria(DATA_DIR, output_dir, num_critera, num_rows=150000000)

    ############ Number of windows in series (segments) ############
    num_windows_series = [0, 1, 2, 4, 8, 16]
    # profile_based_on_num_windows_in_series(DATA_DIR, output_dir, num_windows_series, num_rows=150000000)

    ############ Number of windows in parallel (overlapping) ############
    num_windows_parallel = [0, 1, 2, 4, 8, 16]
    # profile_based_on_num_windows_in_parallel(DATA_DIR, output_dir, num_windows_parallel, num_rows=150000000)

    ############ Various tasks ############
    tasks = [
        'inhospital_mortality',
        'abnormal_lab',
        'imminent_mortality',
        'outlier_detection',
        'readmission_risk',
        'long_term_incidence',
        'intervention_weaning',
    ]
    # profile_based_on_task(DATA_DIR, output_dir, tasks)

    ############ Number of threads ############
    # Warning: Will run inhospital mortality on full dataset, so will take a really long time to load the data with low number of threads
    num_threads = [36, 32, 28, 24, 20, 16, 12, 8, 4, 2, 1]
    # profile_based_on_num_threads(output_dir, num_threads)
