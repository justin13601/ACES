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
import subprocess
import sys
from pathlib import Path

import hydra
import pandas as pd
import polars as pl
import psutil
from EventStream.data.dataset_polars import Dataset
from EventStream.logger import hydra_loguru_init
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from esgpt_task_querying import query


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

    config = "profiling_configs/profile_based_on_num_original_rows.yaml"

    profiling_results = []
    for i in original_rows:
        logger.info(f"====================================={i} Rows=====================================")
        df_temp = df_data.head(i)
        logger.info(f"Number of rows: {df_temp.shape[0]}")
        logger.info(f"Number of patients: {df_temp['subject_id'].n_unique()}")

        pr = cProfile.Profile()
        pr.enable()
        df_result = query.query(config, df_temp)
        pr.disable()
        ps = pstats.Stats(pr, stream=sys.stdout)
        query_time = ps.total_tt
        logger.info(f"Query time: {query_time}")

        logger.info(f"Number of rows: {df_result.shape[0]}")
        logger.info(f"Number of patients: {df_result['subject_id'].n_unique()}")

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
        with open(os.path.join(output_dir, "profiling_result.pkl"), "rb") as f:
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

    if num_rows:
        df_temp = df_data.head(num_rows)
    else:
        df_temp = df_data

    profiling_results = []
    for i in num_predicates:
        logger.info(
            f"====================================={i} Extra Predicates====================================="
        )
        logger.info(f"Number of rows: {df_temp.shape[0]}")
        logger.info(f"Number of patients: {df_temp['subject_id'].n_unique()}")

        config = f"profiling_configs/profile_based_on_num_predicates_{i}.yaml"

        pr = cProfile.Profile()
        pr.enable()
        df_result = query.query(config, df_temp)
        pr.disable()
        ps = pstats.Stats(pr, stream=sys.stdout)
        query_time = ps.total_tt
        logger.info(f"Query time: {query_time}")

        logger.info(f"Number of rows: {df_result.shape[0]}")
        logger.info(f"Number of patients: {df_result['subject_id'].n_unique()}")

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

    if num_rows:
        df_temp = df_data.head(num_rows)
    else:
        df_temp = df_data

    profiling_results = []
    for i in num_criteria:
        logger.info(
            f"====================================={i} Extra Criteria ====================================="
        )
        logger.info(f"Number of rows: {df_temp.shape[0]}")
        logger.info(f"Number of patients: {df_temp['subject_id'].n_unique()}")

        config = f"profiling_configs/profile_based_on_num_criteria_{i}.yaml"

        pr = cProfile.Profile()
        pr.enable()
        df_result = query.query(config, df_temp)
        pr.disable()
        ps = pstats.Stats(pr, stream=sys.stdout)
        query_time = ps.total_tt
        logger.info(f"Query time: {query_time}")

        logger.info(f"Number of rows: {df_result.shape[0]}")
        logger.info(f"Number of patients: {df_result['subject_id'].n_unique()}")

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
    preprocess_time = ps.total_tt
    logger.info(f"Preprocess time: {preprocess_time}")

    if num_rows:
        df_temp = df_data.head(num_rows)
    else:
        df_temp = df_data

    profiling_results = []
    for i in num_criteria:
        logger.info(f"========================={i} Extra Windows in Series ============================")
        logger.info(f"Number of rows: {df_temp.shape[0]}")
        logger.info(f"Number of patients: {df_temp['subject_id'].n_unique()}")

        config = f"profiling_configs/profile_based_on_num_windows_in_series_{i}.yaml"

        pr = cProfile.Profile()
        pr.enable()
        df_result = query.query(config, df_temp)
        pr.disable()
        ps = pstats.Stats(pr, stream=sys.stdout)
        query_time = ps.total_tt
        logger.info(f"Query time: {query_time}")

        logger.info(f"Number of rows: {df_result.shape[0]}")
        logger.info(f"Number of patients: {df_result['subject_id'].n_unique()}")

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

    if num_rows:
        df_temp = df_data.head(num_rows)
    else:
        df_temp = df_data

    profiling_results = []
    for i in num_criteria:
        logger.info(f"============================{i} Extra Windows in Parallel =========================")
        logger.info(f"Number of rows: {df_temp.shape[0]}")
        logger.info(f"Number of patients: {df_temp['subject_id'].n_unique()}")

        config = f"profiling_configs/profile_based_on_num_windows_in_parallel_{i}.yaml"

        pr = cProfile.Profile()
        pr.enable()
        df_result = query.query(config, df_temp)
        pr.disable()
        ps = pstats.Stats(pr, stream=sys.stdout)
        query_time = ps.total_tt
        logger.info(f"Query time: {query_time}")

        logger.info(f"Number of rows: {df_result.shape[0]}")
        logger.info(f"Number of patients: {df_result['subject_id'].n_unique()}")

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

    if num_rows:
        df_temp = df_data.head(num_rows)
    else:
        df_temp = df_data

    profiling_results = []
    for i in tasks:
        logger.info(f"=====================================Task: {i}=====================================")
        logger.info(f"Number of rows: {df_temp.shape[0]}")
        logger.info(f"Number of patients: {df_temp['subject_id'].n_unique()}")

        config = f"../../sample_configs/{i}.yaml"

        pr = cProfile.Profile()
        pr.enable()
        df_result = query.query(config, df_temp)
        pr.disable()
        ps = pstats.Stats(pr, stream=sys.stdout)
        query_time = ps.total_tt
        logger.info(f"Query time: {query_time}")

        logger.info(f"Number of rows: {df_result.shape[0]}")
        logger.info(f"Number of patients: {df_result['subject_id'].n_unique()}")

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


@hydra.main(version_base=None, config_path="configs/", config_name="profiling_config")
def main(cfg: DictConfig):
    hydra_loguru_init()

    cfg = hydra.utils.instantiate(cfg, _convert_="all")

    experiment_dir = Path(cfg["experiment_dir"])

    cfg_fp = experiment_dir / "hydra_config.yaml"
    cfg_fp.parent.mkdir(exist_ok=True, parents=True)
    OmegaConf.save(cfg, cfg_fp)

    output_dir = experiment_dir / "results"
    output_dir.mkdir(exist_ok=True, parents=True)

    experiment_dir / "configs"

    # profile_based_on_num_original_rows(DATA_DIR, output_dir, cfg["num_rows"])
    # profile_based_on_num_predicates(DATA_DIR, output_dir, cfg["num_predicates"], num_rows=10000000)
    # profile_based_on_num_criteria(DATA_DIR, output_dir, cfg["num_criteria"], num_rows=150000000)
    # profile_based_on_num_windows_in_series(
    #   DATA_DIR, output_dir, cfg["num_windows_series"], num_rows=150000000
    # )
    # profile_based_on_num_windows_in_parallel(
    #   DATA_DIR, output_dir, cfg["num_windows_parallel"], num_rows=150000000
    # )
    # profile_based_on_task(DATA_DIR, output_dir, [fp.stem for fp in configs_dir.glob("*.yaml")])

    # Warning: Will run inhospital mortality on full dataset, so will take a really long time to load the data
    # with low number of threads
    profile_based_on_num_threads(output_dir, cfg["num_threads"])


if __name__ == "__main__":
    main()
