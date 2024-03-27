import os
import pandas as pd
import polars as pl

import cProfile, pstats, sys
from task_querying_v2 import *

name = 'profiling'
config_path = f"test_configs/{name}.yaml"

total_rows = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000, 50000000, 100000000, 150000000]

for i in total_rows:
    df_temp = df_data.head(i)
    print(f"Number of rows: {df_temp.shape[0]}")
    print(f"Number of patients: {df_temp['subject_id'].n_unique()}")

    pr = cProfile.Profile()
    pr.enable()

    df_result = query_task(config_path, df_temp, verbose=False)

    pr.disable()
    ps = pstats.Stats(pr, stream=sys.stdout)
    ps.sort_stats("cumtime").print_stats()

    print(f"Number of rows: {df_result.shape[0]}")
    print(f"Number of patients: {df_result['subject_id'].n_unique()}")
    print("=====================================NEXT=====================================")