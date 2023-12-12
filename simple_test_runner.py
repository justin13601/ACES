from typing import Any, Callable
from datetime import datetime
import polars as pl
pl.Config.set_tbl_cols(100)
pl.Config.set_tbl_rows(100)

# cases = [
#     {
#         "message": "...",
#         "args": [],
#         "want": ,
#     },
# ]

def simple_test_runner(
  cases: list[dict[str, Any]], fn: Callable,
  equal_fn: Callable = (lambda x, y: x.frame_equal(y))
):
  for C in cases:
    got = None
    try:
        got = fn(*C['args'])
        if not equal_fn(C['want'], got):
            print(f"{C['message']} Failed.")
            print(f"Want:\n{C['want']}")
            print(f"Got: {got}")
        else:
            print("Passed")
    except Exception as e:
      print(f"{C['message']} Error!")
      print(f"Want:\n{C['want']}")
      print(f"Error: {e}.")


# def simple_test_runner(
#   cases: list[dict[str, Any]], fn: Callable,
#   equal_fn: Callable = (lambda x, y: x == y)
# ):
#   for C in cases:
#     got = None
#     try:
#       got = fn(*C['args'])
#     except Exception as e:
#       print(f"Failed {C['message']}")
#       print(f"Want:\n{C['want']}")
#       print(f"Got error {e}")

#     if not equal_fn(C['want'], got):
#       print(f"Failed {C['message']}")
#       print(f"Want:\n{C['want']}")
#       print(f"Got: {got}")
#       raise ValueError("Failed")


# predicates_df = pl.DataFrame({
#     'subject_id': [1, 1, 1, 2, 2, 3, 3],
#     'timestamp': [
#         datetime(1919, 12, 1),
#         datetime(1922, 12, 1),
#         datetime(1929, 12, 1),
#         datetime(1919, 12, 1),
#         datetime(1929, 12, 1),
#         datetime(1929, 12, 1),
#         datetime(1939, 12, 1),
#     ],
#     'is_A': [1, 1, 0, 0, 1, 0, 0],
#     'is_B': [0, 1, 1, 1, 1, 0, 1],
#     'is_C': [0, 0, 0, 0, 1, 1, 0],
# })

# anchor_to_subtree_root_by_subtree_anchor = pl.DataFrame({
#     'subject_id': [1, 2, 3],
#     'timestamp': [
#         datetime(1919, 12, 1),
#         datetime(1919, 12, 1),
#         datetime(1939, 12, 1),
#     ],
#     'is_A': [0, 0, 0],
#     'is_B': [0, 0, 0],
#     'is_C': [0, 0, 0],
# })

# predicate_cols = ['is_A', 'is_B', 'is_C']

# predicates_df

# result = pl.DataFrame({
#     'subject_id': [1, 1],
#     'timestamp': [
#         datetime(1922, 12, 1),
#         datetime(1929, 12, 1),
#     ],
#     'is_A': [0, 0],
#     'is_B': [1, 2],
#     'is_C': [0, 0],
# })

# cases = [
#     {
#         "message": "Should work",
#         "args": (predicates_df, predicate_cols, (False, "is_B", True), anchor_to_subtree_root_by_subtree_anchor),
#         "want": result,
#     }
# ]


# simple_test_runner(cases, summarize_event_bound_window)