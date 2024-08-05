from datetime import datetime

import polars as pl
import pytest

from src.aces.constraints import check_static_variables


def test_check_static_variables():
    # Create a sample DataFrame
    predicates_df = pl.DataFrame(
        {
            "subject_id": [1, 1, 1, 1, 1, 2, 2, 2],
            "timestamp": [
                None,
                datetime(year=1989, month=12, day=1, hour=12, minute=3),
                datetime(year=1989, month=12, day=2, hour=5, minute=17),
                datetime(year=1989, month=12, day=2, hour=12, minute=3),
                datetime(year=1989, month=12, day=6, hour=11, minute=0),
                None,
                datetime(year=1989, month=12, day=1, hour=13, minute=14),
                datetime(year=1989, month=12, day=3, hour=15, minute=17),
            ],
            "is_A": [0, 1, 4, 1, 0, 3, 3, 3],
            "is_B": [0, 0, 2, 0, 0, 2, 10, 2],
            "is_C": [0, 1, 1, 1, 0, 0, 1, 1],
            "male": [1, 0, 0, 0, 0, 0, 0, 0],
        }
    )

    # Test filtering based on 'male' demographic
    filtered_df = check_static_variables(["male"], predicates_df)
    expected_df = pl.DataFrame(
        {
            "subject_id": [1, 1, 1, 1],
            "timestamp": [
                datetime(year=1989, month=12, day=1, hour=12, minute=3),
                datetime(year=1989, month=12, day=2, hour=5, minute=17),
                datetime(year=1989, month=12, day=2, hour=12, minute=3),
                datetime(year=1989, month=12, day=6, hour=11, minute=0),
            ],
            "is_A": [1, 4, 1, 0],
            "is_B": [0, 2, 0, 0],
            "is_C": [1, 1, 1, 0],
        }
    )
    assert filtered_df.frame_equal(expected_df)

    # Test ValueError when demographic column is missing
    with pytest.raises(ValueError, match="Static predicate 'female' not found in the predicates dataframe."):
        check_static_variables(["female"], predicates_df)


if __name__ == "__main__":
    pytest.main([__file__])
