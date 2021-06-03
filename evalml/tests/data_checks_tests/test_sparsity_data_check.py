import pandas as pd
import pytest

from evalml.data_checks import (
    DataCheckAction,
    DataCheckActionCode,
    DataCheckMessageCode,
    DataCheckWarning,
    SparsityDataCheck,
)

sparsity_data_check_name = SparsityDataCheck.name


def test_sparsity_data_check_init():

    sparsity_check = SparsityDataCheck("multiclass", threshold=4 / 15)
    assert sparsity_check.threshold == 4 / 15

    sparsity_check = SparsityDataCheck("multiclass", threshold=0.2)
    assert sparsity_check.unique_count_threshold == 10

    sparsity_check = SparsityDataCheck(
        "multiclass", threshold=0.1, unique_count_threshold=5
    )
    assert sparsity_check.unique_count_threshold == 5

    with pytest.raises(
        ValueError, match="Threshold must be a float between 0 and 1, inclusive."
    ):
        SparsityDataCheck("multiclass", threshold=-0.1)
    with pytest.raises(
        ValueError, match="Threshold must be a float between 0 and 1, inclusive."
    ):
        SparsityDataCheck("multiclass", threshold=1.1)

    with pytest.raises(
        ValueError, match="Sparsity is only defined for multiclass problem types."
    ):
        SparsityDataCheck("binary", threshold=0.5)
    with pytest.raises(
        ValueError, match="Sparsity is only defined for multiclass problem types."
    ):
        SparsityDataCheck("time series binary", threshold=0.5)
    with pytest.raises(
        ValueError, match="Sparsity is only defined for multiclass problem types."
    ):
        SparsityDataCheck("regression", threshold=0.5)
    with pytest.raises(
        ValueError, match="Sparsity is only defined for multiclass problem types."
    ):
        SparsityDataCheck("time series regression", threshold=0.5)

    with pytest.raises(
        ValueError, match="Unique count threshold must be positive integer."
    ):
        SparsityDataCheck("multiclass", threshold=0.5, unique_count_threshold=-1)
    with pytest.raises(
        ValueError, match="Unique count threshold must be positive integer."
    ):
        SparsityDataCheck("multiclass", threshold=0.5, unique_count_threshold=2.3)


def test_sparsity_data_check_sparsity_score():
    # Application to a Series
    # Here, only 0 exceedes the count_threshold of 3.  0 is 1/3 unique values.  So the score is 1/3.
    data = pd.Series([x % 3 for x in range(10)])  # [0,1,2,0,1,2,0,1,2,0]
    scores = SparsityDataCheck.sparsity_score(data, count_threshold=3)
    assert round(scores, 6) == round(1 / 3, 6), "Sparsity Series check failed."

    # Another application to a Series
    # Here, 1 exceeds the count_threshold of 3.  1 is 1/1 unique values, so the score is 1.
    data = pd.Series([1, 1, 1, 1, 1, 1, 1, 1])
    scores = SparsityDataCheck.sparsity_score(data, count_threshold=3)
    assert scores == 1

    # Another application to a Series
    # Here, 1 does not exceed the count_threshold of 10.  1 is 1/1 unique values, so the score is 0.
    data = pd.Series([1, 1, 1, 1, 1, 1, 1, 1])
    scores = SparsityDataCheck.sparsity_score(data, count_threshold=10)
    assert scores == 0

    # Application to an entire DataFrame
    data = pd.DataFrame(
        {
            "most_sparse": [float(x) for x in range(10)],  # [0,1,2,3,4,5,6,7,8,9]
            "more_sparse": [x % 5 for x in range(10)],  # [0,1,2,3,4,0,1,2,3,4]
            "sparse": [x % 3 for x in range(10)],  # [0,1,2,0,1,2,0,1,2,0]
            "less_sparse": [x % 2 for x in range(10)],  # [0,1,0,1,0,1,0,1,0,1]
            "not_sparse": [float(1) for x in range(10)],
        }
    )  # [1,1,1,1,1,1,1,1,1,1]
    sparsity_score = SparsityDataCheck.sparsity_score
    scores = data.apply(sparsity_score, count_threshold=3)
    ans = pd.Series(
        {
            "most_sparse": 0.000000,
            "more_sparse": 0.000000,
            "sparse": 0.333333,
            "less_sparse": 1.000000,
            "not_sparse": 1.000000,
        }
    )
    assert scores.round(6).equals(ans), "Sparsity DataFrame check failed."


def test_sparsity_data_check_warnings():
    data = pd.DataFrame(
        {
            "most_sparse": [float(x) for x in range(10)],  # [0,1,2,3,4,5,6,7,8,9]
            "more_sparse": [x % 5 for x in range(10)],  # [0,1,2,3,4,0,1,2,3,4]
            "sparse": [x % 3 for x in range(10)],  # [0,1,2,0,1,2,0,1,2,0]
            "less_sparse": [x % 2 for x in range(10)],  # [0,1,0,1,0,1,0,1,0,1]
            "not_sparse": [float(1) for x in range(10)],
        }
    )  # [1,1,1,1,1,1,1,1,1,1]

    sparsity_check = SparsityDataCheck(
        problem_type="multiclass", threshold=0.4, unique_count_threshold=3
    )

    assert sparsity_check.validate(data) == {
        "warnings": [
            DataCheckWarning(
                message="Input columns (most_sparse) for multiclass problem type are too sparse.",
                data_check_name=sparsity_data_check_name,
                message_code=DataCheckMessageCode.TOO_SPARSE,
                details={"column": "most_sparse", "sparsity_score": 0},
            ).to_dict(),
            DataCheckWarning(
                message="Input columns (more_sparse) for multiclass problem type are too sparse.",
                data_check_name=sparsity_data_check_name,
                message_code=DataCheckMessageCode.TOO_SPARSE,
                details={"column": "more_sparse", "sparsity_score": 0},
            ).to_dict(),
            DataCheckWarning(
                message="Input columns (sparse) for multiclass problem type are too sparse.",
                data_check_name=sparsity_data_check_name,
                message_code=DataCheckMessageCode.TOO_SPARSE,
                details={"column": "sparse", "sparsity_score": 0.3333333333333333},
            ).to_dict(),
        ],
        "errors": [],
        "actions": [
            DataCheckAction(
                DataCheckActionCode.DROP_COL, metadata={"column": "most_sparse"}
            ).to_dict(),
            DataCheckAction(
                DataCheckActionCode.DROP_COL, metadata={"column": "more_sparse"}
            ).to_dict(),
            DataCheckAction(
                DataCheckActionCode.DROP_COL, metadata={"column": "sparse"}
            ).to_dict(),
        ],
    }
