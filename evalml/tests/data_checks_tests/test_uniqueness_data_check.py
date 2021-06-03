import numpy as np
import pandas as pd
import pytest

from evalml.data_checks import (
    DataCheckAction,
    DataCheckActionCode,
    DataCheckMessageCode,
    DataCheckWarning,
    UniquenessDataCheck,
)

uniqueness_data_check_name = UniquenessDataCheck.name


def test_uniqueness_data_check_init():
    uniqueness_check = UniquenessDataCheck("regression")
    assert uniqueness_check.threshold == 0.50

    uniqueness_check = UniquenessDataCheck("regression", threshold=0.0)
    assert uniqueness_check.threshold == 0

    uniqueness_check = UniquenessDataCheck("regression", threshold=0.5)
    assert uniqueness_check.threshold == 0.5

    uniqueness_check = UniquenessDataCheck("regression", threshold=1.0)
    assert uniqueness_check.threshold == 1.0

    with pytest.raises(
        ValueError, match="threshold must be a float between 0 and 1, inclusive."
    ):
        UniquenessDataCheck("regression", threshold=-0.1)
    with pytest.raises(
        ValueError, match="threshold must be a float between 0 and 1, inclusive."
    ):
        UniquenessDataCheck("regression", threshold=1.1)


def test_uniqueness_data_check_uniqueness_score():
    uniqueness_score = UniquenessDataCheck.uniqueness_score

    # Test uniqueness for a simple series.
    # [0,1,2,0,1,2,0,1,2,0]
    data = pd.Series([x % 3 for x in range(10)])
    scores = uniqueness_score(data)
    ans = 0.66
    assert scores == ans

    # Test uniqueness for the same series, repeated.  Should be the score.
    # [0,1,2,0,1,2,0,1,2,0,0,1,2,0,1,2,0,1,2,0]
    data = pd.Series([x % 3 for x in range(10)] * 2)
    scores = uniqueness_score(data)
    ans = 0.66
    assert scores == ans

    # Test uniqueness for a simple series with NaN.
    # [0,1,2,0,1,2,0,1,2,0]
    data = pd.Series([x % 3 for x in range(10)] + [np.nan])
    scores = uniqueness_score(data)
    ans = 0.66
    assert scores == ans

    # Test uniqueness in each column of a DataFrame
    data = pd.DataFrame(
        {
            "most_unique": [float(x) for x in range(10)],  # [0,1,2,3,4,5,6,7,8,9]
            "more_unique": [x % 5 for x in range(10)],  # [0,1,2,3,4,0,1,2,3,4]
            "unique": [x % 3 for x in range(10)],  # [0,1,2,0,1,2,0,1,2,0]
            "less_unique": [x % 2 for x in range(10)],  # [0,1,0,1,0,1,0,1,0,1]
            "not_unique": [float(1) for x in range(10)],
        }
    )  # [1,1,1,1,1,1,1,1,1,1]
    scores = data.apply(uniqueness_score)
    ans = pd.Series(
        {
            "most_unique": 0.90,
            "more_unique": 0.80,
            "unique": 0.66,
            "less_unique": 0.50,
            "not_unique": 0.00,
        }
    )
    assert scores.round(7).equals(ans)


def test_uniqueness_data_check_warnings():
    data = pd.DataFrame(
        {
            "regression_unique_enough": [float(x) for x in range(100)],
            "regression_not_unique_enough": [float(1) for x in range(100)],
        }
    )

    uniqueness_check = UniquenessDataCheck(problem_type="regression")
    assert uniqueness_check.validate(data) == {
        "warnings": [
            DataCheckWarning(
                message="Input columns (regression_not_unique_enough) for regression problem type are not unique enough.",
                data_check_name=uniqueness_data_check_name,
                message_code=DataCheckMessageCode.NOT_UNIQUE_ENOUGH,
                details={
                    "column": "regression_not_unique_enough",
                    "uniqueness_score": 0.0,
                },
            ).to_dict()
        ],
        "errors": [],
        "actions": [
            DataCheckAction(
                DataCheckActionCode.DROP_COL,
                metadata={"column": "regression_not_unique_enough"},
            ).to_dict()
        ],
    }

    data = pd.DataFrame(
        {
            "multiclass_too_unique": ["Cats", "Are", "Absolutely", "The", "Best"] * 20,
            "multiclass_not_too_unique": ["Cats", "Cats", "Best", "Best", "Best"] * 20,
        }
    )
    uniqueness_check = UniquenessDataCheck(problem_type="multiclass")
    assert uniqueness_check.validate(data) == {
        "warnings": [
            DataCheckWarning(
                message="Input columns (multiclass_too_unique) for multiclass problem type are too unique.",
                data_check_name=uniqueness_data_check_name,
                message_code=DataCheckMessageCode.TOO_UNIQUE,
                details={
                    "column": "multiclass_too_unique",
                    "uniqueness_score": 0.7999999999999999,
                },
            ).to_dict()
        ],
        "errors": [],
        "actions": [
            DataCheckAction(
                DataCheckActionCode.DROP_COL,
                metadata={"column": "multiclass_too_unique"},
            ).to_dict()
        ],
    }
