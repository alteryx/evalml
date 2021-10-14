import string

import numpy as np
import pandas as pd
import pytest

from evalml.data_checks import (
    DataCheckMessageCode,
    DataCheckWarning,
    OutliersDataCheck,
)

outliers_data_check_name = OutliersDataCheck.name


def test_outliers_data_check_warnings():
    a = np.arange(10) * 0.01
    data = np.tile(a, (100, 10))

    X = pd.DataFrame(data=data)
    X.iloc[0, 3] = 1000
    X.iloc[3, 25] = 1000
    X.iloc[5, 55] = 10000
    X.iloc[10, 72] = -1000
    X.iloc[:, 90] = "string_values"

    outliers_check = OutliersDataCheck()
    assert outliers_check.validate(X) == {
        "warnings": [
            DataCheckWarning(
                message="Column(s) '3', '25', '55', '72' are likely to have outlier data.",
                data_check_name=outliers_data_check_name,
                message_code=DataCheckMessageCode.HAS_OUTLIERS,
                details={
                    "columns": [3, 25, 55, 72],
                    "rows": {3: [0], 25: [3], 55: [5], 72: [10]},
                },
            ).to_dict()
        ],
        "errors": [],
        "actions": [{"code": "DROP_ROWS", "metadata": {"indices": [0, 3, 5, 10]}}],
    }


def test_outliers_data_check_warnings_with_duplicate_outlier_indices():
    a = np.arange(10) * 0.01
    data = np.tile(a, (100, 10))

    X = pd.DataFrame(data=data)
    X.iloc[0, 3] = 1000
    X.iloc[3, 25] = 1000
    X.iloc[0, 55] = 10000
    X.iloc[0, 72] = -1000
    X.iloc[:, 90] = "string_values"

    outliers_check = OutliersDataCheck()
    assert outliers_check.validate(X) == {
        "warnings": [
            DataCheckWarning(
                message="Column(s) '3', '25', '55', '72' are likely to have outlier data.",
                data_check_name=outliers_data_check_name,
                message_code=DataCheckMessageCode.HAS_OUTLIERS,
                details={
                    "columns": [3, 25, 55, 72],
                    "rows": {3: [0], 25: [3], 55: [0], 72: [0]},
                },
            ).to_dict()
        ],
        "errors": [],
        "actions": [{"code": "DROP_ROWS", "metadata": {"indices": [0, 3]}}],
    }


def test_outliers_data_check_input_formats():
    outliers_check = OutliersDataCheck()

    # test empty pd.DataFrame
    assert outliers_check.validate(pd.DataFrame()) == {
        "warnings": [],
        "errors": [],
        "actions": [],
    }

    # test np.array
    a = np.arange(10) * 0.01
    data = np.tile(a, (100, 10))

    X = pd.DataFrame(data=data)
    X.iloc[0, 3] = 1000
    X.iloc[3, 25] = 1000
    X.iloc[5, 55] = 10000
    X.iloc[10, 72] = -1000

    outliers_check = OutliersDataCheck()
    assert outliers_check.validate(X.to_numpy()) == {
        "warnings": [
            DataCheckWarning(
                message="Column(s) '3', '25', '55', '72' are likely to have outlier data.",
                data_check_name=outliers_data_check_name,
                message_code=DataCheckMessageCode.HAS_OUTLIERS,
                details={
                    "columns": [3, 25, 55, 72],
                    "rows": {3: [0], 25: [3], 55: [5], 72: [10]},
                },
            ).to_dict()
        ],
        "errors": [],
        "actions": [{"code": "DROP_ROWS", "metadata": {"indices": [0, 3, 5, 10]}}],
    }

    # test Woodwork
    outliers_check = OutliersDataCheck()
    X.ww.init()
    assert outliers_check.validate(X) == {
        "warnings": [
            DataCheckWarning(
                message="Column(s) '3', '25', '55', '72' are likely to have outlier data.",
                data_check_name=outliers_data_check_name,
                message_code=DataCheckMessageCode.HAS_OUTLIERS,
                details={
                    "columns": [3, 25, 55, 72],
                    "rows": {3: [0], 25: [3], 55: [5], 72: [10]},
                },
            ).to_dict()
        ],
        "errors": [],
        "actions": [{"code": "DROP_ROWS", "metadata": {"indices": [0, 3, 5, 10]}}],
    }


def test_outliers_data_check_string_cols():
    a = np.arange(10) * 0.01
    data = np.tile(a, (100, 2))
    n_cols = 20

    X = pd.DataFrame(
        data=data, columns=[string.ascii_lowercase[i] for i in range(n_cols)]
    )
    X.iloc[0, 3] = 1000

    outliers_check = OutliersDataCheck()
    assert outliers_check.validate(X) == {
        "warnings": [
            DataCheckWarning(
                message="Column(s) 'd' are likely to have outlier data.",
                data_check_name=outliers_data_check_name,
                message_code=DataCheckMessageCode.HAS_OUTLIERS,
                details={"columns": ["d"], "rows": {"d": [0]}},
            ).to_dict()
        ],
        "errors": [],
        "actions": [{"code": "DROP_ROWS", "metadata": {"indices": [0]}}],
    }


def test_outlier_score_all_nan():
    all_nan = pd.DataFrame(
        [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]
    )
    outliers_check = OutliersDataCheck()
    assert outliers_check.validate(all_nan) == {
        "warnings": [],
        "errors": [],
        "actions": [],
    }


def test_outliers_data_check_warnings_has_nan():
    a = np.arange(10) * 0.01
    data = np.tile(a, (100, 10))

    X = pd.DataFrame(data=data)
    X.iloc[0, 3] = np.nan
    X.iloc[3, 25] = 1000
    X.iloc[5, 55] = 10000
    X.iloc[10, 72] = -1000
    X.iloc[:, 90] = "string_values"

    outliers_check = OutliersDataCheck()
    assert outliers_check.validate(X) == {
        "warnings": [
            DataCheckWarning(
                message="Column(s) '25', '55', '72' are likely to have outlier data.",
                data_check_name=outliers_data_check_name,
                message_code=DataCheckMessageCode.HAS_OUTLIERS,
                details={"columns": [25, 55, 72], "rows": {25: [3], 55: [5], 72: [10]}},
            ).to_dict()
        ],
        "errors": [],
        "actions": [{"code": "DROP_ROWS", "metadata": {"indices": [3, 5, 10]}}],
    }


@pytest.mark.parametrize("data_type", ["int", "mixed"])
def test_boxplot_stats(data_type):
    test = pd.Series(
        [32, 33, 34, 95, 96, 36, 37, 1.5 if data_type == "mixed" else 1, 2]
    )

    q1, median, q3 = np.percentile(test, [25, 50, 75])

    try:
        from statsmodels.stats.stattools import medcouple

        medcouple_stat = medcouple(list(test))

        field_bounds = (
            q1 - 1.5 * np.exp(-3.79 * medcouple_stat) * (q3 - q1),
            q3 + 1.5 * np.exp(3.87 * medcouple_stat) * (q3 - q1),
        )
        assert OutliersDataCheck._get_boxplot_data(test) == {
            "score": OutliersDataCheck._no_outlier_prob(9, 4 / 9),
            "values": {
                "q1": q1,
                "median": median,
                "q3": q3,
                "low_bound": field_bounds[0],
                "high_bound": field_bounds[1],
                "low_values": test[test < field_bounds[0]].tolist(),
                "high_values": test[test > field_bounds[1]].tolist(),
                "low_indices": test[test < field_bounds[0]].index.tolist(),
                "high_indices": test[test > field_bounds[1]].index.tolist(),
            },
        }
    except ModuleNotFoundError:
        assert OutliersDataCheck._get_boxplot_data(test) is None
