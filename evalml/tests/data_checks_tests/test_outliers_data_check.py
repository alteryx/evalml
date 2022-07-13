import string

import numpy as np
import pandas as pd
import pytest

from evalml.data_checks import (
    DataCheckActionCode,
    DataCheckActionOption,
    DataCheckMessageCode,
    DataCheckWarning,
    OutliersDataCheck,
)

outliers_data_check_name = OutliersDataCheck.name


def test_outliers_data_check_no_warnings():
    a = np.arange(10) * 0.01
    data = np.tile(a, (100, 10))
    X = pd.DataFrame(data=data)

    outliers_check = OutliersDataCheck()
    assert outliers_check.validate(X) == []


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
    assert outliers_check.validate(X) == [
        DataCheckWarning(
            message="Column(s) '3', '25', '55', '72' are likely to have outlier data.",
            data_check_name=outliers_data_check_name,
            message_code=DataCheckMessageCode.HAS_OUTLIERS,
            details={
                "columns": [3, 25, 55, 72],
                "rows": [0, 3, 5, 10],
                "column_indices": {3: [0], 25: [3], 55: [5], 72: [10]},
            },
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_ROWS,
                    data_check_name=outliers_data_check_name,
                    metadata={"rows": [0, 3, 5, 10]},
                ),
            ],
        ).to_dict(),
    ]


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
    assert outliers_check.validate(X) == [
        DataCheckWarning(
            message="Column(s) '3', '25', '55', '72' are likely to have outlier data.",
            data_check_name=outliers_data_check_name,
            message_code=DataCheckMessageCode.HAS_OUTLIERS,
            details={
                "columns": [3, 25, 55, 72],
                "rows": [0, 3],
                "column_indices": {3: [0], 25: [3], 55: [0], 72: [0]},
            },
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_ROWS,
                    data_check_name=outliers_data_check_name,
                    metadata={"rows": [0, 3]},
                ),
            ],
        ).to_dict(),
    ]


def test_outliers_data_check_input_formats():
    outliers_check = OutliersDataCheck()

    # test empty pd.DataFrame
    assert outliers_check.validate(pd.DataFrame()) == []

    # test np.array
    a = np.arange(10) * 0.01
    data = np.tile(a, (100, 10))

    X = pd.DataFrame(data=data)
    X.iloc[0, 3] = 1000
    X.iloc[3, 25] = 1000
    X.iloc[5, 55] = 10000
    X.iloc[10, 72] = -1000

    outliers_check = OutliersDataCheck()
    assert outliers_check.validate(X.to_numpy()) == [
        DataCheckWarning(
            message="Column(s) '3', '25', '55', '72' are likely to have outlier data.",
            data_check_name=outliers_data_check_name,
            message_code=DataCheckMessageCode.HAS_OUTLIERS,
            details={
                "columns": [3, 25, 55, 72],
                "rows": [0, 3, 5, 10],
                "column_indices": {3: [0], 25: [3], 55: [5], 72: [10]},
            },
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_ROWS,
                    data_check_name=outliers_data_check_name,
                    metadata={"rows": [0, 3, 5, 10]},
                ),
            ],
        ).to_dict(),
    ]

    # test Woodwork
    outliers_check = OutliersDataCheck()
    X.ww.init()
    assert outliers_check.validate(X) == [
        DataCheckWarning(
            message="Column(s) '3', '25', '55', '72' are likely to have outlier data.",
            data_check_name=outliers_data_check_name,
            message_code=DataCheckMessageCode.HAS_OUTLIERS,
            details={
                "columns": [3, 25, 55, 72],
                "rows": [0, 3, 5, 10],
                "column_indices": {3: [0], 25: [3], 55: [5], 72: [10]},
            },
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_ROWS,
                    data_check_name=outliers_data_check_name,
                    metadata={"rows": [0, 3, 5, 10]},
                ),
            ],
        ).to_dict(),
    ]


def test_outliers_data_check_string_cols():
    a = np.arange(10) * 0.01
    data = np.tile(a, (100, 2))
    n_cols = 20

    X = pd.DataFrame(
        data=data,
        columns=[string.ascii_lowercase[i] for i in range(n_cols)],
    )
    X.iloc[0, 3] = 1000

    outliers_check = OutliersDataCheck()
    assert outliers_check.validate(X) == [
        DataCheckWarning(
            message="Column(s) 'd' are likely to have outlier data.",
            data_check_name=outliers_data_check_name,
            message_code=DataCheckMessageCode.HAS_OUTLIERS,
            details={
                "columns": ["d"],
                "rows": [0],
                "column_indices": {"d": [0]},
            },
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_ROWS,
                    data_check_name=outliers_data_check_name,
                    metadata={"rows": [0]},
                ),
            ],
        ).to_dict(),
    ]


def test_outlier_score_all_nan():
    all_nan = pd.DataFrame(
        [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
    )
    outliers_check = OutliersDataCheck()
    assert outliers_check.validate(all_nan) == []


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
    assert outliers_check.validate(X) == [
        DataCheckWarning(
            message="Column(s) '25', '55', '72' are likely to have outlier data.",
            data_check_name=outliers_data_check_name,
            message_code=DataCheckMessageCode.HAS_OUTLIERS,
            details={
                "columns": [25, 55, 72],
                "rows": [3, 5, 10],
                "column_indices": {25: [3], 55: [5], 72: [10]},
            },
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_ROWS,
                    data_check_name=outliers_data_check_name,
                    metadata={"rows": [3, 5, 10]},
                ),
            ],
        ).to_dict(),
    ]


@pytest.mark.parametrize("data_type", ["int", "mixed"])
def test_boxplot_stats(data_type):
    test = pd.Series(
        [32, 33, 34, None, 96, 36, 37, 1.5 if data_type == "mixed" else 1, 2],
    )

    quantiles = test.quantile([0.25, 0.5, 0.75]).to_dict()
    iqr = quantiles[0.75] - quantiles[0.25]
    field_bounds = (quantiles[0.25] - (iqr * 1.5), quantiles[0.75] + (iqr * 1.5))
    pct_outliers = (
        len(test[test <= field_bounds[0]].tolist())
        + len(test[test >= field_bounds[1]].tolist())
    ) / test.count()

    assert OutliersDataCheck.get_boxplot_data(test) == {
        "score": OutliersDataCheck._no_outlier_prob(test.count(), pct_outliers),
        "pct_outliers": pct_outliers,
        "values": {
            "q1": quantiles[0.25],
            "median": quantiles[0.5],
            "q3": quantiles[0.75],
            "low_bound": field_bounds[0],
            "high_bound": field_bounds[1],
            "low_values": test[test < field_bounds[0]].tolist(),
            "high_values": test[test > field_bounds[1]].tolist(),
            "low_indices": test[test < field_bounds[0]].index.tolist(),
            "high_indices": test[test > field_bounds[1]].index.tolist(),
        },
    }
