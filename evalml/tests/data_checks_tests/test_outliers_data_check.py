import string

import numpy as np
import pandas as pd

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
