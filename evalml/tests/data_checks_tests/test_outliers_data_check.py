import string

import numpy as np
import pandas as pd

from evalml.data_checks import (
    DataCheckMessageCode,
    DataCheckWarning,
    OutliersDataCheck
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
    X.iloc[:, 90] = 'string_values'

    outliers_check = OutliersDataCheck()
    assert outliers_check.validate(X) == {
        "warnings": [DataCheckWarning(message="Column(s) '3', '25', '55', '72' are likely to have outlier data.",
                                      data_check_name=outliers_data_check_name,
                                      message_code=DataCheckMessageCode.HAS_OUTLIERS,
                                      details={"columns": [3, 25, 55, 72]}).to_dict()],
        "errors": [],
        "actions": []
    }


def test_outliers_data_check_input_formats():
    outliers_check = OutliersDataCheck()

    # test empty pd.DataFrame
    assert outliers_check.validate(pd.DataFrame()) == {"warnings": [], "errors": [], "actions": []}

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
        "warnings": [DataCheckWarning(message="Column(s) '3', '25', '55', '72' are likely to have outlier data.",
                                      data_check_name=outliers_data_check_name,
                                      message_code=DataCheckMessageCode.HAS_OUTLIERS,
                                      details={"columns": [3, 25, 55, 72]}).to_dict()],
        "errors": [],
        "actions": []
    }

    # test Woodwork
    outliers_check = OutliersDataCheck()
    X.ww.init()
    assert outliers_check.validate(X) == {
        "warnings": [DataCheckWarning(message="Column(s) '3', '25', '55', '72' are likely to have outlier data.",
                                      data_check_name=outliers_data_check_name,
                                      message_code=DataCheckMessageCode.HAS_OUTLIERS,
                                      details={"columns": [3, 25, 55, 72]}).to_dict()],
        "errors": [],
        "actions": []
    }


def test_outliers_data_check_string_cols():
    a = np.arange(10) * 0.01
    data = np.tile(a, (100, 2))
    n_cols = 20

    X = pd.DataFrame(data=data, columns=[string.ascii_lowercase[i] for i in range(n_cols)])
    X.iloc[0, 3] = 1000

    outliers_check = OutliersDataCheck()
    assert outliers_check.validate(X) == {
        "warnings": [DataCheckWarning(message="Column(s) 'd' are likely to have outlier data.",
                                      data_check_name=outliers_data_check_name,
                                      message_code=DataCheckMessageCode.HAS_OUTLIERS,
                                      details={"columns": ["d"]}).to_dict()],
        "errors": [],
        "actions": []
    }


def test_outlier_score():
    a = np.arange(10) * 0.01
    data = np.tile(a, (100, 10))

    X = pd.DataFrame(data=data)
    X.iloc[0, 3] = 1000
    X.iloc[3, 25] = 1000
    X.iloc[5, 55] = 10000
    X.iloc[10, 72] = -1000

    for col in X.columns:
        results = OutliersDataCheck._outlier_score(X[col], convert_column=False)
        if col in [3, 25, 55, 72]:
            assert results['score'] != 1.0
            assert len(results['values']['high_values']) != 0 or len(results['values']['low_values']) != 0
        else:
            assert results['score'] == 1.0
            assert len(results['values']['high_values']) == 0 and len(results['values']['low_values']) == 0


def test_outlier_score_convert_column_to_int():
    has_outlier = pd.Series(np.append(np.arange(10), 1000)).astype(object)
    results = OutliersDataCheck._outlier_score(has_outlier, convert_column=True)
    assert results['score'] != 1.0
    len(results['values']['high_values']) != 0 or len(results['values']['low_values']) != 0
    no_outlier = pd.Series(np.arange(10)).astype(object)
    results = OutliersDataCheck._outlier_score(no_outlier, convert_column=True)
    assert results['score'] == 1.0
    len(results['values']['high_values']) == 0 and len(results['values']['low_values']) == 0


def test_outlier_score_all_nan():
    all_nan = pd.Series([np.nan, np.nan, np.nan])
    assert OutliersDataCheck._outlier_score(all_nan) is None
