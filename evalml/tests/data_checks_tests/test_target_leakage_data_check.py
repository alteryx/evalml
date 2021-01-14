from datetime import datetime

import pandas as pd
import pytest
import woodwork as ww

from evalml.data_checks import (
    DataCheckMessageCode,
    DataCheckWarning,
    TargetLeakageDataCheck
)

target_leakage_data_check_name = TargetLeakageDataCheck.name


def test_target_leakage_data_check_init():
    target_leakage_check = TargetLeakageDataCheck()
    assert target_leakage_check.pct_corr_threshold == 0.95

    target_leakage_check = TargetLeakageDataCheck(pct_corr_threshold=0.0)
    assert target_leakage_check.pct_corr_threshold == 0

    target_leakage_check = TargetLeakageDataCheck(pct_corr_threshold=0.5)
    assert target_leakage_check.pct_corr_threshold == 0.5

    target_leakage_check = TargetLeakageDataCheck(pct_corr_threshold=1.0)
    assert target_leakage_check.pct_corr_threshold == 1.0

    with pytest.raises(ValueError, match="pct_corr_threshold must be a float between 0 and 1, inclusive."):
        TargetLeakageDataCheck(pct_corr_threshold=-0.1)
    with pytest.raises(ValueError, match="pct_corr_threshold must be a float between 0 and 1, inclusive."):
        TargetLeakageDataCheck(pct_corr_threshold=1.1)

    with pytest.raises(ValueError, match="Method 'MUTUAL' not in"):
        TargetLeakageDataCheck(method='MUTUAL')
    with pytest.raises(ValueError, match="Method 'person' not in"):
        TargetLeakageDataCheck(method='person')


def test_target_leakage_data_check_warnings():
    y = pd.Series([1, 0, 1, 1])
    X = pd.DataFrame()
    X["a"] = y * 3
    X["b"] = y - 1
    X["c"] = y / 10
    X["d"] = ~y
    X["e"] = [0, 0, 0, 0]
    y = y.astype(bool)

    leakage_check = TargetLeakageDataCheck(pct_corr_threshold=0.5)
    assert leakage_check.validate(X, y) == {
        "warnings": [DataCheckWarning(message="Column 'a' is 50.0% or more correlated with the target",
                                      data_check_name=target_leakage_data_check_name,
                                      message_code=DataCheckMessageCode.TARGET_LEAKAGE,
                                      details={"column": "a"}).to_dict(),
                     DataCheckWarning(message="Column 'b' is 50.0% or more correlated with the target",
                                      data_check_name=target_leakage_data_check_name,
                                      message_code=DataCheckMessageCode.TARGET_LEAKAGE,
                                      details={"column": "b"}).to_dict(),
                     DataCheckWarning(message="Column 'c' is 50.0% or more correlated with the target",
                                      data_check_name=target_leakage_data_check_name,
                                      message_code=DataCheckMessageCode.TARGET_LEAKAGE,
                                      details={"column": "c"}).to_dict(),
                     DataCheckWarning(message="Column 'd' is 50.0% or more correlated with the target",
                                      data_check_name=target_leakage_data_check_name,
                                      message_code=DataCheckMessageCode.TARGET_LEAKAGE,
                                      details={"column": "d"}).to_dict()],
        "errors": []
    }


@pytest.mark.parametrize("data_type", ['np', 'pd', 'ww'])
def test_target_leakage_data_check_empty(data_type, make_data_type):
    X = make_data_type(data_type, pd.DataFrame())
    y = make_data_type(data_type, pd.Series())
    leakage_check = TargetLeakageDataCheck(pct_corr_threshold=0.8, method='mutual')
    assert leakage_check.validate(X, y) == {"warnings": [], "errors": []}


def test_target_leakage_data_check_input_formats():
    leakage_check = TargetLeakageDataCheck(pct_corr_threshold=0.8)
    y = pd.Series([1, 0, 1, 1])
    X = pd.DataFrame()
    X["a"] = y * 3
    X["b"] = y - 1
    X["c"] = y / 10
    X["d"] = ~y
    X["e"] = [0, 0, 0, 0]
    y = y.astype(bool)

    expected_messages = {
        "warnings": [DataCheckWarning(message="Column 'a' is 80.0% or more correlated with the target",
                                      data_check_name=target_leakage_data_check_name,
                                      message_code=DataCheckMessageCode.TARGET_LEAKAGE,
                                      details={"column": "a"}).to_dict(),
                     DataCheckWarning(message="Column 'b' is 80.0% or more correlated with the target",
                                      data_check_name=target_leakage_data_check_name,
                                      message_code=DataCheckMessageCode.TARGET_LEAKAGE,
                                      details={"column": "b"}).to_dict(),
                     DataCheckWarning(message="Column 'c' is 80.0% or more correlated with the target",
                                      data_check_name=target_leakage_data_check_name,
                                      message_code=DataCheckMessageCode.TARGET_LEAKAGE,
                                      details={"column": "c"}).to_dict(),
                     DataCheckWarning(message="Column 'd' is 80.0% or more correlated with the target",
                                      data_check_name=target_leakage_data_check_name,
                                      message_code=DataCheckMessageCode.TARGET_LEAKAGE,
                                      details={"column": "d"}).to_dict()],
        "errors": []
    }
    # test X as ww.DataTable, y as ww.DataColumn
    assert leakage_check.validate(ww.DataTable(X), ww.DataColumn(y)) == expected_messages

    # test y as list
    assert leakage_check.validate(X, y.values) == expected_messages

    # test X as np.array
    assert leakage_check.validate(X.to_numpy().astype(float), y) == {
        "warnings": [DataCheckWarning(message="Column '0' is 80.0% or more correlated with the target",
                                      data_check_name=target_leakage_data_check_name,
                                      message_code=DataCheckMessageCode.TARGET_LEAKAGE,
                                      details={"column": 0}).to_dict(),
                     DataCheckWarning(message="Column '1' is 80.0% or more correlated with the target",
                                      data_check_name=target_leakage_data_check_name,
                                      message_code=DataCheckMessageCode.TARGET_LEAKAGE,
                                      details={"column": 1}).to_dict(),
                     DataCheckWarning(message="Column '2' is 80.0% or more correlated with the target",
                                      data_check_name=target_leakage_data_check_name,
                                      message_code=DataCheckMessageCode.TARGET_LEAKAGE,
                                      details={"column": 2}).to_dict(),
                     DataCheckWarning(message="Column '3' is 80.0% or more correlated with the target",
                                      data_check_name=target_leakage_data_check_name,
                                      message_code=DataCheckMessageCode.TARGET_LEAKAGE,
                                      details={"column": 3}).to_dict()],
        "errors": []
    }


def test_target_leakage_none():
    leakage_check = TargetLeakageDataCheck(pct_corr_threshold=0.8)
    y = pd.Series([1, 0, 1, 1])
    X = pd.DataFrame()
    X["a"] = [1, 1, 1, 1]
    X["b"] = [0, 0, 0, 0]
    y = y.astype(bool)

    expected_messages = {
        "warnings": [],
        "errors": []
    }

    assert leakage_check.validate(X, y) == expected_messages


def test_target_leakage_types():
    leakage_check = TargetLeakageDataCheck(pct_corr_threshold=0.8)

    y = pd.Series([1, 0, 1, 1])
    X = pd.DataFrame()
    X["a"] = ["a", "b", "a", "a"]
    X["b"] = y - 1
    X["c"] = [datetime.strptime("2015", "%Y"), datetime.strptime("2016", "%Y"), datetime.strptime("2015", "%Y"), datetime.strptime("2015", "%Y")]
    X["d"] = ~y
    X["e"] = [0, 0, 0, 0]
    y = y.astype(bool)

    expected_messages = {
        "warnings": [DataCheckWarning(message="Column 'a' is 80.0% or more correlated with the target",
                                      data_check_name=target_leakage_data_check_name,
                                      message_code=DataCheckMessageCode.TARGET_LEAKAGE,
                                      details={"column": "a"}).to_dict(),
                     DataCheckWarning(message="Column 'b' is 80.0% or more correlated with the target",
                                      data_check_name=target_leakage_data_check_name,
                                      message_code=DataCheckMessageCode.TARGET_LEAKAGE,
                                      details={"column": "b"}).to_dict(),
                     DataCheckWarning(message="Column 'c' is 80.0% or more correlated with the target",
                                      data_check_name=target_leakage_data_check_name,
                                      message_code=DataCheckMessageCode.TARGET_LEAKAGE,
                                      details={"column": "c"}).to_dict(),
                     DataCheckWarning(message="Column 'd' is 80.0% or more correlated with the target",
                                      data_check_name=target_leakage_data_check_name,
                                      message_code=DataCheckMessageCode.TARGET_LEAKAGE,
                                      details={"column": "d"}).to_dict()],
        "errors": []
    }

    assert leakage_check.validate(X, y) == expected_messages


def test_target_leakage_multi():
    leakage_check = TargetLeakageDataCheck(pct_corr_threshold=0.8)

    # test empty pd.DataFrame, empty pd.Series
    assert leakage_check.validate(pd.DataFrame(), pd.Series()) == {"warnings": [], "errors": []}

    y = pd.Series([1, 0, 2, 1, 2, 0])
    X = pd.DataFrame()
    X["a"] = y * 3
    X["b"] = y - 1
    X["c"] = y / 10
    X["d"] = [0, 0, 0, 0, 0, 0]
    X["e"] = ["a", "b", "c", "a", "b", "c"]

    expected_messages = {
        "warnings": [DataCheckWarning(message="Column 'a' is 80.0% or more correlated with the target",
                                      data_check_name=target_leakage_data_check_name,
                                      message_code=DataCheckMessageCode.TARGET_LEAKAGE,
                                      details={"column": "a"}).to_dict(),
                     DataCheckWarning(message="Column 'b' is 80.0% or more correlated with the target",
                                      data_check_name=target_leakage_data_check_name,
                                      message_code=DataCheckMessageCode.TARGET_LEAKAGE,
                                      details={"column": "b"}).to_dict(),
                     DataCheckWarning(message="Column 'c' is 80.0% or more correlated with the target",
                                      data_check_name=target_leakage_data_check_name,
                                      message_code=DataCheckMessageCode.TARGET_LEAKAGE,
                                      details={"column": "c"}).to_dict()],
        "errors": []
    }

    # test X as ww.DataTable, y as ww.DataColumn
    assert leakage_check.validate(ww.DataTable(X), ww.DataColumn(y)) == expected_messages

    #  test y as list
    assert leakage_check.validate(X, y.values) == expected_messages


def test_target_leakage_regression():
    leakage_check = TargetLeakageDataCheck(pct_corr_threshold=0.8)

    # test empty pd.DataFrame, empty pd.Series
    assert leakage_check.validate(pd.DataFrame(), pd.Series()) == {"warnings": [], "errors": []}

    y = pd.Series([0.4, 0.1, 2.3, 4.3, 2.2, 1.8, 3.7, 3.6, 2.4, 0.9, 3.1, 2.8, 4.1, 1.6, 1.2])
    X = pd.DataFrame()
    X["a"] = y * 3
    X["b"] = y - 1
    X["c"] = y / 10
    X["d"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    X["e"] = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o"]

    expected_messages = {
        "warnings": [DataCheckWarning(message="Column 'a' is 80.0% or more correlated with the target",
                                      data_check_name=target_leakage_data_check_name,
                                      message_code=DataCheckMessageCode.TARGET_LEAKAGE,
                                      details={"column": "a"}).to_dict(),
                     DataCheckWarning(message="Column 'b' is 80.0% or more correlated with the target",
                                      data_check_name=target_leakage_data_check_name,
                                      message_code=DataCheckMessageCode.TARGET_LEAKAGE,
                                      details={"column": "b"}).to_dict(),
                     DataCheckWarning(message="Column 'c' is 80.0% or more correlated with the target",
                                      data_check_name=target_leakage_data_check_name,
                                      message_code=DataCheckMessageCode.TARGET_LEAKAGE,
                                      details={"column": "c"}).to_dict(),
                     DataCheckWarning(message="Column 'e' is 80.0% or more correlated with the target",
                                      data_check_name=target_leakage_data_check_name,
                                      message_code=DataCheckMessageCode.TARGET_LEAKAGE,
                                      details={"column": "e"}).to_dict()],
        "errors": []
    }

    # test X as ww.DataTable, y as ww.DataColumn
    assert leakage_check.validate(ww.DataTable(X), ww.DataColumn(y)) == expected_messages

    #  test y as list
    assert leakage_check.validate(X, y.values) == expected_messages


def test_target_leakage_data_check_warnings_pearson():
    y = pd.Series([1, 0, 1, 1])
    X = pd.DataFrame()
    X["a"] = y * 3
    X["b"] = y - 1
    X["c"] = y / 10
    X["d"] = ~y
    X["e"] = [0, 0, 0, 0]
    y = y.astype(bool)

    leakage_check = TargetLeakageDataCheck(pct_corr_threshold=0.5, method='pearson')
    assert leakage_check.validate(X, y) == {
        "warnings": [DataCheckWarning(message="Column 'a' is 50.0% or more correlated with the target",
                                      data_check_name=target_leakage_data_check_name,
                                      message_code=DataCheckMessageCode.TARGET_LEAKAGE,
                                      details={"column": "a"}).to_dict(),
                     DataCheckWarning(message="Column 'b' is 50.0% or more correlated with the target",
                                      data_check_name=target_leakage_data_check_name,
                                      message_code=DataCheckMessageCode.TARGET_LEAKAGE,
                                      details={"column": "b"}).to_dict(),
                     DataCheckWarning(message="Column 'c' is 50.0% or more correlated with the target",
                                      data_check_name=target_leakage_data_check_name,
                                      message_code=DataCheckMessageCode.TARGET_LEAKAGE,
                                      details={"column": "c"}).to_dict(),
                     DataCheckWarning(message="Column 'd' is 50.0% or more correlated with the target",
                                      data_check_name=target_leakage_data_check_name,
                                      message_code=DataCheckMessageCode.TARGET_LEAKAGE,
                                      details={"column": "d"}).to_dict()],
        "errors": []
    }

    y = ["a", "b", "a", "a"]
    leakage_check = TargetLeakageDataCheck(pct_corr_threshold=0.5, method='pearson')
    assert leakage_check.validate(X, y) == {
        "warnings": [],
        "errors": []
    }


def test_target_leakage_data_check_input_formats_pearson():
    leakage_check = TargetLeakageDataCheck(pct_corr_threshold=0.8, method='pearson')

    # test empty pd.DataFrame, empty pd.Series
    assert leakage_check.validate(pd.DataFrame(), pd.Series()) == {"warnings": [], "errors": []}

    y = pd.Series([1, 0, 1, 1])
    X = pd.DataFrame()
    X["a"] = y * 3
    X["b"] = y - 1
    X["c"] = y / 10
    X["d"] = ~y
    X["e"] = [0, 0, 0, 0]
    y = y.astype(bool)

    expected_messages = {
        "warnings": [DataCheckWarning(message="Column 'a' is 80.0% or more correlated with the target",
                                      data_check_name=target_leakage_data_check_name,
                                      message_code=DataCheckMessageCode.TARGET_LEAKAGE,
                                      details={"column": "a"}).to_dict(),
                     DataCheckWarning(message="Column 'b' is 80.0% or more correlated with the target",
                                      data_check_name=target_leakage_data_check_name,
                                      message_code=DataCheckMessageCode.TARGET_LEAKAGE,
                                      details={"column": "b"}).to_dict(),
                     DataCheckWarning(message="Column 'c' is 80.0% or more correlated with the target",
                                      data_check_name=target_leakage_data_check_name,
                                      message_code=DataCheckMessageCode.TARGET_LEAKAGE,
                                      details={"column": "c"}).to_dict(),
                     DataCheckWarning(message="Column 'd' is 80.0% or more correlated with the target",
                                      data_check_name=target_leakage_data_check_name,
                                      message_code=DataCheckMessageCode.TARGET_LEAKAGE,
                                      details={"column": "d"}).to_dict()],
        "errors": []
    }

    # test X as np.array
    assert leakage_check.validate(X.values, y) == {
        "warnings": [DataCheckWarning(message="Column '0' is 80.0% or more correlated with the target",
                                      data_check_name=target_leakage_data_check_name,
                                      message_code=DataCheckMessageCode.TARGET_LEAKAGE,
                                      details={"column": 0}).to_dict(),
                     DataCheckWarning(message="Column '1' is 80.0% or more correlated with the target",
                                      data_check_name=target_leakage_data_check_name,
                                      message_code=DataCheckMessageCode.TARGET_LEAKAGE,
                                      details={"column": 1}).to_dict(),
                     DataCheckWarning(message="Column '2' is 80.0% or more correlated with the target",
                                      data_check_name=target_leakage_data_check_name,
                                      message_code=DataCheckMessageCode.TARGET_LEAKAGE,
                                      details={"column": 2}).to_dict(),
                     DataCheckWarning(message="Column '3' is 80.0% or more correlated with the target",
                                      data_check_name=target_leakage_data_check_name,
                                      message_code=DataCheckMessageCode.TARGET_LEAKAGE,
                                      details={"column": 3}).to_dict()],
        "errors": []
    }

    # test X as ww.DataTable, y as ww.DataColumn
    assert leakage_check.validate(ww.DataTable(X), ww.DataColumn(y)) == expected_messages

    #  test y as list
    assert leakage_check.validate(X, y.values) == expected_messages


def test_target_leakage_none_pearson():
    leakage_check = TargetLeakageDataCheck(pct_corr_threshold=0.8, method='pearson')
    y = pd.Series([1, 0, 1, 1])
    X = pd.DataFrame()
    X["a"] = [1, 1, 1, 1]
    X["b"] = [0, 0, 0, 0]
    y = y.astype(bool)

    expected_messages = {
        "warnings": [],
        "errors": []
    }

    assert leakage_check.validate(X, y) == expected_messages
