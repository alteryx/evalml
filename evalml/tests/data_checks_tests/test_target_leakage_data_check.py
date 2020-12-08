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


def test_target_leakage_data_check_input_formats():
    leakage_check = TargetLeakageDataCheck(pct_corr_threshold=0.8)

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

    # test X as ww.DataTable, y as ww.DataColumn
    assert leakage_check.validate(ww.DataTable(X), ww.DataColumn(y)) == expected_messages

    #  test y as list
    assert leakage_check.validate(X, y.values) == expected_messages

    # test X as np.array
    assert leakage_check.validate(X.to_numpy(), y) == {
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
