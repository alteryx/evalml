import re
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import woodwork as ww
from woodwork.config import CONFIG_DEFAULTS

from evalml.data_checks import (
    DataCheckActionCode,
    DataCheckActionOption,
    DataCheckMessageCode,
    DataCheckWarning,
    TargetLeakageDataCheck,
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

    with pytest.raises(
        ValueError,
        match="pct_corr_threshold must be a float between 0 and 1, inclusive.",
    ):
        TargetLeakageDataCheck(pct_corr_threshold=-0.1)
    with pytest.raises(
        ValueError,
        match="pct_corr_threshold must be a float between 0 and 1, inclusive.",
    ):
        TargetLeakageDataCheck(pct_corr_threshold=1.1)

    with pytest.raises(ValueError, match="Method 'MUTUAL' not in"):
        TargetLeakageDataCheck(method="MUTUAL")
    with pytest.raises(ValueError, match="Method 'person' not in"):
        TargetLeakageDataCheck(method="person")


def test_target_leakage_data_check_warnings():
    y = pd.Series(range(30))
    X = pd.DataFrame()
    X["a"] = y * 3
    X["b"] = y - 1
    X["c"] = y / 10
    X["d"] = y % 2
    X["e"] = [0] * 30
    X.ww.init(logical_types={"d": "Boolean"})

    leakage_check = TargetLeakageDataCheck(pct_corr_threshold=0.5)
    assert leakage_check.validate(X, y) == [
        DataCheckWarning(
            message="Columns 'a', 'b', 'c' are 50.0% or more correlated with the target",
            data_check_name=target_leakage_data_check_name,
            message_code=DataCheckMessageCode.TARGET_LEAKAGE,
            details={"columns": ["a", "b", "c"]},
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_COL,
                    data_check_name=target_leakage_data_check_name,
                    metadata={"columns": ["a", "b", "c"]},
                ),
            ],
        ).to_dict(),
    ]


def test_target_leakage_data_check_singular_warning():
    y = pd.Series(range(30))
    X = pd.DataFrame()
    X["a"] = y * 3
    X["b"] = [0] * 30

    leakage_check = TargetLeakageDataCheck(pct_corr_threshold=0.5)
    assert leakage_check.validate(X, y) == [
        DataCheckWarning(
            message="Column 'a' is 50.0% or more correlated with the target",
            data_check_name=target_leakage_data_check_name,
            message_code=DataCheckMessageCode.TARGET_LEAKAGE,
            details={"columns": ["a"]},
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_COL,
                    data_check_name=target_leakage_data_check_name,
                    metadata={"columns": ["a"]},
                ),
            ],
        ).to_dict(),
    ]


@pytest.mark.parametrize("data_type", ["np", "pd", "ww"])
def test_target_leakage_data_check_empty(data_type, make_data_type):
    X = make_data_type(data_type, pd.DataFrame())
    y = make_data_type(data_type, pd.Series())
    leakage_check = TargetLeakageDataCheck(pct_corr_threshold=0.8, method="mutual_info")
    assert leakage_check.validate(X, y) == []


def test_target_leakage_data_check_input_formats():
    leakage_check = TargetLeakageDataCheck(pct_corr_threshold=0.8)
    y = pd.Series(range(30))
    X = pd.DataFrame()
    X["a"] = y * 3
    X["b"] = y - 1
    X["c"] = y / 10
    X["d"] = y % 2
    X["e"] = [0] * 30
    X.ww.init(logical_types={"d": "Boolean"})

    expected = [
        DataCheckWarning(
            message="Columns 'a', 'b', 'c' are 80.0% or more correlated with the target",
            data_check_name=target_leakage_data_check_name,
            message_code=DataCheckMessageCode.TARGET_LEAKAGE,
            details={"columns": ["a", "b", "c"]},
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_COL,
                    data_check_name=target_leakage_data_check_name,
                    metadata={"columns": ["a", "b", "c"]},
                ),
            ],
        ).to_dict(),
    ]
    # test X, y with ww
    X_ww = X.copy()
    X_ww.ww.init()
    y_ww = ww.init_series(y)
    assert leakage_check.validate(X_ww, y_ww) == expected

    # test y as list
    assert leakage_check.validate(X, y.values) == expected

    # test X as np.array
    assert leakage_check.validate(X.to_numpy().astype(float), y) == [
        DataCheckWarning(
            message="Columns '0', '1', '2' are 80.0% or more correlated with the target",
            data_check_name=target_leakage_data_check_name,
            message_code=DataCheckMessageCode.TARGET_LEAKAGE,
            details={"columns": [0, 1, 2]},
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_COL,
                    data_check_name=target_leakage_data_check_name,
                    metadata={"columns": [0, 1, 2]},
                ),
            ],
        ).to_dict(),
    ]


def test_target_leakage_none():
    leakage_check = TargetLeakageDataCheck(pct_corr_threshold=0.8)
    y = pd.Series([1, 0, 1, 1])
    X = pd.DataFrame()
    X["a"] = [1, 1, 1, 1]
    X["b"] = [0, 0, 0, 0]
    y = y.astype(bool)

    assert leakage_check.validate(X, y) == []


def test_target_leakage_types():
    leakage_check = TargetLeakageDataCheck(pct_corr_threshold=0.8)

    y = pd.Series([1, 0, 1, 1] * 6 + [1])
    X = pd.DataFrame()
    X["a"] = ["a", "b", "a", "a"] * 6 + ["a"]
    X["b"] = y

    X["c"] = [
        datetime.strptime("2015", "%Y"),
        datetime.strptime("2016", "%Y"),
        datetime.strptime("2015", "%Y"),
        datetime.strptime("2015", "%Y"),
    ] * 6 + [datetime.strptime("2015", "%Y")]
    X["d"] = ~y
    X["e"] = np.zeros(len(y))
    y = y.astype(bool)
    X.ww.init(logical_types={"a": "categorical", "d": "Boolean", "b": "Boolean"})

    expected = [
        DataCheckWarning(
            message="Columns 'a', 'b' are 80.0% or more correlated with the target",
            data_check_name=target_leakage_data_check_name,
            message_code=DataCheckMessageCode.TARGET_LEAKAGE,
            details={"columns": ["a", "b"]},
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_COL,
                    data_check_name=target_leakage_data_check_name,
                    metadata={"columns": ["a", "b"]},
                ),
            ],
        ).to_dict(),
    ]

    assert leakage_check.validate(X, y) == expected


def test_target_leakage_multi():
    leakage_check = TargetLeakageDataCheck(pct_corr_threshold=0.8)

    # test empty pd.DataFrame, empty pd.Series
    assert leakage_check.validate(pd.DataFrame(), pd.Series()) == []

    y = pd.Series(
        [1, 0, 2, 1, 2, 0, 1, 0, 2, 1, 2, 0, 1, 0, 2, 1, 2, 0, 1, 0, 2, 1, 2, 0, 1],
    )
    X = pd.DataFrame()
    X["a"] = y * 3
    X["b"] = y - 1
    X["c"] = y / 10
    X["d"] = np.zeros(len(y))
    X["e"] = [
        "a",
        "b",
        "c",
        "a",
        "b",
        "c",
        "a",
        "b",
        "c",
        "a",
        "b",
        "c",
        "a",
        "b",
        "c",
        "a",
        "b",
        "c",
        "a",
        "b",
        "c",
        "a",
        "b",
        "c",
        "a",
    ]

    expected = [
        DataCheckWarning(
            message="Columns 'a', 'b', 'c' are 80.0% or more correlated with the target",
            data_check_name=target_leakage_data_check_name,
            message_code=DataCheckMessageCode.TARGET_LEAKAGE,
            details={"columns": ["a", "b", "c"]},
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_COL,
                    data_check_name=target_leakage_data_check_name,
                    metadata={"columns": ["a", "b", "c"]},
                ),
            ],
        ).to_dict(),
    ]

    # test X, y with ww
    X_ww = X.copy()
    X_ww.ww.init()  # logical_types={"b": "Categorical", "c": "Categorical"})
    y_ww = ww.init_series(y)  # , logical_type="Categorical")
    assert leakage_check.validate(X_ww, y_ww) == expected

    #  test y as list
    assert leakage_check.validate(X, y.values) == expected


def test_target_leakage_regression():
    leakage_check = TargetLeakageDataCheck(pct_corr_threshold=0.8)

    # test empty pd.DataFrame, empty pd.Series
    assert leakage_check.validate(pd.DataFrame(), pd.Series()) == []

    y = pd.Series(
        [
            0.7,
            0.8,
            1.6,
            2.8,
            4.0,
            5.1,
            5.2,
            5.9,
            6.1,
            6.4,
            8.7,
            9.5,
            13.2,
            13.7,
            14.8,
            15.2,
            15.8,
            19.0,
            19.2,
            19.6,
            20.0,
            20.6,
            21.1,
            23.3,
            25.3,
            25.8,
        ],
    )
    X = pd.DataFrame()
    X["a"] = y * 3
    X["b"] = y - 1
    X["c"] = y / 10
    X["d"] = np.zeros(len(y))
    X.ww.init()

    expected = [
        DataCheckWarning(
            message="Columns 'a', 'b', 'c' are 80.0% or more correlated with the target",
            data_check_name=target_leakage_data_check_name,
            message_code=DataCheckMessageCode.TARGET_LEAKAGE,
            details={"columns": ["a", "b", "c"]},
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_COL,
                    data_check_name=target_leakage_data_check_name,
                    metadata={"columns": ["a", "b", "c"]},
                ),
            ],
        ).to_dict(),
    ]

    # test X, y with ww
    y_ww = ww.init_series(y)
    assert leakage_check.validate(X, y_ww) == expected

    #  test y as list
    assert leakage_check.validate(X, y.values) == expected


def test_target_leakage_data_check_warnings_pearson():
    y = pd.Series([1, 0, 1, 1] * 10)
    X = pd.DataFrame()
    X["a"] = y * 3
    X["b"] = y - 1
    X["c"] = y / 10
    X["d"] = ~y
    X["e"] = [0, 0, 0, 0] * 10
    y = y.astype(bool)

    leakage_check = TargetLeakageDataCheck(pct_corr_threshold=0.5, method="pearson")
    # pearsons does not support boolean columns
    assert leakage_check.validate(X, y) == []

    y = y.astype(int)
    assert leakage_check.validate(X, y) == [
        DataCheckWarning(
            message="Columns 'a', 'b', 'c', 'd' are 50.0% or more correlated with the target",
            data_check_name=target_leakage_data_check_name,
            message_code=DataCheckMessageCode.TARGET_LEAKAGE,
            details={"columns": ["a", "b", "c", "d"]},
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_COL,
                    data_check_name=target_leakage_data_check_name,
                    metadata={"columns": ["a", "b", "c", "d"]},
                ),
            ],
        ).to_dict(),
    ]

    y = ["a", "b", "a", "a"] * 10
    leakage_check = TargetLeakageDataCheck(pct_corr_threshold=0.5, method="pearson")
    assert leakage_check.validate(X, y) == []


def test_target_leakage_data_check_input_formats_pearson():
    leakage_check = TargetLeakageDataCheck(pct_corr_threshold=0.8, method="pearson")

    # test empty pd.DataFrame, empty pd.Series
    assert leakage_check.validate(pd.DataFrame(), pd.Series()) == []

    y = pd.Series([1, 0, 1, 1] * 10)
    X = pd.DataFrame()
    X["a"] = y * 3
    X["b"] = y - 1
    X["c"] = y / 10
    X["d"] = ~y
    X["e"] = [0, 0, 0, 0] * 10

    expected = [
        DataCheckWarning(
            message="Columns 'a', 'b', 'c', 'd' are 80.0% or more correlated with the target",
            data_check_name=target_leakage_data_check_name,
            message_code=DataCheckMessageCode.TARGET_LEAKAGE,
            details={"columns": ["a", "b", "c", "d"]},
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_COL,
                    data_check_name=target_leakage_data_check_name,
                    metadata={"columns": ["a", "b", "c", "d"]},
                ),
            ],
        ).to_dict(),
    ]

    # test X as np.array
    assert leakage_check.validate(X.values, y) == [
        DataCheckWarning(
            message="Columns '0', '1', '2', '3' are 80.0% or more correlated with the target",
            data_check_name=target_leakage_data_check_name,
            message_code=DataCheckMessageCode.TARGET_LEAKAGE,
            details={"columns": [0, 1, 2, 3]},
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_COL,
                    data_check_name=target_leakage_data_check_name,
                    metadata={"columns": [0, 1, 2, 3]},
                ),
            ],
        ).to_dict(),
    ]

    # test X, y with ww
    X_ww = X.copy()
    X_ww.ww.init()
    y_ww = ww.init_series(y)
    assert leakage_check.validate(X_ww, y_ww) == expected

    #  test y as list
    assert leakage_check.validate(X, y.values) == expected


@pytest.mark.parametrize("measures", ["pearson", "spearman", "mutual_info", "max"])
def test_target_leakage_none_measures(measures):
    leakage_check = TargetLeakageDataCheck(pct_corr_threshold=0.5, method=measures)
    y = pd.Series([1, 0, 1, 1] * 6 + [1])
    X = pd.DataFrame()
    X["a"] = ["a", "b", "a", "a"] * 6 + ["a"]
    X["b"] = y
    y = y.astype(bool)

    if measures in ["pearson", "spearman"]:
        assert leakage_check.validate(X, y) == []
        return
    assert len(leakage_check.validate(X, y))


def test_target_leakage_maintains_logical_types():
    X = pd.DataFrame({"A": pd.Series(range(1, 26)), "B": pd.Series(range(26, 51))})
    y = pd.Series(range(1, 26))

    X.ww.init(logical_types={"A": "Unknown", "B": "Double"})
    messages = TargetLeakageDataCheck().validate(X, y)

    # Mutual information is not supported for Unknown logical types, so should not be included
    assert not any(message["message"].startswith("Column 'A'") for message in messages)
    assert len(messages) == 1


def test_target_leakage_methods():
    methods = CONFIG_DEFAULTS["correlation_metrics"]

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Method '{}' not in available correlation methods. Available methods include {}".format(
                "fake_method",
                methods,
            ),
        ),
    ):
        TargetLeakageDataCheck(method="fake_method")


def test_target_leakage_target_string():
    y = pd.Series([1, 0, 1, 1] * 10)
    X = pd.DataFrame()
    X["target_y"] = y * 3
    X["target_y_y"] = y - 1
    X["target"] = y / 10
    X["d"] = ~y
    X["e"] = [0, 1, 2, 3] * 10
    leakage_check = TargetLeakageDataCheck(pct_corr_threshold=0.8)

    expected = [
        DataCheckWarning(
            message="Columns 'target_y', 'target_y_y', 'target', 'd' are 80.0% or more correlated with the target",
            data_check_name=target_leakage_data_check_name,
            message_code=DataCheckMessageCode.TARGET_LEAKAGE,
            details={"columns": ["target_y", "target_y_y", "target", "d"]},
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_COL,
                    data_check_name=target_leakage_data_check_name,
                    metadata={"columns": ["target_y", "target_y_y", "target", "d"]},
                ),
            ],
        ).to_dict(),
    ]
    assert leakage_check.validate(X, y) == expected


def test_target_leakage_use_all():
    with pytest.raises(ValueError, match="Cannot use 'all' as the method"):
        TargetLeakageDataCheck(method="all")
