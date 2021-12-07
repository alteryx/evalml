import numpy as np
import pandas as pd
import woodwork as ww
from pandas.testing import assert_frame_equal, assert_series_equal

from evalml.automl import get_default_primary_search_objective
from evalml.data_checks import (
    DataCheckAction,
    DefaultDataChecks,
    OutliersDataCheck,
)
from evalml.data_checks.highly_null_data_check import HighlyNullDataCheck
from evalml.data_checks.invalid_target_data_check import InvalidTargetDataCheck
from evalml.pipelines import BinaryClassificationPipeline
from evalml.pipelines.components import (
    DropColumns,
    DropRowsTransformer,
    TargetImputer,
)
from evalml.pipelines.utils import make_pipeline_from_actions


def test_data_checks_with_healthy_data(X_y_binary):
    # Checks do not return any error.
    X, y = X_y_binary
    data_check = DefaultDataChecks(
        "binary", get_default_primary_search_objective("binary")
    )
    data_check_output = data_check.validate(X, y)
    assert make_pipeline_from_actions(
        "binary", data_check_output["actions"]
    ) == BinaryClassificationPipeline(component_graph={}, parameters={}, random_seed=0)


def test_data_checks_suggests_drop_cols():
    X = pd.DataFrame(
        {
            "lots_of_null": [None, 2, None, 3, 5],
            "all_null": [None, None, None, None, None],
            "no_null": [1, 2, 3, 4, 5],
        }
    )
    y = pd.Series([1, 0, 0, 1, 1])
    data_check = HighlyNullDataCheck()
    data_checks_output = data_check.validate(X, y)

    actions = [
        DataCheckAction.convert_dict_to_action(action)
        for action in data_checks_output["actions"]
    ]
    action_pipeline = make_pipeline_from_actions("binary", actions)
    assert action_pipeline == BinaryClassificationPipeline(
        component_graph={"Drop Columns Transformer": [DropColumns, "X", "y"]},
        parameters={"Drop Columns Transformer": {"columns": ["all_null"]}},
        random_seed=0,
    )

    X_t = pd.DataFrame(
        {
            "lots_of_null": [None, 2, None, 3, 5],
            "all_null": [None, None, None, None, None],
            "no_null": [1, 2, 3, 4, 5],
        }
    )
    X_expected = pd.DataFrame(
        {
            "lots_of_null": [None, 2, None, 3, 5],
            "no_null": [1, 2, 3, 4, 5],
        }
    )

    action_pipeline.fit(X, y)
    X_t = action_pipeline.transform(X, y)
    assert_frame_equal(X_expected, X_t)


def test_data_checks_impute_cols():
    X = pd.DataFrame()
    y = ww.init_series(pd.Series([0, 1, 1, None, None]))

    data_check = InvalidTargetDataCheck("binary", "Log Loss Binary")
    data_checks_output = data_check.validate(None, y)

    actions = [
        DataCheckAction.convert_dict_to_action(action)
        for action in data_checks_output["actions"]
    ]
    action_pipeline = make_pipeline_from_actions("binary", actions)
    assert action_pipeline == BinaryClassificationPipeline(
        component_graph={"Target Imputer": [TargetImputer, "X", "y"]},
        parameters={
            "Target Imputer": {"impute_strategy": "most_frequent", "fill_value": None}
        },
        random_seed=0,
    )

    y_expected = ww.init_series(pd.Series([0, 1, 1, 1, 1]), logical_type="double")
    y_t = ww.init_series(pd.Series([0, 1, 1, None, None]))
    action_pipeline.fit(X, y)
    _, y_t = action_pipeline.transform(X, y)
    assert_series_equal(y_expected, y_t)


def test_data_checks_suggests_drop_rows():
    a = np.arange(10) * 0.01
    data = np.tile(a, (100, 10))

    X = pd.DataFrame(data=data)
    X.iloc[0, 3] = 1000
    X.iloc[3, 25] = 1000
    X.iloc[5, 55] = 10000
    X.iloc[10, 72] = -1000
    X.iloc[:, 90] = "string_values"
    y = pd.Series(np.tile([0, 1], 50))

    outliers_check = OutliersDataCheck()
    data_checks_output = outliers_check.validate(X)

    actions = [
        DataCheckAction.convert_dict_to_action(action)
        for action in data_checks_output["actions"]
    ]
    action_pipeline = make_pipeline_from_actions("binary", actions)
    assert action_pipeline == BinaryClassificationPipeline(
        component_graph={"Drop Rows Transformer": [DropRowsTransformer, "X", "y"]},
        parameters={"Drop Rows Transformer": {"indices_to_drop": [0, 3, 5, 10]}},
        random_seed=0,
    )

    X_expected = X.drop([0, 3, 5, 10])
    X_expected.ww.init()
    y_expected = y.drop([0, 3, 5, 10])

    action_pipeline.fit(X, y)
    X_t, y_t = action_pipeline.transform(X, y)
    assert_frame_equal(X_expected, X_t)
    assert_series_equal(y_expected, y_t)
