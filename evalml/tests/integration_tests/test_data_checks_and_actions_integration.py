from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import woodwork as ww
from pandas.testing import assert_frame_equal, assert_series_equal

from evalml.automl import get_default_primary_search_objective
from evalml.data_checks import (
    DateTimeFormatDataCheck,
    DefaultDataChecks,
    OutliersDataCheck,
)
from evalml.data_checks.invalid_target_data_check import InvalidTargetDataCheck
from evalml.data_checks.null_data_check import NullDataCheck
from evalml.pipelines import (
    BinaryClassificationPipeline,
    TimeSeriesImputer,
    TimeSeriesRegressionPipeline,
    TimeSeriesRegularizer,
)
from evalml.pipelines.components import DropColumns, DropRowsTransformer, TargetImputer
from evalml.pipelines.components.transformers.imputers.per_column_imputer import (
    PerColumnImputer,
)
from evalml.pipelines.multiclass_classification_pipeline import (
    MulticlassClassificationPipeline,
)
from evalml.pipelines.regression_pipeline import RegressionPipeline
from evalml.pipelines.utils import make_pipeline_from_data_check_output


def test_data_checks_with_healthy_data(X_y_binary):
    # Checks do not return any error.
    X, y = X_y_binary
    data_check = DefaultDataChecks(
        "binary",
        get_default_primary_search_objective("binary"),
    )
    data_checks_output = data_check.validate(X, y)

    assert make_pipeline_from_data_check_output(
        "binary",
        data_checks_output,
    ) == BinaryClassificationPipeline(component_graph={}, parameters={}, random_seed=0)


@patch("evalml.pipelines.utils.make_pipeline_from_actions")
def test_make_pipeline_no_problem_configuration(mocked_make_pipeline_from_actions):
    with pytest.raises(
        ValueError,
        match="problem_configuration must be a dict containing values",
    ):
        make_pipeline_from_data_check_output(
            problem_type="time series regression",
            data_check_output={},
            problem_configuration=None,
        )


def test_data_checks_ts_regularizer_imputer(uneven_continuous):
    X = pd.DataFrame(
        {
            "Dates": uneven_continuous,
            "First_Column": [1, 3, 1, 5, 1, 7] * 23,
            "Second_Column": ["First", "Second", "First", "Third", "First", "Fourth"]
            * 23,
        },
    )
    y = pd.Series([i for i in range(len(uneven_continuous))])

    problem_config = {
        "gap": 0,
        "max_delay": 0,
        "forecast_horizon": 0,
        "time_index": "Dates",
    }
    data_check = DateTimeFormatDataCheck(datetime_column="Dates")
    data_checks_output = data_check.validate(X, y)

    ww_payload = None
    for output in data_checks_output:
        if output["code"] == "DATETIME_HAS_UNEVEN_INTERVALS":
            ww_payload = output["action_options"][0]["parameters"]["frequency_payload"][
                "default_value"
            ]
            break

    action_pipeline = make_pipeline_from_data_check_output(
        "time series regression",
        data_checks_output,
        problem_config,
    )

    assert action_pipeline == TimeSeriesRegressionPipeline(
        component_graph={
            "Time Series Regularizer": [TimeSeriesRegularizer, "X", "y"],
            "Time Series Imputer": [
                TimeSeriesImputer,
                "Time Series Regularizer.x",
                "Time Series Regularizer.y",
            ],
        },
        parameters={
            "Time Series Regularizer": {
                "time_index": "Dates",
                "frequency_payload": ww_payload,
                "window_length": 4,
                "threshold": 0.4,
            },
            "Time Series Imputer": {
                "categorical_impute_strategy": "forwards_fill",
                "numeric_impute_strategy": "interpolate",
                "target_impute_strategy": "forwards_fill",
            },
            "pipeline": {
                "gap": 0,
                "max_delay": 0,
                "forecast_horizon": 0,
                "time_index": "Dates",
            },
        },
        random_seed=0,
    )
    X_expected = pd.DataFrame(
        {
            "Dates": pd.date_range("2014-01-01", periods=137, freq="2D"),
            "First_Column": [1, 3, 1, 5, 1, 7, 1]
            + [1, 5, 1, 7, 1, 3] * 21
            + [1, 5, 1, 7],
            "Second_Column": [
                "First",
                "Second",
                "First",
                "Third",
                "First",
                "Fourth",
                "First",
            ]
            + ["First", "Third", "First", "Fourth", "First", "Second"] * 21
            + ["First", "Third", "First", "Fourth"],
        },
    )
    X_expected.ww.init(
        logical_types={"First_Column": "double", "Second_Column": "categorical"},
    )
    y_expected = pd.Series(
        [0, 1, 2, 3, 4, 5, 6] + [i for i in range(8, 138)],
        name="target",
    )
    action_pipeline.fit(X, y)
    X_t, y_t = action_pipeline.transform(X, y)
    assert_frame_equal(X_expected, X_t)
    np.testing.assert_array_equal(y_expected.values, y_t.values)


def test_data_checks_suggests_drop_and_impute_cols():
    X = pd.DataFrame(
        {
            "null_with_categorical": ["a", None, "b", "c", "c"],
            "lots_of_null": [None, 7, None, 3, 5],
            "all_null": [None, None, None, None, None],
            "no_null": [1, 2, 3, 4, 5],
        },
    )
    X.ww.init(logical_types={"null_with_categorical": "categorical"})
    y = pd.Series([1, 0, 0, 1, 1])
    data_check = NullDataCheck()
    data_checks_output = data_check.validate(X, y)

    action_pipeline = make_pipeline_from_data_check_output("binary", data_checks_output)
    assert action_pipeline == BinaryClassificationPipeline(
        component_graph={
            "Per Column Imputer": [PerColumnImputer, "X", "y"],
            "Drop Columns Transformer": [
                DropColumns,
                "Per Column Imputer.x",
                "y",
            ],
        },
        parameters={
            "Per Column Imputer": {
                "impute_strategies": {
                    "null_with_categorical": {"impute_strategy": "most_frequent"},
                    "lots_of_null": {"impute_strategy": "mean"},
                },
                "default_impute_strategy": "most_frequent",
            },
            "Drop Columns Transformer": {"columns": ["all_null"]},
        },
        random_seed=0,
    )
    X_expected = pd.DataFrame(
        {
            "null_with_categorical": ["a", "c", "b", "c", "c"],
            "lots_of_null": [5, 7, 5, 3, 5],
            "no_null": [1, 2, 3, 4, 5],
        },
    )
    X_expected.ww.init(
        logical_types={
            "lots_of_null": "IntegerNullable",
            "null_with_categorical": "categorical",
        },
    )
    action_pipeline.fit(X, y)
    X_t = action_pipeline.transform(X, y)
    assert_frame_equal(X_expected, X_t)


@pytest.mark.parametrize("problem_type", ["binary", "multiclass", "regression"])
def test_data_checks_impute_cols(problem_type):
    X = pd.DataFrame()
    if problem_type == "binary":
        y = ww.init_series(pd.Series([0, 1, 1, None, None]))
        objective = "Log Loss Binary"
        expected_pipeline_class = BinaryClassificationPipeline
        y_expected = ww.init_series(
            pd.Series([0, 1, 1, 1, 1]),
            logical_type="Integer",
        )

    elif problem_type == "multiclass":
        y = ww.init_series(pd.Series([0, 1, 2, 2, None]))
        objective = "Log Loss Multiclass"
        expected_pipeline_class = MulticlassClassificationPipeline
        y_expected = ww.init_series(
            pd.Series([0, 1, 2, 2, 2]),
            logical_type="Integer",
        )

    else:
        y = ww.init_series(pd.Series([0, 0.1, 0.2, None, None]))
        objective = "R2"
        expected_pipeline_class = RegressionPipeline
        y_expected = ww.init_series(
            pd.Series([0, 0.1, 0.2, 0.1, 0.1]),
            logical_type="double",
        )
    data_check = InvalidTargetDataCheck(problem_type, objective)
    data_checks_output = data_check.validate(None, y)

    action_pipeline = make_pipeline_from_data_check_output(
        problem_type,
        data_checks_output,
    )
    expected_parameters = (
        {"Target Imputer": {"impute_strategy": "mean", "fill_value": None}}
        if problem_type == "regression"
        else {
            "Target Imputer": {"impute_strategy": "most_frequent", "fill_value": None},
        }
    )
    assert action_pipeline == expected_pipeline_class(
        component_graph={"Target Imputer": [TargetImputer, "X", "y"]},
        parameters=expected_parameters,
        random_seed=0,
    )

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

    action_pipeline = make_pipeline_from_data_check_output("binary", data_checks_output)
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
