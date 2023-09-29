from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_series_equal

from evalml.pipelines import MultiseriesRegressionPipeline
from evalml.pipelines.utils import unstack_multiseries
from evalml.preprocessing import split_multiseries_data


@pytest.fixture(scope="module")
def component_graph():
    return {
        "Time Series Featurizer": ["Time Series Featurizer", "X", "y"],
        "Baseline Multiseries": [
            "Multiseries Time Series Baseline Regressor",
            "Time Series Featurizer.x",
            "y",
        ],
    }


@pytest.fixture(scope="module")
def pipeline_parameters():
    return {
        "pipeline": {
            "time_index": "date",
            "max_delay": 10,
            "forecast_horizon": 7,
            "gap": 0,
            "series_id": "series_id",
        },
        "Time Series Featurizer": {
            "time_index": "date",
            "max_delay": 10,
            "forecast_horizon": 7,
            "gap": 0,
            "delay_features": False,
            "delay_target": True,
        },
        "Baseline Multiseries": {"gap": 0, "forecast_horizon": 7},
    }


def test_multiseries_pipeline_init(component_graph, pipeline_parameters):
    pipeline = MultiseriesRegressionPipeline(component_graph, pipeline_parameters)
    assert pipeline.series_id == "series_id"
    assert pipeline.time_index == "date"


def test_multiseries_pipeline_invalid_params(component_graph):
    with pytest.raises(
        ValueError,
        match="time_index, gap, max_delay, and forecast_horizon parameters cannot be omitted from the parameters dict",
    ):
        MultiseriesRegressionPipeline(
            component_graph,
            {"Baseline Multiseries": {"gap": 10, "forecast_horizon": 7}},
        )

    with pytest.raises(
        ValueError,
        match="series_id must be defined for multiseries time series pipelines",
    ):
        MultiseriesRegressionPipeline(
            component_graph,
            {
                "pipeline": {
                    "time_index": "date",
                    "max_delay": 10,
                    "forecast_horizon": 7,
                    "gap": 0,
                },
            },
        )


def test_multiseries_pipeline_fit(
    multiseries_ts_data_stacked,
    component_graph,
    pipeline_parameters,
):
    X, y = multiseries_ts_data_stacked
    pipeline = MultiseriesRegressionPipeline(component_graph, pipeline_parameters)

    pipeline.fit(X, y)
    assert pipeline._is_fitted
    assert pipeline.frequency is not None


def test_multiseries_pipeline_predict_in_sample(
    multiseries_ts_data_stacked,
    component_graph,
    pipeline_parameters,
):
    X, y = multiseries_ts_data_stacked
    X_train, X_holdout, y_train, y_holdout = split_multiseries_data(
        X,
        y,
        "series_id",
        "date",
    )

    pipeline = MultiseriesRegressionPipeline(component_graph, pipeline_parameters)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict_in_sample(
        X_holdout,
        y_holdout,
        X_train=X_train,
        y_train=y_train,
    )
    expected = pd.Series(
        range(55, 65),
        index=range(90, 100),
        name="target",
        dtype="float64",
    )
    pd.testing.assert_series_equal(y_pred, expected)


@pytest.mark.parametrize("forecast_horizon", [1, 7])
def test_multiseries_pipeline_predict(
    forecast_horizon,
    multiseries_ts_data_stacked,
    component_graph,
    pipeline_parameters,
):
    X, y = multiseries_ts_data_stacked
    X_train, X_holdout, y_train, y_holdout = split_multiseries_data(
        X,
        y,
        "series_id",
        "date",
    )

    pipeline_parameters["pipeline"]["forecast_horizon"] = forecast_horizon
    pipeline_parameters["Time Series Featurizer"]["forecast_horizon"] = forecast_horizon
    pipeline_parameters["Baseline Multiseries"]["forecast_horizon"] = forecast_horizon

    pipeline = MultiseriesRegressionPipeline(component_graph, pipeline_parameters)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_holdout, y_holdout, X_train=X_train, y_train=y_train)

    # All predicted values are present in the delayed features, should match predict_in_sample
    if forecast_horizon == 7:
        expected = pd.Series(
            range(55, 65),
            index=range(90, 100),
            name="target",
            dtype="float64",
        )
    # Only the first predicted value is present in the delayed features
    else:
        expected = pd.Series(
            [85, 86, 87, 88, 89, 0, 0, 0, 0, 0],
            index=range(90, 100),
            name="target",
            dtype="float64",
        )
    pd.testing.assert_series_equal(y_pred, expected)


@pytest.mark.parametrize("forecast_horizon,gap", [[3, 0], [5, 2], [2, 5]])
@pytest.mark.parametrize("numeric_idx", [True, False])
def test_time_series_get_forecast_period(
    forecast_horizon,
    gap,
    numeric_idx,
    multiseries_ts_data_stacked,
    component_graph,
    pipeline_parameters,
):
    X, y = multiseries_ts_data_stacked
    if numeric_idx:
        X = X.reset_index(drop=True)

    pipeline_parameters["pipeline"]["forecast_horizon"] = forecast_horizon
    pipeline_parameters["Time Series Featurizer"]["forecast_horizon"] = forecast_horizon
    pipeline_parameters["Baseline Multiseries"]["forecast_horizon"] = forecast_horizon
    pipeline_parameters["pipeline"]["gap"] = gap
    pipeline_parameters["Time Series Featurizer"]["gap"] = gap
    pipeline_parameters["Baseline Multiseries"]["gap"] = gap

    clf = MultiseriesRegressionPipeline(component_graph, pipeline_parameters)

    with pytest.raises(
        ValueError,
        match="Pipeline must be fitted before getting forecast.",
    ):
        clf.get_forecast_period(X)

    clf.fit(X, y)
    result = clf.get_forecast_period(X)

    len_unique_series_id = len(X["series_id"].unique())

    assert result.shape[0] == forecast_horizon * len_unique_series_id
    assert all(
        result.index
        == range(
            len(X) + (gap * len_unique_series_id),
            len(X)
            + (gap * len_unique_series_id)
            + (forecast_horizon * len_unique_series_id),
        ),
    )
    assert result.iloc[0]["date"] == X.iloc[-1]["date"] + np.timedelta64(
        1 + gap,
        clf.frequency,
    )
    assert np.issubdtype(result.dtypes["date"], np.datetime64)
    assert list(result.columns) == ["date", "series_id"]


@pytest.mark.parametrize("forecast_horizon,gap", [[3, 0], [5, 2], [2, 5]])
def test_time_series_get_forecast_predictions(
    forecast_horizon,
    gap,
    multiseries_ts_data_stacked,
    component_graph,
    pipeline_parameters,
):
    X, y = multiseries_ts_data_stacked

    X_train, y_train = X.iloc[:25], y.iloc[:25]
    X_validation = X.iloc[25 + (gap * 5) : 25 + (gap * 5) + (forecast_horizon * 5)]

    pipeline_parameters["pipeline"]["forecast_horizon"] = forecast_horizon
    pipeline_parameters["Time Series Featurizer"]["forecast_horizon"] = forecast_horizon
    pipeline_parameters["Baseline Multiseries"]["forecast_horizon"] = forecast_horizon
    pipeline_parameters["pipeline"]["gap"] = gap
    pipeline_parameters["Time Series Featurizer"]["gap"] = gap
    pipeline_parameters["Baseline Multiseries"]["gap"] = gap

    clf = MultiseriesRegressionPipeline(component_graph, pipeline_parameters)

    clf.fit(X_train, y_train)
    forecast_preds = clf.get_forecast_predictions(X=X_train, y=y_train)
    X_val_preds = clf.predict(X_validation, X_train=X_train, y_train=y_train)
    assert_series_equal(forecast_preds, X_val_preds)


@pytest.mark.parametrize("set_coverage", [True, False])
@pytest.mark.parametrize("add_decomposer", [True, False])
@pytest.mark.parametrize("ts_native_estimator", [True, False])
def test_time_series_pipeline_get_prediction_intervals(
    ts_native_estimator,
    add_decomposer,
    set_coverage,
    multiseries_ts_data_stacked,
):
    X, y = multiseries_ts_data_stacked
    y = pd.Series(np.random.rand(100), name="target")
    component_graph = {
        "Regressor": [
            "VARMAX Regressor" if ts_native_estimator else "VARMAX Regressor",
            "X" if not add_decomposer else "STL Decomposer.x",
            "y" if not add_decomposer else "STL Decomposer.y",
        ],
    }
    if add_decomposer:
        component_graph.update(
            {
                "STL Decomposer": [
                    "STL Decomposer",
                    "X",
                    "y",
                ],
            },
        )

    pipeline_parameters = {
        "pipeline": {
            "time_index": "date",
            "max_delay": 10,
            "forecast_horizon": 7,
            "gap": 0,
            "series_id": "series_id",
        },
    }

    pipeline = MultiseriesRegressionPipeline(
        component_graph=component_graph,
        parameters=pipeline_parameters,
    )
    X_train, y_train = X[:65], y[:65]
    X_validation, y_validation = X[65:], y[65:]
    mock_X, _ = unstack_multiseries(
        X_train,
        y_train,
        series_id="series_id",
        time_index="date",
        target_name="target",
    )
    mock_transform_return_value = (
        mock_X,
        pd.DataFrame(np.random.rand(13, 5)),
    )
    with patch(
        "evalml.pipelines.components.transformers.preprocessing.stl_decomposer.STLDecomposer.transform",
        MagicMock(return_value=mock_transform_return_value),
    ):
        pipeline.fit(X_train, y_train)

    coverage = [0.75, 0.85, 0.95] if set_coverage else None

    pl_intervals = pipeline.get_prediction_intervals(
        X=X_validation,
        y=y_validation,
        X_train=X_train,
        y_train=y_train,
        coverage=coverage,
    )

    if set_coverage is False:
        coverage = [0.95]

    if set_coverage:
        pairs = [(0.75, 0.85), (0.85, 0.95)]
        for pair in pairs:
            assert all(
                [
                    narrower >= broader
                    for narrower, broader in zip(
                        pl_intervals[f"{pair[0]}_lower"],
                        pl_intervals[f"{pair[1]}_lower"],
                    )
                ],
            )
            assert all(
                [
                    narrower <= broader
                    for narrower, broader in zip(
                        pl_intervals[f"{pair[0]}_upper"],
                        pl_intervals[f"{pair[1]}_upper"],
                    )
                ],
            )
    for cover_value in coverage:
        assert all(
            [
                lower < upper
                for lower, upper in zip(
                    pl_intervals[f"{cover_value}_lower"],
                    pl_intervals[f"{cover_value}_upper"],
                )
            ],
        )
