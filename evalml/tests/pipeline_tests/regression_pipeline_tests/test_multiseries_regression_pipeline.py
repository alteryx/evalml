import pandas as pd
import pytest

from evalml.pipelines import MultiseriesRegressionPipeline
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
