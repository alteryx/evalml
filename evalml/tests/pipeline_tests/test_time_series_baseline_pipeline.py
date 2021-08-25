import numpy as np
import pytest
from pandas.testing import assert_series_equal

from evalml.model_family import ModelFamily
from evalml.pipelines.components import TimeSeriesBaselineEstimator
from evalml.pipelines import TimeSeriesRegressionPipeline, TimeSeriesMulticlassClassificationPipeline, TimeSeriesBinaryClassificationPipeline
from evalml.problem_types import ProblemTypes


def _make_timeseries_baseline_pipeline(problem_type, gap, forecast_horizon):
    pipeline_class, pipeline_name = {
        ProblemTypes.TIME_SERIES_REGRESSION: (
            TimeSeriesRegressionPipeline,
            "Time Series Baseline Regression Pipeline",
        ),
        ProblemTypes.TIME_SERIES_MULTICLASS: (
            TimeSeriesMulticlassClassificationPipeline,
            "Time Series Baseline Multiclass Pipeline",
        ),
        ProblemTypes.TIME_SERIES_BINARY: (
            TimeSeriesBinaryClassificationPipeline,
            "Time Series Baseline Binary Pipeline",
        ),
    }[problem_type]
    baseline = pipeline_class(
        component_graph=["Delayed Feature Transformer", "Time Series Baseline Estimator"],
        custom_name=pipeline_name,
        parameters={
            "pipeline": {
                "date_index": None,
                "gap": gap,
                "max_delay": 0,
                "forecast_horizon": forecast_horizon
            },
            "Delayed Feature Transformer": {"max_delay": 0, "gap": gap, "forecast_horizon": forecast_horizon,
                                            "delay_target": True, "delay_features": False},
            "Time Series Baseline Estimator": {
                "gap": gap,
                "forecast_horizon": forecast_horizon
            },
        },
    )
    return baseline


def test_time_series_baseline_regressor_init():
    baseline = TimeSeriesBaselineEstimator()
    assert baseline.model_family == ModelFamily.BASELINE


def test_time_series_baseline_gap_negative():
    with pytest.raises(ValueError, match="gap value must be a positive integer."):
        TimeSeriesBaselineEstimator(gap=-1)


def test_time_series_baseline_outside_of_pipeline(X_y_regression):
    X, y = X_y_regression

    estimator = TimeSeriesBaselineEstimator(gap=0, forecast_horizon=2)
    estimator.fit(X, y)
    with pytest.raises(ValueError, match="with a DelayedFeaturesTransformer"):
        estimator.predict(X)


@pytest.mark.parametrize("forecast_horizon,gap", [[3, 0], [10, 1], [3, 2]])
@pytest.mark.parametrize("problem_type", [ProblemTypes.TIME_SERIES_REGRESSION])
def test_time_series_baseline(forecast_horizon, gap, problem_type, X_y_regression, X_y_binary, X_y_multi):

    if problem_type.TIME_SERIES_REGRESSION:
        X, y = X_y_regression
    elif problem_type.TIME_SERIES_BINARY:
        X, y = X_y_binary
    else:
        X, y = X_y_multi

    X_train, y_train = X[:50], y[:50]
    X_validation = X[(50 + gap):(50 + gap + forecast_horizon)]

    clf = _make_timeseries_baseline_pipeline(problem_type, gap, forecast_horizon)
    clf.fit(X_train, y_train)

    np.testing.assert_allclose(y[50-forecast_horizon:50], clf.predict(X_validation, None, X_train, y_train).values)
    np.testing.assert_allclose(clf.estimator.feature_importance, np.array([0.0]))


