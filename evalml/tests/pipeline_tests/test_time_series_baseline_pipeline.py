import numpy as np
import pandas as pd
import pytest

from evalml.model_family import ModelFamily
from evalml.pipelines.components import TimeSeriesBaselineEstimator
from evalml.pipelines.utils import make_timeseries_baseline_pipeline
from evalml.problem_types import ProblemTypes


def test_time_series_baseline_regressor_init():
    baseline = TimeSeriesBaselineEstimator()
    assert baseline.model_family == ModelFamily.BASELINE


def test_time_series_baseline_gap_negative():
    with pytest.raises(ValueError, match="gap value must be a positive integer."):
        TimeSeriesBaselineEstimator(gap=-1)


def test_time_series_baseline_estimator_y_is_none(X_y_regression):
    X, y = X_y_regression

    estimator = TimeSeriesBaselineEstimator(gap=0, forecast_horizon=2)

    with pytest.raises(ValueError, match="if y is None"):
        estimator.fit(X, None)


def test_time_series_baseline_outside_of_pipeline(X_y_regression):
    X, y = X_y_regression

    estimator = TimeSeriesBaselineEstimator(gap=0, forecast_horizon=2)
    estimator.fit(X, y)
    with pytest.raises(ValueError, match="with a DelayedFeaturesTransformer"):
        estimator.predict(X)


@pytest.mark.parametrize("forecast_horizon,gap", [[3, 0], [10, 1], [3, 2]])
@pytest.mark.parametrize(
    "problem_type",
    [
        ProblemTypes.TIME_SERIES_REGRESSION,
        ProblemTypes.TIME_SERIES_BINARY,
        ProblemTypes.TIME_SERIES_MULTICLASS,
    ],
)
def test_time_series_baseline(
    forecast_horizon, gap, problem_type, X_y_regression, X_y_binary, X_y_multi
):

    if problem_type.TIME_SERIES_REGRESSION:
        X, y = X_y_regression
    elif problem_type.TIME_SERIES_BINARY:
        X, y = X_y_binary
    else:
        X, y = X_y_multi

    X = pd.DataFrame(X)
    y = pd.Series(y)

    X_train, y_train = X.iloc[:50], y.iloc[:50]
    X_validation = X.iloc[(50 + gap) : (50 + gap + forecast_horizon)]

    clf = make_timeseries_baseline_pipeline(problem_type, gap, forecast_horizon)
    clf.fit(X_train, y_train)
    np.testing.assert_allclose(
        y[50 - forecast_horizon : 50],
        clf.predict(X_validation, None, X_train, y_train).values,
    )
    np.testing.assert_allclose(clf.estimator.feature_importance, np.array([0.0]))
