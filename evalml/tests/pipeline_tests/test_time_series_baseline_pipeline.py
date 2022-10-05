import numpy as np
import pytest
from pandas.testing import assert_series_equal

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
    with pytest.raises(ValueError, match="with a Time Series Featurizer"):
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
    forecast_horizon,
    gap,
    problem_type,
    ts_data,
):
    X, _, y = ts_data(problem_type=problem_type)

    X_train, y_train = X.iloc[:15], y.iloc[:15]
    X_validation = X.iloc[(15 + gap) : (15 + gap + forecast_horizon)]

    clf = make_timeseries_baseline_pipeline(
        problem_type,
        gap,
        forecast_horizon,
        time_index="date",
    )
    clf.fit(X_train, y_train)
    np.testing.assert_allclose(
        y[15 - forecast_horizon : 15],
        clf.predict(X_validation, None, X_train, y_train).values,
    )
    transformed = clf.transform_all_but_final(X_train, y_train)
    delay_index = transformed.columns.tolist().index(
        f"target_delay_{forecast_horizon + gap}",
    )
    importance = np.array([0] * transformed.shape[1])
    importance[delay_index] = 1
    np.testing.assert_allclose(clf.estimator.feature_importance, importance)


@pytest.mark.parametrize("forecast_horizon,gap", [[3, 0], [10, 2], [2, 5]])
def test_time_series_get_forecast_periods(forecast_horizon, gap, ts_data):
    X, _, y = ts_data(problem_type=ProblemTypes.TIME_SERIES_REGRESSION)
    clf = make_timeseries_baseline_pipeline(
        ProblemTypes.TIME_SERIES_REGRESSION,
        gap,
        forecast_horizon,
        time_index="date",
    )
    clf.fit(X, y)
    result = clf.get_forecast_periods(X)

    assert result.size == forecast_horizon + gap
    assert result.iloc[0] == X.iloc[-1]["date"] + np.timedelta64(1, "D")
    assert np.issubdtype(result.dtype, np.datetime64)
    assert result.name == "date"


@pytest.mark.parametrize("forecast_horizon,gap", [[3, 0], [10, 2], [2, 5]])
def test_time_series_get_forecast_predictions(forecast_horizon, gap, ts_data):
    X, _, y = ts_data(problem_type=ProblemTypes.TIME_SERIES_REGRESSION)

    X_train, y_train = X.iloc[:15], y.iloc[:15]
    X_validation = X.iloc[15 : (15 + gap + forecast_horizon)]

    clf = make_timeseries_baseline_pipeline(
        ProblemTypes.TIME_SERIES_REGRESSION,
        gap,
        forecast_horizon,
        time_index="date",
    )
    clf.fit(X_train, y_train)
    forecast_preds = clf.get_forecast_predictions(X=X_train, y=y_train)
    X_val_preds = clf.predict(X_validation, X_train=X_train, y_train=y_train)

    assert_series_equal(forecast_preds, X_val_preds)
