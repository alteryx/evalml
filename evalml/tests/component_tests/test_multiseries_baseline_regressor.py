import pandas as pd
import pytest

from evalml.model_family import ModelFamily
from evalml.pipelines.components import MultiseriesTimeSeriesBaselineRegressor


def test_multiseries_time_series_baseline_regressor_init():
    baseline = MultiseriesTimeSeriesBaselineRegressor()
    assert baseline.model_family == ModelFamily.BASELINE
    assert baseline.is_multiseries
    assert baseline.start_delay == 2

    baseline = MultiseriesTimeSeriesBaselineRegressor(gap=2, forecast_horizon=5)
    assert baseline.start_delay == 7


def test_multiseries_time_series_baseline_gap_negative():
    with pytest.raises(ValueError, match="gap value must be a positive integer."):
        MultiseriesTimeSeriesBaselineRegressor(gap=-1)


def test_multiseries_time_series_baseline_estimator_invalid_y(
    X_y_multiseries_regression,
):
    X, _ = X_y_multiseries_regression

    estimator = MultiseriesTimeSeriesBaselineRegressor(gap=0, forecast_horizon=2)

    with pytest.raises(ValueError, match="if y is None"):
        estimator.fit(X, None)
    with pytest.raises(ValueError, match="y must be a DataFrame"):
        estimator.fit(X, pd.Series(range(100)))


def test_multiseries_time_series_baseline_lags(X_y_multiseries_regression):
    X, y = X_y_multiseries_regression

    estimator = MultiseriesTimeSeriesBaselineRegressor(gap=0, forecast_horizon=2)
    estimator.fit(X, y)

    assert len(estimator._delayed_target) == len(y) + 2
    assert (estimator._delayed_target.columns == y.columns).all()


def test_multiseries_time_series_baseline_includes_future(X_y_multiseries_regression):
    X, y = X_y_multiseries_regression

    estimator = MultiseriesTimeSeriesBaselineRegressor(gap=1, forecast_horizon=2)
    estimator.fit(X, y)

    X_future = pd.DataFrame(columns=X.columns, index=range(len(X), len(X) + 10))
    y_pred = estimator.predict(X_future)

    pd.testing.assert_frame_equal(
        y_pred[:3].reset_index(drop=True),
        y[-3:].reset_index(drop=True),
    )
    assert (y_pred[3:] == 0).all().all()
