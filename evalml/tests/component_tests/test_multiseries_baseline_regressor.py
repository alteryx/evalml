import pandas as pd
import pytest

from evalml.model_family import ModelFamily
from evalml.pipelines.components import (
    MultiseriesTimeSeriesBaselineRegressor,
    TimeSeriesFeaturizer,
)


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

    feat = TimeSeriesFeaturizer(time_index="index", gap=0, forecast_horizon=2)
    feat.fit(X, y)
    X_t = feat.transform(X, y)

    estimator = MultiseriesTimeSeriesBaselineRegressor(gap=0, forecast_horizon=2)
    estimator.fit(X_t, y)

    pred = estimator.predict(X_t)
    expected = y.shift(2)
    expected.columns = [f"{col}_delay_2" for col in expected.columns]
    pd.testing.assert_frame_equal(pred, expected)
