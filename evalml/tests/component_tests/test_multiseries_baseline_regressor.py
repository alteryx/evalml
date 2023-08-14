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
    multiseries_ts_data_unstacked,
):
    X, _ = multiseries_ts_data_unstacked

    estimator = MultiseriesTimeSeriesBaselineRegressor(gap=0, forecast_horizon=2)

    with pytest.raises(ValueError, match="if y is None"):
        estimator.fit(X, None)
    with pytest.raises(ValueError, match="y must be a DataFrame"):
        estimator.fit(X, pd.Series(range(100)))


def test_multiseries_baseline_no_featurizer(multiseries_ts_data_unstacked):
    X, y = multiseries_ts_data_unstacked

    estimator = MultiseriesTimeSeriesBaselineRegressor(gap=0, forecast_horizon=2)
    estimator.fit(X, y)

    with pytest.raises(ValueError, match="is meant to be used in a pipeline with "):
        estimator.predict(X)


def test_multiseries_time_series_baseline_lags(multiseries_ts_data_unstacked):
    X, y = multiseries_ts_data_unstacked

    feat = TimeSeriesFeaturizer(time_index="date", gap=0, forecast_horizon=2)
    feat.fit(X, y)
    X_t = feat.transform(X, y)

    estimator = MultiseriesTimeSeriesBaselineRegressor(gap=0, forecast_horizon=2)
    estimator.fit(X_t, y)

    pred = estimator.predict(X_t)
    expected = y.shift(2)
    pd.testing.assert_frame_equal(pred, expected)
