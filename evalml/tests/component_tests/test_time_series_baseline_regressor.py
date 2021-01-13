import numpy as np
import pytest
from pandas.testing import assert_series_equal

from evalml.model_family import ModelFamily
from evalml.pipelines.components import TimeSeriesBaselineRegressor


def test_time_series_baseline_regressor_init():
    baseline = TimeSeriesBaselineRegressor()
    assert baseline.model_family == ModelFamily.BASELINE


def test_time_series_baseline_gap_negative():
    with pytest.raises(ValueError, match='gap value must be a positive integer.'):
        TimeSeriesBaselineRegressor(gap=-1)


def test_time_series_baseline_y_is_None(X_y_regression):
    X, _ = X_y_regression
    clf = TimeSeriesBaselineRegressor()
    clf.fit(X, y=None)
    with pytest.raises(ValueError):
        clf.predict(X, y=None)


def test_time_series_baseline(ts_data):
    X, y = ts_data

    clf = TimeSeriesBaselineRegressor(gap=1)
    clf.fit(X, y)

    assert_series_equal(y.astype("Int64"), clf.predict(X, y).to_series())
    np.testing.assert_allclose(clf.feature_importance, np.array([0.0] * X.shape[1]))


def test_time_series_baseline_gap_0(ts_data):
    X, y = ts_data

    y_true = y.shift(periods=1)

    clf = TimeSeriesBaselineRegressor(gap=0)
    clf.fit(X, y)

    assert_series_equal(y_true, clf.predict(X, y).to_series())
    np.testing.assert_allclose(clf.feature_importance, np.array([0.0] * X.shape[1]))


def test_time_series_baseline_no_X(ts_data):
    _, y = ts_data

    clf = TimeSeriesBaselineRegressor()
    clf.fit(X=None, y=y)

    assert_series_equal(y.astype("Int64"), clf.predict(X=None, y=y).to_series())
    np.testing.assert_allclose(clf.feature_importance, np.array([]))
