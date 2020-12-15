import numpy as np
import pytest

from evalml.model_family import ModelFamily
from evalml.pipelines.components import GAMClassifier


def test_gam_classifier_init():
    baseline = GAMClassifier()
    assert baseline.model_family == ModelFamily.LINEAR_MODEL

'''
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

    np.testing.assert_allclose(clf.predict(X, y), y)
    np.testing.assert_allclose(clf.feature_importance, np.array([0.0] * X.shape[1]))


def test_time_series_baseline_gap_0(ts_data):
    X, y = ts_data

    y_true = y.shift(periods=1)

    clf = TimeSeriesBaselineRegressor(gap=0)
    clf.fit(X, y)

    np.testing.assert_allclose(clf.predict(X, y), y_true)
    np.testing.assert_allclose(clf.feature_importance, np.array([0.0] * X.shape[1]))


def test_time_series_baseline_no_X(ts_data):
    _, y = ts_data

    clf = TimeSeriesBaselineRegressor()
    clf.fit(X=None, y=y)

    np.testing.assert_allclose(clf.predict(X=None, y=y), y)
    np.testing.assert_allclose(clf.feature_importance, np.array([]))
'''