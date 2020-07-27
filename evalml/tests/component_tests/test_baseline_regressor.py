import numpy as np
import pytest

from evalml.model_family import ModelFamily
from evalml.pipelines.components import BaselineRegressor


def test_baseline_init():
    baseline = BaselineRegressor()
    assert baseline.parameters["strategy"] == "mean"
    assert baseline.model_family == ModelFamily.BASELINE


def test_baseline_invalid_strategy():
    with pytest.raises(ValueError):
        BaselineRegressor(strategy="unfortunately invalid strategy")


def test_baseline_y_is_None(X_y_regression):
    X, _ = X_y_regression
    with pytest.raises(ValueError):
        BaselineRegressor().fit(X, y=None)


def test_baseline_mean(X_y_regression):
    X, y = X_y_regression
    mean = y.mean()
    clf = BaselineRegressor()
    clf.fit(X, y)
    np.testing.assert_allclose(clf.predict(X), np.array([mean] * len(X)))
    np.testing.assert_allclose(clf.feature_importance, np.array([0.0] * X.shape[1]))


def test_baseline_median(X_y_regression):
    X, y = X_y_regression
    median = np.median(y)
    clf = BaselineRegressor(strategy="median")
    clf.fit(X, y)
    np.testing.assert_allclose(clf.predict(X), np.array([median] * len(X)))
    np.testing.assert_allclose(clf.feature_importance, np.array([0.0] * X.shape[1]))
