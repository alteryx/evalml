import numpy as np
import pytest

from evalml.pipelines.components import BaselineRegressor


def test_baseline_invalid_strategy():
    with pytest.raises(ValueError):
        BaselineRegressor(strategy="unfortunately invalid strategy")


def test_baseline_access_without_fit(X_y_reg):
    X, _ = X_y_reg
    clf = BaselineRegressor()
    with pytest.raises(RuntimeError):
        clf.predict(X)
    with pytest.raises(RuntimeError):
        clf.feature_importances


def test_baseline_y_is_None(X_y_reg):
    X, _ = X_y_reg
    with pytest.raises(ValueError):
        BaselineRegressor().fit(X, y=None)


def test_baseline_mean(X_y_reg):
    X, y = X_y_reg
    mean = y.mean()
    clf = BaselineRegressor()
    clf.fit(X, y)
    np.testing.assert_allclose(clf.predict(X), np.array([mean] * len(X)))
    np.testing.assert_allclose(clf.feature_importances, np.array([0.0] * X.shape[1]))


def test_baseline_median(X_y_reg):
    X, y = X_y_reg
    median = np.median(y)
    clf = BaselineRegressor(strategy="median")
    clf.fit(X, y)
    np.testing.assert_allclose(clf.predict(X), np.array([median] * len(X)))
    np.testing.assert_allclose(clf.feature_importances, np.array([0.0] * X.shape[1]))
