import numpy as np
import pytest

from evalml.pipelines.components import ZeroRRegressor


def test_zeror_invalid_strategy():
    with pytest.raises(ValueError):
        ZeroRRegressor(strategy="unfortunately invalid strategy")


def test_zeror_access_without_fit(X_y_reg):
    X, _ = X_y_reg
    clf = ZeroRRegressor()
    with pytest.raises(RuntimeError):
        clf.predict(X)
    with pytest.raises(RuntimeError):
        clf.feature_importances


def test_zeror_y_is_None(X_y_reg):
    X, _ = X_y_reg
    with pytest.raises(ValueError):
        ZeroRRegressor().fit(X, y=None)


def test_zeror_mean(X_y_reg):
    X, y = X_y_reg
    mean = y.mean()
    clf = ZeroRRegressor()
    clf.fit(X, y)
    np.testing.assert_allclose(clf.predict(X), np.array([mean] * len(X)))
    np.testing.assert_allclose(clf.feature_importances, np.array([0.0] * X.shape[1]))


def test_zeror_median(X_y_reg):
    X, y = X_y_reg
    median = np.median(y)
    clf = ZeroRRegressor(strategy="median")
    clf.fit(X, y)
    np.testing.assert_allclose(clf.predict(X), np.array([median] * len(X)))
    np.testing.assert_allclose(clf.feature_importances, np.array([0.0] * X.shape[1]))
