import numpy as np
import pandas as pd
import pytest

from evalml.pipelines.components import ZeroRRegressor


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


def test_zeror_mean(X_y):
    X, y = X_y
    values, counts = np.unique(y, return_counts=True)
    mode = values[counts.argmax()]

    clf = ZeroRRegressor()
    clf.fit(X, y)
    # np.testing.assert_allclose(clf.predict(X), np.array([mode] * len(X)))
    # predicted_proba = clf.predict_proba(X)
    # assert predicted_proba.shape == (len(X), 2)
    # np.testing.assert_allclose(predicted_proba, np.array([[1.0 if i == mode else 0.0 for i in range(len(values))]] * len(X)))
    # np.testing.assert_allclose(clf.feature_importances, np.array([0.0] * len(X)))


def test_zeror_mode():
    X = pd.DataFrame([[1, 2, 3, 0, 1]])
    y = pd.Series([1, 0, 2, 0, 1])
    clf = ZeroRRegressor()
    clf.fit(X, y)
    # np.testing.assert_allclose(clf.predict(X), np.array([0] * len(X)))
    # predicted_proba = clf.predict_proba(X)
    # assert predicted_proba.shape == (len(X), 3)
    # np.testing.assert_allclose(predicted_proba, np.array([[1.0 if i == 0 else 0.0 for i in range(3)]] * len(X)))
    # np.testing.assert_allclose(clf.feature_importances, np.array([0.0] * len(X)))
