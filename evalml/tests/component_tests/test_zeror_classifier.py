import numpy as np
import pandas as pd
import pytest

from evalml.pipelines.components import ZeroRClassifier
from evalml.utils import get_random_state


def test_zeror_invalid_strategy():
    with pytest.raises(ValueError):
        ZeroRClassifier(strategy="unfortunately invalid strategy")


def test_zeror_access_without_fit(X_y):
    X, _ = X_y
    clf = ZeroRClassifier(strategy="mode")
    with pytest.raises(RuntimeError):
        clf.predict(X)
    with pytest.raises(RuntimeError):
        clf.predict_proba(X)
    with pytest.raises(RuntimeError):
        clf.feature_importances

    clf = ZeroRClassifier(strategy="random")
    with pytest.raises(RuntimeError):
        clf.predict(X)
    with pytest.raises(RuntimeError):
        clf.predict_proba(X)
    with pytest.raises(RuntimeError):
        clf.feature_importances


def test_zeror_y_is_None(X_y):
    X, _ = X_y
    with pytest.raises(ValueError):
        ZeroRClassifier().fit(X, y=None)


def test_zeror_binary_mode(X_y):
    X, y = X_y
    values, counts = np.unique(y, return_counts=True)
    mode = values[counts.argmax()]

    clf = ZeroRClassifier(strategy="mode")
    clf.fit(X, y)
    np.testing.assert_allclose(clf.predict(X), np.array([mode] * len(X)))
    predicted_proba = clf.predict_proba(X)
    assert predicted_proba.shape == (len(X), 2)
    np.testing.assert_allclose(predicted_proba, np.array([[1.0 if i == mode else 0.0 for i in range(len(values))]] * len(X)))
    np.testing.assert_allclose(clf.feature_importances, np.array([0.0] * X.shape[1]))


def test_zeror_binary_random(X_y):
    X, y = X_y
    values = np.unique(y)
    clf = ZeroRClassifier(strategy="random", random_state=0)
    clf.fit(X, y)
    np.testing.assert_allclose(clf.predict(X), get_random_state(0).choice(np.unique(y), len(X)))
    predicted_proba = clf.predict_proba(X)
    assert predicted_proba.shape == (len(X), 2)
    np.testing.assert_allclose(predicted_proba, np.array([[0.5 for i in range(len(values))]] * len(X)))
    np.testing.assert_allclose(clf.feature_importances, np.array([0.0] * X.shape[1]))


def test_zeror_multiclass_mode(X_y_multi):
    X, y = X_y_multi
    values, counts = np.unique(y, return_counts=True)
    mode = values[counts.argmax()]

    clf = ZeroRClassifier()
    clf.fit(X, y)
    np.testing.assert_allclose(clf.predict(X), np.array([mode] * len(X)))
    predicted_proba = clf.predict_proba(X)
    assert predicted_proba.shape == (len(X), 3)
    np.testing.assert_allclose(predicted_proba, np.array([[1.0 if i == mode else 0.0 for i in range(len(values))]] * len(X)))
    np.testing.assert_allclose(clf.feature_importances, np.array([0.0] * X.shape[1]))


def test_zeror_multiclass_random(X_y_multi):
    X, y = X_y_multi
    values = np.unique(y)
    clf = ZeroRClassifier(strategy="random", random_state=0)
    clf.fit(X, y)
    np.testing.assert_allclose(clf.predict(X), get_random_state(0).choice(np.unique(y), len(X)))
    predicted_proba = clf.predict_proba(X)
    assert predicted_proba.shape == (len(X), 3)
    np.testing.assert_allclose(predicted_proba, np.array([[1. / 3 for i in range(len(values))]] * len(X)))
    np.testing.assert_allclose(clf.feature_importances, np.array([0.0] * X.shape[1]))


def test_zeror_no_mode():
    X = pd.DataFrame([[1, 2, 3, 0, 1]])
    y = pd.Series([1, 0, 2, 0, 1])
    clf = ZeroRClassifier()
    clf.fit(X, y)
    np.testing.assert_allclose(clf.predict(X), np.array([0] * len(X)))
    predicted_proba = clf.predict_proba(X)
    assert predicted_proba.shape == (len(X), 3)
    np.testing.assert_allclose(predicted_proba, np.array([[1.0 if i == 0 else 0.0 for i in range(3)]] * len(X)))
    np.testing.assert_allclose(clf.feature_importances, np.array([0.0] * X.shape[1]))
