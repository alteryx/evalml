import numpy as np
import pandas as pd
import pytest

from evalml.model_family import ModelFamily
from evalml.pipelines.components import BaselineClassifier
from evalml.utils import get_random_state


def test_baseline_init():
    baseline = BaselineClassifier()
    assert baseline.parameters["strategy"] == "mode"
    assert baseline.model_family == ModelFamily.BASELINE


def test_baseline_invalid_strategy():
    with pytest.raises(ValueError):
        BaselineClassifier(strategy="unfortunately invalid strategy")


def test_baseline_access_without_fit(X_y_binary):
    X, _ = X_y_binary
    clf = BaselineClassifier()
    with pytest.raises(RuntimeError):
        clf.predict(X)
    with pytest.raises(RuntimeError):
        clf.predict_proba(X)
    with pytest.raises(RuntimeError):
        clf.feature_importance


def test_baseline_y_is_None(X_y_binary):
    X, _ = X_y_binary
    with pytest.raises(ValueError):
        BaselineClassifier().fit(X, y=None)


def test_baseline_binary_mode(X_y_binary):
    X = pd.DataFrame({'one': [1, 2, 3, 4], 'two': [2, 3, 4, 5], 'three': [1, 2, 3, 4]})
    y = pd.Series([10, 11, 10, 10])
    clf = BaselineClassifier(strategy="mode")
    clf.fit(X, y)
    np.testing.assert_allclose(clf.predict(X), np.array([10] * len(X)))
    predicted_proba = clf.predict_proba(X)
    assert predicted_proba.shape == (len(X), 2)
    expected_predicted_proba = pd.DataFrame({10: [1., 1., 1., 1.], 11: [0., 0., 0., 0.]})
    pd.testing.assert_frame_equal(expected_predicted_proba, predicted_proba)
    np.testing.assert_allclose(clf.feature_importance, np.array([0.0] * X.shape[1]))


def test_baseline_binary_random(X_y_binary):
    X, y = X_y_binary
    values = np.unique(y)
    clf = BaselineClassifier(strategy="random", random_state=0)
    clf.fit(X, y)
    np.testing.assert_allclose(clf.predict(X), get_random_state(0).choice(np.unique(y), len(X)))
    predicted_proba = clf.predict_proba(X)
    assert predicted_proba.shape == (len(X), 2)
    np.testing.assert_allclose(predicted_proba, np.array([[0.5 for i in range(len(values))]] * len(X)))
    np.testing.assert_allclose(clf.feature_importance, np.array([0.0] * X.shape[1]))


def test_baseline_binary_random_weighted(X_y_binary):
    X, y = X_y_binary
    values, counts = np.unique(y, return_counts=True)
    percent_freq = counts.astype(float) / len(y)
    assert percent_freq.sum() == 1.0
    clf = BaselineClassifier(strategy="random_weighted", random_state=0)
    clf.fit(X, y)
    np.testing.assert_allclose(clf.predict(X), get_random_state(0).choice(np.unique(y), len(X), p=percent_freq))
    predicted_proba = clf.predict_proba(X)
    assert predicted_proba.shape == (len(X), 2)
    np.testing.assert_allclose(predicted_proba, np.array([[percent_freq[i] for i in range(len(values))]] * len(X)))
    np.testing.assert_allclose(clf.feature_importance, np.array([0.0] * X.shape[1]))


def test_baseline_multiclass_mode():
    X = pd.DataFrame({'one': [1, 2, 3, 4], 'two': [2, 3, 4, 5], 'three': [1, 2, 3, 4]})
    y = pd.Series([10, 11, 12, 11])
    clf = BaselineClassifier(strategy="mode")
    clf.fit(X, y)
    np.testing.assert_allclose(clf.predict(X), np.array([11] * len(X)))
    predicted_proba = clf.predict_proba(X)
    assert predicted_proba.shape == (len(X), 3)
    expected_predicted_proba = pd.DataFrame({10: [0., 0., 0., 0.], 11: [1., 1., 1., 1.], 12: [0., 0., 0., 0.]})
    pd.testing.assert_frame_equal(expected_predicted_proba, predicted_proba)
    np.testing.assert_allclose(clf.feature_importance, np.array([0.0] * X.shape[1]))


def test_baseline_multiclass_random(X_y_multi):
    X, y = X_y_multi
    values = np.unique(y)
    clf = BaselineClassifier(strategy="random", random_state=0)
    clf.fit(X, y)
    np.testing.assert_allclose(clf.predict(X), get_random_state(0).choice(np.unique(y), len(X)))
    predicted_proba = clf.predict_proba(X)
    assert predicted_proba.shape == (len(X), 3)
    np.testing.assert_allclose(predicted_proba, np.array([[1. / 3 for i in range(len(values))]] * len(X)))
    np.testing.assert_allclose(clf.feature_importance, np.array([0.0] * X.shape[1]))


def test_baseline_multiclass_random_weighted(X_y_multi):
    X, y = X_y_multi
    values, counts = np.unique(y, return_counts=True)
    percent_freq = counts.astype(float) / len(y)
    assert percent_freq.sum() == 1.0

    clf = BaselineClassifier(strategy="random_weighted", random_state=0)
    clf.fit(X, y)
    np.testing.assert_allclose(clf.predict(X), get_random_state(0).choice(np.unique(y), len(X), p=percent_freq))
    predicted_proba = clf.predict_proba(X)
    assert predicted_proba.shape == (len(X), 3)
    np.testing.assert_allclose(predicted_proba, np.array([[percent_freq[i] for i in range(len(values))]] * len(X)))
    np.testing.assert_allclose(clf.feature_importance, np.array([0.0] * X.shape[1]))


def test_baseline_no_mode():
    X = pd.DataFrame([[1, 2, 3, 0, 1]])
    y = pd.Series([1, 0, 2, 0, 1])
    clf = BaselineClassifier()
    clf.fit(X, y)
    np.testing.assert_allclose(clf.predict(X), np.array([0] * len(X)))
    predicted_proba = clf.predict_proba(X)
    assert predicted_proba.shape == (len(X), 3)
    np.testing.assert_allclose(predicted_proba, np.array([[1.0 if i == 0 else 0.0 for i in range(3)]] * len(X)))
    np.testing.assert_allclose(clf.feature_importance, np.array([0.0] * X.shape[1]))
