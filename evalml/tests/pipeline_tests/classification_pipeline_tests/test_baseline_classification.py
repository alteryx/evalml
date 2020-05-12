import numpy as np

from evalml.pipelines import BaselineBinaryPipeline, BaselineMulticlassPipeline
from evalml.utils import get_random_state


def test_baseline_binary_random(X_y):
    X, y = X_y
    values = np.unique(y)
    parameters = {
        "Baseline Classifier": {
            "strategy": "random"
        }
    }
    clf = BaselineBinaryPipeline(parameters=parameters)
    clf.fit(X, y)
    predicted_proba = clf.predict_proba(X)

    np.testing.assert_allclose(clf.predict(X), get_random_state(0).choice(np.unique(y), len(X)))
    assert predicted_proba.shape == (len(X), 2)
    np.testing.assert_allclose(predicted_proba, np.array([[0.5 for i in range(len(values))]] * len(X)))
    np.testing.assert_allclose(clf.feature_importances.iloc[:, 1], np.array([0.0] * X.shape[1]))


def test_baseline_binary_random_weighted(X_y):
    X, y = X_y
    values, counts = np.unique(y, return_counts=True)
    percent_freq = counts.astype(float) / len(y)
    assert percent_freq.sum() == 1.0

    parameters = {
        "Baseline Classifier": {
            "strategy": "random_weighted"
        }
    }
    clf = BaselineBinaryPipeline(parameters=parameters)
    clf.fit(X, y)
    predicted_proba = clf.predict_proba(X)

    np.testing.assert_allclose(clf.predict(X), get_random_state(0).choice(np.unique(y), len(X), p=percent_freq))
    assert predicted_proba.shape == (len(X), 2)
    np.testing.assert_allclose(predicted_proba, np.array([[percent_freq[i] for i in range(len(values))]] * len(X)))
    np.testing.assert_allclose(clf.feature_importances.iloc[:, 1], np.array([0.0] * X.shape[1]))


def test_baseline_binary_mode(X_y):
    X, y = X_y
    values, counts = np.unique(y, return_counts=True)
    mode = values[counts.argmax()]
    parameters = {
        "Baseline Classifier": {
            "strategy": "mode"
        }
    }
    clf = BaselineBinaryPipeline(parameters=parameters)
    clf.fit(X, y)
    np.testing.assert_allclose(clf.predict(X), np.array([mode] * len(X)))
    predicted_proba = clf.predict_proba(X)
    assert predicted_proba.shape == (len(X), 2)
    np.testing.assert_allclose(predicted_proba, np.array([[1.0 if i == mode else 0.0 for i in range(len(values))]] * len(X)))
    np.testing.assert_allclose(clf.feature_importances.iloc[:, 1], np.array([0.0] * X.shape[1]))


def test_baseline_multi_random(X_y_multi):
    X, y = X_y_multi
    values = np.unique(y)
    parameters = {
        "Baseline Classifier": {
            "strategy": "random"
        }
    }
    clf = BaselineMulticlassPipeline(parameters=parameters)
    clf.fit(X, y)
    predicted_proba = clf.predict_proba(X)

    np.testing.assert_allclose(clf.predict(X), get_random_state(0).choice(np.unique(y), len(X)))
    assert predicted_proba.shape == (len(X), 3)
    np.testing.assert_allclose(predicted_proba, np.array([[1. / 3 for i in range(len(values))]] * len(X)))
    np.testing.assert_allclose(clf.feature_importances.iloc[:, 1], np.array([0.0] * X.shape[1]))


def test_baseline_multi_random_weighted(X_y_multi):
    X, y = X_y_multi
    values, counts = np.unique(y, return_counts=True)
    percent_freq = counts.astype(float) / len(y)
    assert percent_freq.sum() == 1.0

    parameters = {
        "Baseline Classifier": {
            "strategy": "random_weighted"
        }
    }
    clf = BaselineMulticlassPipeline(parameters=parameters)
    clf.fit(X, y)
    predicted_proba = clf.predict_proba(X)

    np.testing.assert_allclose(clf.predict(X), get_random_state(0).choice(np.unique(y), len(X), p=percent_freq))
    assert predicted_proba.shape == (len(X), 3)
    np.testing.assert_allclose(predicted_proba, np.array([[percent_freq[i] for i in range(len(values))]] * len(X)))
    np.testing.assert_allclose(clf.feature_importances.iloc[:, 1], np.array([0.0] * X.shape[1]))


def test_baseline_multi_mode(X_y_multi):
    X, y = X_y_multi
    values, counts = np.unique(y, return_counts=True)
    mode = values[counts.argmax()]
    parameters = {
        "Baseline Classifier": {
            "strategy": "mode"
        }
    }
    clf = BaselineMulticlassPipeline(parameters=parameters)
    clf.fit(X, y)
    np.testing.assert_allclose(clf.predict(X), np.array([mode] * len(X)))
    predicted_proba = clf.predict_proba(X)
    assert predicted_proba.shape == (len(X), 3)
    np.testing.assert_allclose(predicted_proba, np.array([[1.0 if i == mode else 0.0 for i in range(len(values))]] * len(X)))
    np.testing.assert_allclose(clf.feature_importances.iloc[:, 1], np.array([0.0] * X.shape[1]))
