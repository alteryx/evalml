import numpy as np

from evalml.pipelines import ZeroRBinaryPipeline, ZeroRMulticlassPipeline
from evalml.utils import get_random_state


def test_zeror_binary_random(X_y):
    X, y = X_y
    values = np.unique(y)
    parameters = {
        "ZeroR Classifier": {
            "strategy": "random"
        }
    }
    clf = ZeroRBinaryPipeline(parameters=parameters)
    clf.fit(X, y)
    predicted_proba = clf.predict_proba(X)

    np.testing.assert_allclose(clf.predict(X), get_random_state(0).choice(np.unique(y), len(X)))
    assert predicted_proba.shape == (len(X), 2)
    np.testing.assert_allclose(predicted_proba, np.array([[0.5 for i in range(len(values))]] * len(X)))
    np.testing.assert_allclose(clf.feature_importances.iloc[:, 1], np.array([0.0] * X.shape[1]))


def test_zeror_binary_mode(X_y):
    X, y = X_y
    values, counts = np.unique(y, return_counts=True)
    mode = values[counts.argmax()]
    parameters = {
        "ZeroR Classifier": {
            "strategy": "mode"
        }
    }
    clf = ZeroRBinaryPipeline(parameters=parameters)
    clf.fit(X, y)
    np.testing.assert_allclose(clf.predict(X), np.array([mode] * len(X)))
    predicted_proba = clf.predict_proba(X)
    assert predicted_proba.shape == (len(X), 2)
    np.testing.assert_allclose(predicted_proba, np.array([[1.0 if i == mode else 0.0 for i in range(len(values))]] * len(X)))
    np.testing.assert_allclose(clf.feature_importances.iloc[:, 1], np.array([0.0] * X.shape[1]))


def test_zeror_multi_random(X_y_multi):
    X, y = X_y_multi
    values = np.unique(y)
    parameters = {
        "ZeroR Classifier": {
            "strategy": "random"
        }
    }
    clf = ZeroRMulticlassPipeline(parameters=parameters)
    clf.fit(X, y)
    predicted_proba = clf.predict_proba(X)

    np.testing.assert_allclose(clf.predict(X), get_random_state(0).choice(np.unique(y), len(X)))
    assert predicted_proba.shape == (len(X), 3)
    np.testing.assert_allclose(predicted_proba, np.array([[1. / 3 for i in range(len(values))]] * len(X)))
    np.testing.assert_allclose(clf.feature_importances.iloc[:, 1], np.array([0.0] * X.shape[1]))


def test_zeror_multi_mode(X_y_multi):
    X, y = X_y_multi
    values, counts = np.unique(y, return_counts=True)
    mode = values[counts.argmax()]
    parameters = {
        "ZeroR Classifier": {
            "strategy": "mode"
        }
    }
    clf = ZeroRMulticlassPipeline(parameters=parameters)
    clf.fit(X, y)
    np.testing.assert_allclose(clf.predict(X), np.array([mode] * len(X)))
    predicted_proba = clf.predict_proba(X)
    assert predicted_proba.shape == (len(X), 3)
    np.testing.assert_allclose(predicted_proba, np.array([[1.0 if i == mode else 0.0 for i in range(len(values))]] * len(X)))
    np.testing.assert_allclose(clf.feature_importances.iloc[:, 1], np.array([0.0] * X.shape[1]))
