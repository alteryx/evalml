import numpy as np
import pytest

from evalml.pipelines import BaselineRegressionPipeline


def test_baseline_mean(X_y_reg):
    X, y = X_y_reg
    mean = y.mean()
    parameters = {
        "Baseline Regressor": {
            "strategy": "mean"
        }
    }
    clf = BaselineRegressionPipeline(parameters=parameters)
    clf.fit(X, y)
    np.testing.assert_allclose(clf.predict(X), np.array([mean] * len(X)))
    np.testing.assert_allclose(clf.feature_importances.iloc[:, 1], np.array([0.0] * X.shape[1]))


def test_baseline_median(X_y_reg):
    X, y = X_y_reg
    median = np.median(y)
    parameters = {
        "Baseline Regressor": {
            "strategy": "median"
        }
    }
    clf = BaselineRegressionPipeline(parameters=parameters)
    clf.fit(X, y)
    np.testing.assert_allclose(clf.predict(X), np.array([median] * len(X)))
    np.testing.assert_allclose(clf.feature_importances.iloc[:, 1], np.array([0.0] * X.shape[1]))


def test_clone(X_y_reg):
    X, y = X_y_reg
    parameters = {
        "Baseline Regressor": {
            "strategy": "mean"
        }
    }
    clf = BaselineRegressionPipeline(parameters=parameters)
    clf.fit(X, y)

    cloned_clf = clf.clone()
    with pytest.raises(RuntimeError):
        cloned_clf.predict(X)
    cloned_clf.fit(X, y)

    np.testing.assert_allclose(clf.predict(X), cloned_clf.predict(X))
