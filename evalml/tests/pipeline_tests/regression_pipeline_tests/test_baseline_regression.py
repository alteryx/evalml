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
    X_t = clf.predict(X)

    # Test unlearned clone
    cloned_clf = clf.clone(learned=False)
    with pytest.raises(RuntimeError):
        cloned_clf.predict(X)
    cloned_clf.fit(X, y)

    np.testing.assert_allclose(X_t, cloned_clf.predict(X))

    # Test learned clone
    clf_clone = clf.clone()
    assert isinstance(clf_clone, BaselineRegressionPipeline)
    assert clf_clone.component_graph[-1].parameters['strategy'] == "mean"
    X_t_clone = clf_clone.predict(X)

    np.testing.assert_almost_equal(X_t.values, X_t_clone.values)
