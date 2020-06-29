import numpy as np

from evalml.pipelines import BaselineRegressionPipeline


def test_baseline_mean(X_y_regression):
    X, y = X_y_regression
    mean = y.mean()
    parameters = {
        "Baseline Regressor": {
            "strategy": "mean"
        }
    }
    clf = BaselineRegressionPipeline(parameters=parameters)
    clf.fit(X, y)
    np.testing.assert_allclose(clf.predict(X), np.array([mean] * len(X)))
    np.testing.assert_allclose(clf.feature_importance.iloc[:, 1], np.array([0.0] * X.shape[1]))


def test_baseline_median(X_y_regression):
    X, y = X_y_regression
    median = np.median(y)
    parameters = {
        "Baseline Regressor": {
            "strategy": "median"
        }
    }
    clf = BaselineRegressionPipeline(parameters=parameters)
    clf.fit(X, y)
    np.testing.assert_allclose(clf.predict(X), np.array([median] * len(X)))
    np.testing.assert_allclose(clf.feature_importance.iloc[:, 1], np.array([0.0] * X.shape[1]))
