import numpy as np

from evalml.pipelines import ZeroRRegressionPipeline


def test_zeror_mean(X_y_reg):
    X, y = X_y_reg
    mean = y.mean()
    parameters = {
        "ZeroR Regressor": {
            "strategy": "mean"
        }
    }
    clf = ZeroRRegressionPipeline(parameters=parameters)
    clf.fit(X, y)
    np.testing.assert_allclose(clf.predict(X), np.array([mean] * len(X)))
    np.testing.assert_allclose(clf.feature_importances.iloc[:, 1], np.array([0.0] * X.shape[1]))


def test_zeror_median(X_y_reg):
    X, y = X_y_reg
    median = np.median(y)
    parameters = {
        "ZeroR Regressor": {
            "strategy": "median"
        }
    }
    clf = ZeroRRegressionPipeline(parameters=parameters)
    clf.fit(X, y)
    np.testing.assert_allclose(clf.predict(X), np.array([median] * len(X)))
    np.testing.assert_allclose(clf.feature_importances.iloc[:, 1], np.array([0.0] * X.shape[1]))
