import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

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

    expected_predictions = pd.Series([mean] * len(X))
    predictions = clf.predict(X)
    assert_series_equal(expected_predictions, predictions.to_series())
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
    expected_predictions = pd.Series([median] * len(X))
    predictions = clf.predict(X)
    assert_series_equal(expected_predictions, predictions.to_series())
    np.testing.assert_allclose(clf.feature_importance.iloc[:, 1], np.array([0.0] * X.shape[1]))
