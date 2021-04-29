import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from evalml.pipelines import (
    BaselineRegressionPipeline,
    MeanBaselineRegressionPipeline
)


def test_baseline_regression_init(X_y_binary):
    parameters = {
        "Baseline Regressor": {
            "strategy": "median"
        }
    }
    clf = BaselineRegressionPipeline(parameters=parameters)
    assert clf.custom_hyperparameters is None
    assert clf.name == "Baseline Regression Pipeline"

    clf = MeanBaselineRegressionPipeline({})
    assert clf.custom_hyperparameters == {"strategy": ["mean"]}
    assert clf.name == "Mean Baseline Regression Pipeline"


def test_baseline_regression_new_clone():
    parameters = {
        "Baseline Regressor": {
            "strategy": "mean"
        }
    }
    clf = BaselineRegressionPipeline(parameters=parameters)
    cloned_clf = clf.clone()
    assert cloned_clf == clf
    assert cloned_clf.name == "Baseline Regression Pipeline"
    assert cloned_clf.parameters == parameters

    new_parameters = {
        "Baseline Regressor": {
            "strategy": "median"
        }
    }
    new_clf = clf.new(parameters=new_parameters)
    assert new_clf.name == "Baseline Regression Pipeline"
    assert new_clf.parameters == new_parameters


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
    assert_series_equal(expected_predictions, predictions)
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
    assert_series_equal(expected_predictions, predictions)
    np.testing.assert_allclose(clf.feature_importance.iloc[:, 1], np.array([0.0] * X.shape[1]))
