import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from evalml.model_family import ModelFamily
from evalml.pipelines.components import BaselineRegressor


def test_baseline_init():
    baseline = BaselineRegressor()
    assert baseline.parameters["strategy"] == "mean"
    assert baseline.model_family == ModelFamily.BASELINE


def test_baseline_invalid_strategy():
    with pytest.raises(ValueError):
        BaselineRegressor(strategy="unfortunately invalid strategy")


def test_baseline_y_is_None(X_y_regression):
    X, _ = X_y_regression
    with pytest.raises(ValueError):
        BaselineRegressor().fit(X, y=None)


def test_baseline_mean(X_y_regression):
    X, y = X_y_regression
    mean = y.mean()
    clf = BaselineRegressor()

    fitted = clf.fit(X, y)
    assert isinstance(fitted, BaselineRegressor)

    expected_predictions = pd.Series([mean] * len(X))
    predictions = clf.predict(X)
    assert_series_equal(expected_predictions, predictions)
    np.testing.assert_allclose(clf.feature_importance, np.array([0.0] * X.shape[1]))


def test_baseline_median(X_y_regression):
    X, y = X_y_regression
    median = np.median(y)
    clf = BaselineRegressor(strategy="median")
    clf.fit(X, y)

    expected_predictions = pd.Series([median] * len(X))
    predictions = clf.predict(X)
    assert_series_equal(expected_predictions, predictions)
    np.testing.assert_allclose(clf.feature_importance, np.array([0.0] * X.shape[1]))
