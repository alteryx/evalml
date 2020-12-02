import numpy as np
import pandas as pd
import pytest

from evalml.model_family import ModelFamily
from evalml.pipelines.components import TimeSeriesBaselineRegressor


def test_baseline_init():
    baseline = TimeSeriesBaselineRegressor()
    assert baseline.model_family == ModelFamily.BASELINE


def test_baseline_y_is_None(X_y_regression):
    X, _ = X_y_regression
    clf = TimeSeriesBaselineRegressor()
    clf.fit(X, y=None)
    with pytest.raises(ValueError):
        clf.predict(X, y=None)


def test_baseline_mean():
    X = pd.DataFrame()
    y = pd.Series([1, 2, 3, 4, 5, 6])
    y_true = pd.Series([1, 1, 2, 3, 4, 5])

    clf = TimeSeriesBaselineRegressor()
    clf.fit(X, y=None)

    np.testing.assert_allclose(clf.predict(X, y), y_true)
    np.testing.assert_allclose(clf.feature_importance, np.array([0.0] * X.shape[1]))
