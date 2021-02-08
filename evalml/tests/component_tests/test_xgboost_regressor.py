import string

import numpy as np
import pandas as pd
import pytest
from pytest import importorskip

from evalml.pipelines.components import XGBoostRegressor
from evalml.utils import SEED_BOUNDS, get_random_state

xgb = importorskip('xgboost', reason='Skipping test because xgboost not installed')


def test_xgboost_regressor_random_seed_bounds_seed(X_y_regression):
    """ensure xgboost's RNG doesn't fail for the min/max bounds we support on user-inputted random seeds"""
    X, y = X_y_regression
    col_names = ["col_{}".format(i) for i in range(len(X[0]))]
    X = pd.DataFrame(X, columns=col_names)
    y = pd.Series(y)
    clf = XGBoostRegressor(n_estimators=1, max_depth=1, random_seed=SEED_BOUNDS.min_bound)
    fitted = clf.fit(X, y)
    assert isinstance(fitted, XGBoostRegressor)
    clf = XGBoostRegressor(n_estimators=1, max_depth=1, random_seed=SEED_BOUNDS.max_bound)
    clf.fit(X, y)


def test_xgboost_feature_name_with_random_ascii(X_y_regression):
    X, y = X_y_regression
    clf = XGBoostRegressor()
    X = get_random_state(clf.random_seed).random((X.shape[0], len(string.printable)))
    col_names = ['column_{}'.format(ascii_char) for ascii_char in string.printable]
    X = pd.DataFrame(X, columns=col_names)
    clf.fit(X, y)
    predictions = clf.predict(X)
    assert len(predictions) == len(y)
    assert not np.isnan(predictions.to_series()).all()

    assert len(clf.feature_importance) == len(X.columns)
    assert not np.isnan(clf.feature_importance).all().all()


@pytest.mark.parametrize("data_type", ['pd', 'ww'])
def test_xgboost_multiindex(data_type, X_y_regression, make_data_type):
    X, y = X_y_regression
    X = pd.DataFrame(X)
    col_names = [('column_{}'.format(num), '{}'.format(num)) for num in range(len(X.columns))]
    X.columns = pd.MultiIndex.from_tuples(col_names)
    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)

    clf = XGBoostRegressor()
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert not y_pred.to_series().isnull().values.any()
