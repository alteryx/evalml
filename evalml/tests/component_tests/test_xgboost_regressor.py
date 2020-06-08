import string

import numpy as np
import pandas as pd
from pytest import importorskip, raises

from evalml.pipelines.components import XGBoostRegressor
from evalml.utils import SEED_BOUNDS

xgb = importorskip('xgboost', reason='Skipping test because xgboost not installed')


def test_xgboost_regressor_random_state_bounds_seed(X_y_reg):
    """ensure xgboost's RNG doesn't fail for the min/max bounds we support on user-inputted random seeds"""
    X, y = X_y_reg
    col_names = ["col_{}".format(i) for i in range(len(X[0]))]
    X = pd.DataFrame(X, columns=col_names)
    y = pd.Series(y)
    clf = XGBoostRegressor(n_estimators=1, max_depth=1, random_state=SEED_BOUNDS.min_bound)
    clf.fit(X, y)
    clf = XGBoostRegressor(n_estimators=1, max_depth=1, random_state=SEED_BOUNDS.max_bound)
    clf.fit(X, y)


def test_xgboost_regressor_random_state_bounds_rng(X_y_reg):
    """when a RNG is inputted for random_state, ensure the sample we take to get a random seed for xgboost is in xgboost's supported range"""

    def make_mock_random_state(return_value):

        class MockRandomState(np.random.RandomState):

            def randint(self, min_bound, max_bound):
                return return_value
        return MockRandomState()

    X, y = X_y_reg
    col_names = ["col_{}".format(i) for i in range(len(X[0]))]
    X = pd.DataFrame(X, columns=col_names)
    y = pd.Series(y)
    rng = make_mock_random_state(XGBoostRegressor.SEED_MIN)
    clf = XGBoostRegressor(n_estimators=1, max_depth=1, random_state=rng)
    clf.fit(X, y)
    rng = make_mock_random_state(XGBoostRegressor.SEED_MAX)
    clf = XGBoostRegressor(n_estimators=1, max_depth=1, random_state=rng)
    clf.fit(X, y)


def test_xgboost_feature_name_with_random_ascii(X_y):
    X, y = X_y
    clf = XGBoostRegressor()
    X = clf.random_state.random((X.shape[0], len(string.printable)))
    col_names = ['column_{}'.format(ascii_char) for ascii_char in string.printable]
    X = pd.DataFrame(X, columns=col_names)
    clf.fit(X, y)
    assert len(clf.feature_importances) == len(X.columns)
    assert not np.isnan(clf.feature_importances).all().all()

    predictions = clf.predict(X)
    assert len(predictions) == len(y)
    assert not np.isnan(predictions).all()


def test_clone(X_y):
    X, y = X_y
    col_names = ["col_{}".format(i) for i in range(len(X[0]))]
    X = pd.DataFrame(X, columns=col_names)
    y = pd.Series(y)

    clf = XGBoostRegressor(n_estimators=2, max_depth=1)
    clf.fit(X, y)
    X_t = clf.predict(X)

    clf_clone = clf.clone()
    assert clf.random_state == clf_clone.random_state
    with raises(xgb.core.XGBoostError):
        clf_clone.predict(X)
    clf_clone.fit(X, y)
    X_t_clone = clf_clone.predict(X)

    assert clf_clone.parameters['n_estimators'] == 2
    np.testing.assert_almost_equal(X_t, X_t_clone)
