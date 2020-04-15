import numpy as np
import pandas as pd
from pytest import importorskip

from evalml.pipelines.components import XGBoostClassifier
from evalml.utils import SEED_BOUNDS

xgb = importorskip('xgboost', reason='Skipping test because xgboost not installed')


def test_xgboost_classifier_random_state_bounds_seed(X_y):
    """ensure xgboost's RNG doesn't fail for the min/max bounds we support on user-inputted random seeds"""
    X, y = X_y
    col_names = ["col_{}".format(i) for i in range(len(X[0]))]
    X = pd.DataFrame(X, columns=col_names)
    y = pd.Series(y)
    clf = XGBoostClassifier(n_estimators=1, max_depth=1, random_state=SEED_BOUNDS.min_bound)
    clf.fit(X, y)
    clf = XGBoostClassifier(n_estimators=1, max_depth=1, random_state=SEED_BOUNDS.max_bound)
    clf.fit(X, y)


def test_xgboost_classifier_random_state_bounds_rng(X_y):
    """when a RNG is inputted for random_state, ensure the sample we take to get a random seed for xgboost is in xgboost's supported range"""

    def make_mock_random_state(return_value):

        class MockRandomState(np.random.RandomState):

            def randint(self, min_bound, max_bound):
                return return_value
        return MockRandomState()

    X, y = X_y
    col_names = ["col_{}".format(i) for i in range(len(X[0]))]
    X = pd.DataFrame(X, columns=col_names)
    y = pd.Series(y)
    rng = make_mock_random_state(XGBoostClassifier.SEED_MIN)
    clf = XGBoostClassifier(n_estimators=1, max_depth=1, random_state=rng)
    clf.fit(X, y)
    rng = make_mock_random_state(XGBoostClassifier.SEED_MAX)
    clf = XGBoostClassifier(n_estimators=1, max_depth=1, random_state=rng)
    clf.fit(X, y)


def test_xgboost_feature_names_with_symbols(X_y):
    X, y = X_y
    col_names = ["[[<<col_{}>>]]".format(i) for i in range(len(X[0]))]
    X = pd.DataFrame(X, columns=col_names)
    clf = XGBoostClassifier()
    clf.fit(X, y)
    assert list(X.columns) == col_names
