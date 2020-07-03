import string

import numpy as np
import pandas as pd
from pytest import importorskip

from evalml.pipelines.components import CatBoostClassifier
from evalml.utils import SEED_BOUNDS

importorskip('catboost', reason='Skipping test because catboost not installed')


def test_catboost_classifier_random_state_bounds_seed(X_y_binary):
    """ensure catboost's RNG doesn't fail for the min/max bounds we support on user-inputted random seeds"""
    X, y = X_y_binary
    col_names = ["col_{}".format(i) for i in range(len(X[0]))]
    X = pd.DataFrame(X, columns=col_names)
    y = pd.Series(y)
    clf = CatBoostClassifier(n_estimators=1, max_depth=1, random_state=SEED_BOUNDS.min_bound)
    clf.fit(X, y)
    clf = CatBoostClassifier(n_estimators=1, max_depth=1, random_state=SEED_BOUNDS.max_bound)
    clf.fit(X, y)


def test_catboost_classifier_random_state_bounds_rng(X_y_binary):
    """when a RNG is inputted for random_state, ensure the sample we take to get a random seed for catboost is in catboost's supported range"""

    def make_mock_random_state(return_value):

        class MockRandomState(np.random.RandomState):

            def randint(self, min_bound, max_bound):
                return return_value
        return MockRandomState()

    X, y = X_y_binary
    col_names = ["col_{}".format(i) for i in range(len(X[0]))]
    X = pd.DataFrame(X, columns=col_names)
    y = pd.Series(y)
    rng = make_mock_random_state(CatBoostClassifier.SEED_MIN)
    clf = CatBoostClassifier(n_estimators=1, max_depth=1, random_state=rng)
    clf.fit(X, y)
    rng = make_mock_random_state(CatBoostClassifier.SEED_MAX)
    clf = CatBoostClassifier(n_estimators=1, max_depth=1, random_state=rng)
    clf.fit(X, y)


def test_catboost_feature_name_with_random_ascii(X_y_binary):
    X, y = X_y_binary
    clf = CatBoostClassifier()
    X = clf.random_state.random((X.shape[0], len(string.printable)))
    col_names = ['column_{}'.format(ascii_char) for ascii_char in string.printable]
    X = pd.DataFrame(X, columns=col_names)
    clf.fit(X, y)
    assert len(clf.feature_importance) == len(X.columns)
    assert not np.isnan(clf.feature_importance).all().all()

    predictions = clf.predict(X)
    assert len(predictions) == len(y)
    assert not np.isnan(predictions).all()
