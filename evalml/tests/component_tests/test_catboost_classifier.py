import numpy as np
import pandas as pd
from pytest import importorskip, raises

from evalml.pipelines.components import CatBoostClassifier
from evalml.utils import SEED_BOUNDS

catboost = importorskip('catboost', reason='Skipping test because catboost not installed')


def test_catboost_classifier_random_state_bounds_seed(X_y_reg):
    """ensure catboost's RNG doesn't fail for the min/max bounds we support on user-inputted random seeds"""
    X, y = X_y_reg
    col_names = ["col_{}".format(i) for i in range(len(X[0]))]
    X = pd.DataFrame(X, columns=col_names)
    y = pd.Series(y)
    clf = CatBoostClassifier(n_estimators=1, max_depth=1, random_state=SEED_BOUNDS.min_bound)
    clf.fit(X, y)
    clf = CatBoostClassifier(n_estimators=1, max_depth=1, random_state=SEED_BOUNDS.max_bound)
    clf.fit(X, y)


def test_catboost_classifier_random_state_bounds_rng(X_y_reg):
    """when a RNG is inputted for random_state, ensure the sample we take to get a random seed for catboost is in catboost's supported range"""

    def make_mock_random_state(return_value):

        class MockRandomState(np.random.RandomState):

            def randint(self, min_bound, max_bound):
                return return_value
        return MockRandomState()

    X, y = X_y_reg
    col_names = ["col_{}".format(i) for i in range(len(X[0]))]
    X = pd.DataFrame(X, columns=col_names)
    y = pd.Series(y)
    rng = make_mock_random_state(CatBoostClassifier.SEED_MIN)
    clf = CatBoostClassifier(n_estimators=1, max_depth=1, random_state=rng)
    clf.fit(X, y)
    rng = make_mock_random_state(CatBoostClassifier.SEED_MAX)
    clf = CatBoostClassifier(n_estimators=1, max_depth=1, random_state=rng)
    clf.fit(X, y)


def test_clone(X_y):
    X, y = X_y
    col_names = ["col_{}".format(i) for i in range(len(X[0]))]
    X = pd.DataFrame(X, columns=col_names)
    y = pd.Series(y)

    clf = CatBoostClassifier(n_estimators=2, max_depth=1, bootstrap_type='Bernoulli')
    clf.fit(X, y)
    X_t = clf.predict(X)

    # Test unlearned clone
    clf_clone = clf.clone()
    assert clf_clone.parameters['bootstrap_type'] == 'Bernoulli'
    with raises(catboost.CatBoostError):
        clf_clone.predict(X)
    clf_clone.fit(X, y)
    X_t_clone = clf_clone.predict(X)

    # Test learned clone
    clf_clone = clf.clone(deep=True)
    assert clf_clone.parameters['bootstrap_type'] == 'Bernoulli'

    X_t_clone = clf_clone.predict(X)
    assert clf_clone.parameters['n_estimators'] == 2
    np.testing.assert_almost_equal(X_t, X_t_clone)
