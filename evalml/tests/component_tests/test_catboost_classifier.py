import pandas as pd
from pytest import importorskip

from evalml.pipelines.components import CatBoostClassifier
from evalml.utils import SEED_BOUNDS

importorskip('catboost', reason='Skipping test because catboost not installed')


def test_catboost_classifier_random_seed_bounds_seed(X_y_binary):
    """ensure catboost's RNG doesn't fail for the min/max bounds we support on user-inputted random seeds"""
    X, y = X_y_binary
    col_names = ["col_{}".format(i) for i in range(len(X[0]))]
    X = pd.DataFrame(X, columns=col_names)
    y = pd.Series(y)
    clf = CatBoostClassifier(n_estimators=1, max_depth=1, random_seed=SEED_BOUNDS.min_bound)
    clf.fit(X, y)
    clf = CatBoostClassifier(n_estimators=1, max_depth=1, random_seed=SEED_BOUNDS.max_bound)
    fitted = clf.fit(X, y)
    assert isinstance(fitted, CatBoostClassifier)
