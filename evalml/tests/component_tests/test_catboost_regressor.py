import pandas as pd
from pytest import importorskip

from evalml.pipelines.components import CatBoostRegressor
from evalml.utils import SEED_BOUNDS

importorskip('catboost', reason='Skipping test because catboost not installed')


def test_catboost_regressor_random_seed_bounds_seed(X_y_regression):
    """ensure catboost's RNG doesn't fail for the min/max bounds we support on user-inputted random seeds"""
    X, y = X_y_regression
    col_names = ["col_{}".format(i) for i in range(len(X[0]))]
    X = pd.DataFrame(X, columns=col_names)
    y = pd.Series(y)
    clf = CatBoostRegressor(n_estimators=1, max_depth=1, random_seed=SEED_BOUNDS.min_bound)
    clf.fit(X, y)
    clf = CatBoostRegressor(n_estimators=1, max_depth=1, random_seed=SEED_BOUNDS.max_bound)
    fitted = clf.fit(X, y)
    assert isinstance(fitted, CatBoostRegressor)
