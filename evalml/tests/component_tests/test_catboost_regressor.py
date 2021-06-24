import warnings

import pandas as pd
from pytest import importorskip

from evalml.pipelines.components import CatBoostRegressor
from evalml.utils import SEED_BOUNDS

importorskip("catboost", reason="Skipping test because catboost not installed")


def test_catboost_regressor_random_seed_bounds_seed(X_y_regression):
    """ensure catboost's RNG doesn't fail for the min/max bounds we support on user-inputted random seeds"""
    X, y = X_y_regression
    col_names = ["col_{}".format(i) for i in range(len(X[0]))]
    X = pd.DataFrame(X, columns=col_names)
    y = pd.Series(y)
    clf = CatBoostRegressor(
        n_estimators=1, max_depth=1, random_seed=SEED_BOUNDS.min_bound
    )
    clf.fit(X, y)
    clf = CatBoostRegressor(
        n_estimators=1, max_depth=1, random_seed=SEED_BOUNDS.max_bound
    )
    fitted = clf.fit(X, y)
    assert isinstance(fitted, CatBoostRegressor)


def test_catboost_regressor_init_n_jobs():
    n_jobs = 2
    clf = CatBoostRegressor(n_jobs=n_jobs)
    assert clf._component_obj.get_param("thread_count") == n_jobs


def test_catboost_regressor_init_thread_count():
    with warnings.catch_warnings(record=True) as w:
        CatBoostRegressor(thread_count=2)
    assert len(w) == 1
    assert "Parameter 'thread_count' will be ignored. " in str(w[-1].message)
