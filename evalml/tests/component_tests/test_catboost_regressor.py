import warnings

import pandas as pd

from evalml.pipelines.components import CatBoostRegressor
from evalml.utils import SEED_BOUNDS


def test_catboost_regressor_random_seed_bounds_seed(X_y_regression):
    """ensure catboost's RNG doesn't fail for the min/max bounds we support on user-inputted random seeds."""
    X, y = X_y_regression
    col_names = ["col_{}".format(i) for i in range(len(X[0]))]
    X.ww.columns = col_names
    clf = CatBoostRegressor(
        n_estimators=1,
        max_depth=1,
        random_seed=SEED_BOUNDS.min_bound,
    )
    clf.fit(X, y)
    clf = CatBoostRegressor(
        n_estimators=1,
        max_depth=1,
        random_seed=SEED_BOUNDS.max_bound,
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


def test_catboost_regressor_double_categories_in_X():
    X = pd.DataFrame({"double_cats": pd.Series([1.0, 2.0, 3.0, 4.0, 5.0] * 20)})
    y = pd.Series(range(100))
    X.ww.init(logical_types={"double_cats": "Categorical"})

    clf = CatBoostRegressor()
    fitted = clf.fit(X, y)
    assert isinstance(fitted, CatBoostRegressor)


# --> do we need to actually predict? Or do something after just fitting?


# test with float in X
# test that converting to Int64 isn't problematic for estimator
# test that when nans are present it's not problematic for estimator
# test that when we turn floats to strings it's not problematic for estimator

# --> these need to test with predict as well once fit is fixed!
