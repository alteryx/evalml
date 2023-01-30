import warnings

import pandas as pd
import woodwork as ww

from evalml.pipelines.components import CatBoostClassifier
from evalml.utils import SEED_BOUNDS


def test_catboost_classifier_random_seed_bounds_seed(X_y_binary):
    """ensure catboost's RNG doesn't fail for the min/max bounds we support on user-inputted random seeds."""
    X, y = X_y_binary
    col_names = ["col_{}".format(i) for i in range(len(X[0]))]
    X.ww.columns = col_names
    clf = CatBoostClassifier(
        n_estimators=1,
        max_depth=1,
        random_seed=SEED_BOUNDS.min_bound,
    )
    clf.fit(X, y)
    clf = CatBoostClassifier(
        n_estimators=1,
        max_depth=1,
        random_seed=SEED_BOUNDS.max_bound,
    )
    fitted = clf.fit(X, y)
    assert isinstance(fitted, CatBoostClassifier)


def test_catboost_classifier_init_n_jobs():
    n_jobs = 2
    clf = CatBoostClassifier(n_jobs=n_jobs)
    assert clf._component_obj.get_param("thread_count") == n_jobs


def test_catboost_classifier_init_thread_count():
    with warnings.catch_warnings(record=True) as w:
        CatBoostClassifier(thread_count=2)
    assert len(w) == 1
    assert "Parameter 'thread_count' will be ignored. " in str(w[-1].message)


def test_catboost_classifier_double_categories_in_X():
    X = pd.DataFrame({"double_cats": pd.Series([1.0, 2.0, 3.0, 4.0, 5.0] * 20)})
    y = pd.Series(range(100))
    y.ww.init()
    X.ww.init(logical_types={"double_cats": "Categorical"})

    clf = CatBoostClassifier()
    clf.fit(X, y)


def test_catboost_classifier_double_categories_in_y():
    X = pd.DataFrame({"cats": pd.Series([1, 2, 3, 4, 5] * 20)})
    X.ww.init(logical_types={"cats": "Categorical"})
    # --> this was never causing errors it seems - test more though to confirm!!
    y = pd.Series(
        [1.0, 2.0, 3.0, 4.0, 5.0] * 20,
    )
    ww.init_series(y, logical_type="Categorical")

    clf = CatBoostClassifier()
    fitted = clf.fit(X, y)
    assert isinstance(fitted, CatBoostClassifier)
