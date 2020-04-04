import numpy as np
import pandas as pd
from pytest import importorskip

from evalml.pipelines.components import CatBoostClassifier
from evalml.utils import SEED_BOUNDS, get_random_seed, get_random_state

importorskip('catboost', reason='Skipping test because catboost not installed')


def test_catboost_random_state_bounds(X_y):
    X, y = X_y
    col_names = ["col_{}".format(i) for i in range(len(X[0]))]
    X = pd.DataFrame(X, columns=col_names)
    y = pd.Series(y)
    # ensure the RNG used in catboost doesn't throw for the min and max seed values, as int and RNG
    clf = CatBoostClassifier(n_estimators=1, max_depth=1, random_state=CatBoostClassifier.SEED_MIN)
    clf.fit(X, y)
    clf = CatBoostClassifier(n_estimators=1, max_depth=1, random_state=CatBoostClassifier.SEED_MAX)
    clf.fit(X, y)
    rng = np.random.RandomState(CatBoostClassifier.SEED_MIN)
    clf = CatBoostClassifier(n_estimators=1, max_depth=1, random_state=rng)
    clf.fit(X, y)
    rng = np.random.RandomState(CatBoostClassifier.SEED_MAX)
    clf = CatBoostClassifier(n_estimators=1, max_depth=1, random_state=rng)
    clf.fit(X, y)
