import string

import numpy as np
import pandas as pd
import pytest
from pytest import importorskip

from evalml.pipelines.components import XGBoostClassifier
from evalml.problem_types import ProblemTypes
from evalml.utils import SEED_BOUNDS, get_random_state

xgb = importorskip('xgboost', reason='Skipping test because xgboost not installed')


def test_xgboost_classifier_random_state_bounds_seed(X_y_binary):
    """ensure xgboost's RNG doesn't fail for the min/max bounds we support on user-inputted random seeds"""
    X, y = X_y_binary
    col_names = ["col_{}".format(i) for i in range(len(X[0]))]
    X = pd.DataFrame(X, columns=col_names)
    y = pd.Series(y)
    clf = XGBoostClassifier(n_estimators=1, max_depth=1, random_state=SEED_BOUNDS.min_bound)
    fitted = clf.fit(X, y)
    assert isinstance(fitted, XGBoostClassifier)
    clf = XGBoostClassifier(n_estimators=1, max_depth=1, random_state=SEED_BOUNDS.max_bound)
    clf.fit(X, y)


@pytest.mark.parametrize("problem_type", [ProblemTypes.BINARY, ProblemTypes.MULTICLASS])
def test_xgboost_feature_name_with_random_ascii(problem_type, X_y_binary, X_y_multi):
    clf = XGBoostClassifier()
    if problem_type == ProblemTypes.BINARY:
        X, y = X_y_binary
        expected_cols = 2

    elif problem_type == ProblemTypes.MULTICLASS:
        X, y = X_y_multi
        expected_cols = 3

    X = get_random_state(clf.random_state).random((X.shape[0], len(string.printable)))
    col_names = ['column_{}'.format(ascii_char) for ascii_char in string.printable]
    X = pd.DataFrame(X, columns=col_names)

    clf.fit(X, y)
    predictions = clf.predict(X)
    assert len(predictions) == len(y)
    assert not np.isnan(predictions.to_series()).all()

    predictions = clf.predict_proba(X)
    assert predictions.shape == (len(y), expected_cols)
    assert not np.isnan(predictions.to_dataframe()).all().all()

    assert len(clf.feature_importance) == len(X.columns)
    assert not np.isnan(clf.feature_importance).all().all()
