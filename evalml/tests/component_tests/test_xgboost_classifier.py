import string
import warnings

import numpy as np
import pandas as pd
import pytest
from pytest import importorskip

from evalml.pipelines.components import XGBoostClassifier
from evalml.problem_types import ProblemTypes
from evalml.utils import SEED_BOUNDS, get_random_state

xgb = importorskip("xgboost", reason="Skipping test because xgboost not installed")


@pytest.mark.parametrize("metric", ["error", "logloss"])
def test_xgboost_classifier_default_evaluation_metric(metric):
    xgb = XGBoostClassifier(eval_metric=metric)
    assert xgb.parameters["eval_metric"] == metric
    assert xgb._component_obj.get_params()["eval_metric"] == metric

    xgb = XGBoostClassifier()
    assert xgb.parameters["eval_metric"] == "logloss"
    assert xgb._component_obj.get_params()["eval_metric"] == "logloss"


def test_xgboost_classifier_random_seed_bounds_seed(X_y_binary):
    """ensure xgboost's RNG doesn't fail for the min/max bounds we support on user-inputted random seeds"""
    X, y = X_y_binary
    col_names = ["col_{}".format(i) for i in range(len(X[0]))]
    X = pd.DataFrame(X, columns=col_names)
    y = pd.Series(y)
    clf = XGBoostClassifier(
        n_estimators=1, max_depth=1, random_seed=SEED_BOUNDS.min_bound
    )
    fitted = clf.fit(X, y)
    assert isinstance(fitted, XGBoostClassifier)
    clf = XGBoostClassifier(
        n_estimators=1, max_depth=1, random_seed=SEED_BOUNDS.max_bound
    )
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

    X = get_random_state(clf.random_seed).random((X.shape[0], len(string.printable)))
    col_names = ["column_{}".format(ascii_char) for ascii_char in string.printable]
    X = pd.DataFrame(X, columns=col_names)

    clf.fit(X, y)
    predictions = clf.predict(X)
    assert len(predictions) == len(y)
    assert not np.isnan(predictions).all()

    predictions = clf.predict_proba(X)
    assert predictions.shape == (len(y), expected_cols)
    assert not np.isnan(predictions).all().all()

    assert len(clf.feature_importance) == len(X.columns)
    assert not np.isnan(clf.feature_importance).all().all()


@pytest.mark.parametrize("data_type", ["pd", "ww"])
def test_xgboost_multiindex(data_type, X_y_binary, make_data_type):
    X, y = X_y_binary
    X = pd.DataFrame(X)
    col_names = [
        ("column_{}".format(num), "{}".format(num)) for num in range(len(X.columns))
    ]
    X.columns = pd.MultiIndex.from_tuples(col_names)
    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)

    clf = XGBoostClassifier()
    clf.fit(X, y)
    y_pred = clf.predict(X)
    y_pred_proba = clf.predict_proba(X)
    assert not y_pred.isnull().values.any()
    assert not y_pred_proba.isnull().values.any().any()


def test_xgboost_predict_all_boolean_columns():
    X = pd.DataFrame({"a": [True, False, True], "b": [True, False, True]})
    y = pd.Series([True, False, False])
    xgb = XGBoostClassifier()
    xgb.fit(X, y)
    preds = xgb.predict(X)
    assert isinstance(preds, pd.Series)
    assert not preds.isna().any()


@pytest.mark.parametrize(
    "y,label_encoder",
    [
        ([True, False, False], True),
        ([1.0, 1.1, 1.1], True),
        (["One", "Two", "Two"], True),
        ([0, 1, 1], False),
    ],
)
def test_xgboost_catch_warnings_label_encoder(y, label_encoder):
    X = pd.DataFrame({"a": [True, False, True], "b": [True, False, True]})
    y = pd.Series(y)
    xgb = XGBoostClassifier()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        xgb.fit(X, y)
    assert len(w) == 0
    preds = xgb.predict(X)
    # make sure the predicted outputs are the same labels as the passed-in labels
    assert preds[0] in y.values
    if label_encoder:
        assert xgb._label_encoder is not None
        return
    assert xgb._label_encoder is None
