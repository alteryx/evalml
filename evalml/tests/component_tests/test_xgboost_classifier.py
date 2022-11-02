import string
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from evalml.pipelines.components import XGBoostClassifier
from evalml.problem_types import ProblemTypes
from evalml.utils import SEED_BOUNDS, get_random_state


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
    X.ww.columns = col_names
    clf = XGBoostClassifier(
        n_estimators=1,
        max_depth=1,
        random_seed=SEED_BOUNDS.min_bound,
    )
    fitted = clf.fit(X, y)
    assert isinstance(fitted, XGBoostClassifier)
    clf = XGBoostClassifier(
        n_estimators=1,
        max_depth=1,
        random_seed=SEED_BOUNDS.max_bound,
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


@patch("xgboost.XGBClassifier.fit")
@patch("xgboost.XGBClassifier.predict")
@patch("xgboost.XGBClassifier.predict_proba")
def test_xgboost_preserves_schema_in_rename(mock_predict_proba, mock_predict, mock_fit):
    X = pd.DataFrame({"a": [1, 2, 3, 4]})
    X.ww.init(logical_types={"a": "NaturalLanguage"})
    original_schema = X.ww.rename(columns={"a": 0}).ww.schema

    xgb = XGBoostClassifier()
    xgb.fit(X, pd.Series([0, 1, 1, 0]))
    assert mock_fit.call_args[0][0].ww.schema == original_schema
    xgb.predict(X)
    assert mock_predict.call_args[0][0].ww.schema == original_schema
    xgb.predict_proba(X)
    assert mock_predict_proba.call_args[0][0].ww.schema == original_schema
