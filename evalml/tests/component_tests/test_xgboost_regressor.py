import string
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from evalml.pipelines.components import XGBoostRegressor
from evalml.utils import SEED_BOUNDS, get_random_state


def test_xgboost_regressor_random_seed_bounds_seed(X_y_regression):
    """ensure xgboost's RNG doesn't fail for the min/max bounds we support on user-inputted random seeds"""
    X, y = X_y_regression
    col_names = ["col_{}".format(i) for i in range(len(X[0]))]
    X.ww.columns = col_names
    clf = XGBoostRegressor(
        n_estimators=1,
        max_depth=1,
        random_seed=SEED_BOUNDS.min_bound,
    )
    fitted = clf.fit(X, y)
    assert isinstance(fitted, XGBoostRegressor)
    clf = XGBoostRegressor(
        n_estimators=1,
        max_depth=1,
        random_seed=SEED_BOUNDS.max_bound,
    )
    clf.fit(X, y)


def test_xgboost_feature_name_with_random_ascii(X_y_regression):
    X, y = X_y_regression
    clf = XGBoostRegressor()
    X = get_random_state(clf.random_seed).random((X.shape[0], len(string.printable)))
    col_names = ["column_{}".format(ascii_char) for ascii_char in string.printable]
    X = pd.DataFrame(X, columns=col_names)
    clf.fit(X, y)
    predictions = clf.predict(X)
    assert len(predictions) == len(y)
    assert not np.isnan(predictions).all()

    assert len(clf.feature_importance) == len(X.columns)
    assert not np.isnan(clf.feature_importance).all().all()


@pytest.mark.parametrize("data_type", ["pd", "ww"])
def test_xgboost_multiindex(data_type, X_y_regression, make_data_type):
    X, y = X_y_regression
    X = pd.DataFrame(X)
    col_names = [
        ("column_{}".format(num), "{}".format(num)) for num in range(len(X.columns))
    ]
    X.columns = pd.MultiIndex.from_tuples(col_names)
    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)

    clf = XGBoostRegressor()
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert not y_pred.isnull().values.any()


def test_xgboost_predict_all_boolean_columns():
    X = pd.DataFrame({"a": [True, False, True], "b": [True, False, True]})
    y = pd.Series([2, 3, 4])
    xgb = XGBoostRegressor()
    xgb.fit(X, y)
    preds = xgb.predict(X)
    assert isinstance(preds, pd.Series)
    assert not preds.isna().any()


@patch("xgboost.XGBRegressor.fit")
@patch("xgboost.XGBRegressor.predict")
def test_xgboost_preserves_schema_in_rename(mock_predict, mock_fit):
    X = pd.DataFrame({"a": [1, 2, 3, 4]})
    X.ww.init(logical_types={"a": "NaturalLanguage"})
    original_schema = X.ww.rename(columns={"a": 0}).ww.schema

    xgb = XGBoostRegressor()
    xgb.fit(X, pd.Series([0, 1, 1, 0]))
    assert mock_fit.call_args[0][0].ww.schema == original_schema
    xgb.predict(X)
    assert mock_predict.call_args[0][0].ww.schema == original_schema
