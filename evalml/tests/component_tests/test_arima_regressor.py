import numpy as np
import pandas as pd
import pytest
from pytest import importorskip

from evalml.model_family import ModelFamily
from evalml.pipelines.components import ARIMARegressor
from evalml.problem_types import ProblemTypes

arima = importorskip('statsmodels.tsa.arima_model', reason='Skipping test because ARIMA not installed')


def test_model_family():
    assert ARIMARegressor.model_family == ModelFamily.ARIMA


def test_problem_types():
    assert set(ARIMARegressor.supported_problem_types) == {ProblemTypes.TIME_SERIES_REGRESSION}


def test_feature_importance(ts_data):
    X, y = ts_data
    clf = ARIMARegressor()
    clf.fit(X, y)
    clf.feature_importance == np.zeros(1)


def test_fit_predict_ts_with_X_index(ts_data):
    X, y = ts_data
    assert isinstance(X.index, pd.DatetimeIndex)

    a_clf = arima.ARIMA(endog=y, exog=X, order=(1, 0, 0), dates=X.index)
    a_clf.fit(solver='nm')
    y_pred_a = a_clf.predict(params=(1, 0, 0))

    clf = ARIMARegressor(p=1, d=0, q=0)
    clf.fit(X=X, y=y)
    y_pred = clf.predict(X=X, y=y)

    assert (y_pred == y_pred_a).all()


def test_fit_predict_ts_no_X(ts_data):
    X, y = ts_data

    a_clf = arima.ARIMA(endog=y, order=(1, 0, 0), dates=y.index)
    a_clf.fit(solver='nm')
    y_pred_a = a_clf.predict(params=(1, 0, 0))

    clf = ARIMARegressor(p=1, d=0, q=0)
    clf.fit(X=None, y=y)
    y_pred = clf.predict(X=None, y=y)

    assert (y_pred == y_pred_a).all()


def test_fit_predict_date_col(ts_data):
    X, y = ts_data

    a_clf = arima.ARIMA(endog=y, exog=X, order=(1, 0, 0), dates=X.index)
    a_clf.fit(solver='nm')
    y_pred_a = a_clf.predict(params=(1, 0, 0))

    X = X.reset_index()
    clf = ARIMARegressor(p=1, d=0, q=0, date_column='index')
    clf.fit(X=X, y=y)
    y_pred = clf.predict(X=X, y=y)

    assert (y_pred == y_pred_a).all()


def test_fit_predict_no_date_col_or_index(ts_data):
    X, y = ts_data
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    clf = ARIMARegressor()
    with pytest.raises(ValueError):
        clf.fit(X, y)
