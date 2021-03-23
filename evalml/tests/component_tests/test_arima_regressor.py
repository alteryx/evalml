import numpy as np
import pandas as pd
import pytest
from pytest import importorskip

from evalml.model_family import ModelFamily
from evalml.pipelines.components import ARIMARegressor
from evalml.problem_types import ProblemTypes

arima = importorskip('statsmodels.tsa.arima.model', reason='Skipping test because is ARIMA not installed')


def test_model_family():
    assert ARIMARegressor.model_family == ModelFamily.ARIMA


def test_problem_types():
    assert set(ARIMARegressor.supported_problem_types) == {ProblemTypes.TIME_SERIES_REGRESSION}


def test_feature_importance(ts_data):
    X, y = ts_data
    clf = ARIMARegressor()
    clf.fit(X, y)
    clf.feature_importance == np.zeros(1)


def test_fit_predict_ts_with_X_and_y_index(ts_data):
    X, y = ts_data
    assert isinstance(X.index, pd.DatetimeIndex)
    assert isinstance(y.index, pd.DatetimeIndex)

    a_clf = arima.ARIMA(endog=y, exog=X, order=(1, 0, 0), trend='n', dates=X.index)
    clf = a_clf.fit()
    y_pred_a = clf.predict()

    m_clf = ARIMARegressor(p=1, d=0, q=0)
    clf_ = m_clf.fit(X=X, y=y)
    y_pred = clf_.predict(X=X)

    assert (y_pred == y_pred_a).all()


def test_fit_predict_ts_with_X_not_y_index(ts_data):
    X, y = ts_data
    assert isinstance(X.index, pd.DatetimeIndex)
    assert isinstance(y.index, pd.DatetimeIndex)

    a_clf = arima.ARIMA(endog=y, exog=X, order=(1, 0, 0), trend='n', dates=X.index)
    clf = a_clf.fit()
    y_pred_a = clf.predict()

    y = y.reset_index(drop=True)
    assert not isinstance(y.index, pd.DatetimeIndex)

    m_clf = ARIMARegressor(p=1, d=0, q=0)
    clf_ = m_clf.fit(X=X, y=y)
    y_pred = clf_.predict(X=X)

    assert (y_pred == y_pred_a).all()


def test_fit_predict_ts_with_y_not_X_index_with_iterable_p_and_q(ts_data):
    X, y = ts_data

    a_clf = arima.ARIMA(endog=y, exog=X, order=([1, 2], 0, [1, 2]), trend='n', dates=X.index)
    clf = a_clf.fit()
    y_pred_a = clf.predict(params=(1, 0, 0))

    X_no_ind = X.reset_index(drop=True)
    assert isinstance(y.index, pd.DatetimeIndex)
    assert not isinstance(X_no_ind.index, pd.DatetimeIndex)

    m_clf = ARIMARegressor(p=[1, 2], d=0, q=[1, 2])
    clf_ = m_clf.fit(X=X_no_ind, y=y)
    y_pred = clf_.predict(X=X, y=y)

    assert (y_pred == y_pred_a).all()


def test_predict_ts_without_X(ts_data):
    X, y = ts_data

    a_clf = arima.ARIMA(endog=y, exog=X, order=(1, 0, 0), trend='n', dates=X.index)
    clf = a_clf.fit()
    y_pred_a = clf.predict(params=(1, 0, 0))

    m_clf = ARIMARegressor(p=1, d=0, q=0)
    clf_ = m_clf.fit(X=X, y=y)
    y_pred = clf_.predict(y=y)

    assert (y_pred == y_pred_a).all()


def test_fit_ts_with_not_X_not_y_index(ts_data):
    X, y = ts_data
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    assert not isinstance(y.index, pd.DatetimeIndex)
    assert not isinstance(X.index, pd.DatetimeIndex)

    clf = ARIMARegressor(p=1, d=0, q=0)
    with pytest.raises(ValueError, match="If not it will look for the datetime column in the index of X or y."):
        clf.fit(X=X, y=y)


def test_predict_ts_with_not_X_index(ts_data):
    X, y = ts_data
    X = X.reset_index(drop=True)
    assert not isinstance(X.index, pd.DatetimeIndex)

    m_clf = ARIMARegressor(p=1, d=0, q=0)
    clf_ = m_clf.fit(X=X, y=y)
    with pytest.raises(ValueError, match="If not it will look for the datetime column in the index of X."):
        clf_.predict(X)


def test_fit_ts_without_y(ts_data):
    X, y = ts_data

    clf = ARIMARegressor(p=1, d=0, q=0)
    with pytest.raises(ValueError, match="ARIMA Regressor requires y as input."):
        clf.fit(X=X)


def test_fit_predict_ts_no_X(ts_data):
    X, y = ts_data

    a_clf = arima.ARIMA(endog=y, order=(1, 0, 0), trend='n', dates=y.index)
    clf = a_clf.fit()
    y_pred_a = clf.predict(params=(1, 0, 0))

    m_clf = ARIMARegressor(p=1, d=0, q=0)
    clf_ = m_clf.fit(X=None, y=y)
    y_pred = clf_.predict(X=X, y=y)

    assert (y_pred == y_pred_a).all()


def test_fit_predict_date_col_named(ts_data):
    X, y = ts_data

    a_clf = arima.ARIMA(endog=y, exog=X, order=(1, 0, 0), trend='n', dates=X.index)
    clf = a_clf.fit()
    y_pred_a = clf.predict(params=(1, 0, 0))

    X = X.reset_index()
    assert not isinstance(X.index, pd.DatetimeIndex)
    m_clf = ARIMARegressor(p=1, d=0, q=0, date_column='index')
    clf_ = m_clf.fit(X=X, y=y)
    y_pred = clf_.predict(X=X, y=y)

    assert (y_pred == y_pred_a).all()
