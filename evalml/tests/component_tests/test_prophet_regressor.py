import pytest
from pytest import importorskip

from evalml.model_family import ModelFamily
from evalml.pipelines.components import ProphetRegressor
from evalml.problem_types import ProblemTypes
from evalml.utils.gen_utils import suppress_stdout_stderr

prophet = importorskip('fbprophet', reason='Skipping test because xgboost not installed')


def test_model_family():
    assert ProphetRegressor.model_family == ModelFamily.PROPHET


def test_problem_types():
    assert set(ProphetRegressor.supported_problem_types) == {ProblemTypes.TIME_SERIES_REGRESSION}


def test_fit_predict_ts(ts_data):
    X, y = ts_data

    def build_prophet_df(X, y=None):
        date_col = X.reset_index()
        date_col = date_col['index']
        date_col = date_col.rename('ds')
        prophet_df = date_col.to_frame()
        y.index = prophet_df.index
        prophet_df['y'] = y
        return prophet_df

    p_clf = prophet.Prophet()
    prophet_df = build_prophet_df(X, y)

    with suppress_stdout_stderr():
        p_clf.fit(prophet_df)
    y_pred_p = p_clf.predict(prophet_df)['yhat']

    clf = ProphetRegressor()
    clf.fit(X, y)
    y_pred = clf.predict(X)

    assert (y_pred == y_pred_p).all()


def test_fit_predict_ds_col(ts_data):
    X, y = ts_data

    def build_prophet_df(X, y=None):
        date_col = X.reset_index()
        date_col = date_col['index']
        date_col = date_col.rename('ds')
        prophet_df = date_col.to_frame()
        y.index = prophet_df.index
        prophet_df['y'] = y
        return prophet_df

    p_clf = prophet.Prophet()
    prophet_df = build_prophet_df(X, y)

    with suppress_stdout_stderr():
        p_clf.fit(prophet_df)
    y_pred_p = p_clf.predict(prophet_df)['yhat']

    X = X.reset_index()
    X = X['index'].rename('ds').to_frame()
    clf = ProphetRegressor()
    clf.fit(X, y)
    y_pred = clf.predict(X)

    assert (y_pred == y_pred_p).all()


def test_fit_predict_no_ds(X_y_binary):
    X, y = X_y_binary

    clf = ProphetRegressor()
    with pytest.raises(ValueError):
        clf.fit(X, y)
