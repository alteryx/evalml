import numpy as np
from pytest import importorskip

from evalml.model_family import ModelFamily
from evalml.pipelines.components import ARIMARegressor
from evalml.problem_types import ProblemTypes

arima = importorskip('statsmodels.tsa.arima_model', reason='Skipping test because ARIMA not installed')

def test_model_family():
    assert ARIMARegressor.model_family == ModelFamily.ARIMA


def test_problem_types():
    assert set(ARIMARegressor.supported_problem_types) == {ProblemTypes.TIME_SERIES_REGRESSION}


# def test_init_with_other_params():
#     clf = ProphetRegressor(daily_seasonality=True, mcmc_samples=5, interval_width=0.8, uncertainty_samples=0)
#     assert clf.parameters == {'changepoint_prior_scale': 0.05,
#                               'daily_seasonality': True,
#                               'holidays_prior_scale': 10,
#                               'interval_width': 0.8,
#                               'mcmc_samples': 5,
#                               'seasonality_mode': 'additive',
#                               'seasonality_prior_scale': 10,
#                               'uncertainty_samples': 0}


def test_feature_importance(ts_data):
    X, y = ts_data
    clf = ARIMARegressor()
    clf.fit(X, y)
    clf.feature_importance == np.zeros(1)


# def test_fit_predict_ts_with_X_index(ts_data):
#     X, y = ts_data
#     assert isinstance(X.index, pd.DatetimeIndex)

#     p_clf = prophet.Prophet()
#     prophet_df = ProphetRegressor.build_prophet_df(X=X, y=y, date_column='ds')

#     with suppress_stdout_stderr():
#         p_clf.fit(prophet_df)
#     y_pred_p = p_clf.predict(prophet_df)['yhat']

#     clf = ProphetRegressor()
#     clf.fit(X, y)
#     y_pred = clf.predict(X)

#     assert (y_pred == y_pred_p).all()


# def test_fit_predict_ts_with_y_index(ts_data):
#     X, y = ts_data
#     X = X.reset_index(drop=True)
#     assert isinstance(y.index, pd.DatetimeIndex)

#     p_clf = prophet.Prophet()
#     prophet_df = ProphetRegressor.build_prophet_df(X=X, y=y, date_column='ds')

#     with suppress_stdout_stderr():
#         p_clf.fit(prophet_df)
#     y_pred_p = p_clf.predict(prophet_df)['yhat']

#     clf = ProphetRegressor()
#     clf.fit(X, y)
#     y_pred = clf.predict(X, y)

#     assert (y_pred == y_pred_p).all()


def test_fit_predict_ts_no_X(ts_data):
    X, y = ts_data

    a_clf = arima.ARIMA(endog=y, order=(1, 0, 0), dates=y.index)
    a_clf.fit(solver='nm')
    y_pred_a = a_clf.predict(params=(1, 0, 0))

    clf = ARIMARegressor(p=1, d=0, q=0)
    clf.fit(X=None, y=y)
    y_pred = clf.predict(X=None, y=y)

    assert (y_pred == y_pred_a).all()


# def test_fit_predict_date_col(ts_data):
#     X, y = ts_data

#     p_clf = prophet.Prophet()
#     prophet_df = ProphetRegressor.build_prophet_df(X=X, y=y, date_column='ds')

#     with suppress_stdout_stderr():
#         p_clf.fit(prophet_df)
#     y_pred_p = p_clf.predict(prophet_df)['yhat']

#     X = X.reset_index()
#     X = X['index'].rename('ds').to_frame()
#     clf = ProphetRegressor(date_column='ds')
#     clf.fit(X, y)
#     y_pred = clf.predict(X)

#     assert (y_pred == y_pred_p).all()


# def test_fit_predict_no_date_col_or_index(X_y_binary):
#     X, y = X_y_binary

#     clf = ProphetRegressor()
#     with pytest.raises(ValueError):
#         clf.fit(X, y)
