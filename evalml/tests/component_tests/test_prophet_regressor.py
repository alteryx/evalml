import numpy as np
import pandas as pd
import pytest
from pytest import importorskip

from evalml.model_family import ModelFamily
from evalml.pipelines.components import ProphetRegressor
from evalml.problem_types import ProblemTypes

prophet = importorskip("prophet", reason="Skipping test because prophet not installed")


def test_model_family():
    assert ProphetRegressor.model_family == ModelFamily.PROPHET


def test_cmdstanpy_backend():
    m = prophet.Prophet(stan_backend="CMDSTANPY")
    assert m.stan_backend.get_type() == "CMDSTANPY"


def test_problem_types():
    assert set(ProphetRegressor.supported_problem_types) == {
        ProblemTypes.TIME_SERIES_REGRESSION
    }


def test_init_with_other_params():
    clf = ProphetRegressor(
        daily_seasonality=True,
        mcmc_samples=5,
        interval_width=0.8,
        uncertainty_samples=0,
    )
    assert clf.parameters == {
        "changepoint_prior_scale": 0.05,
        "daily_seasonality": True,
        "date_index": None,
        "holidays_prior_scale": 10,
        "interval_width": 0.8,
        "mcmc_samples": 5,
        "seasonality_mode": "additive",
        "seasonality_prior_scale": 10,
        "uncertainty_samples": 0,
        "stan_backend": "CMDSTANPY",
    }


def test_feature_importance(ts_data):
    X, y = ts_data
    clf = ProphetRegressor(uncertainty_samples=False, changepoint_prior_scale=2.0)
    clf.fit(X, y)
    clf.feature_importance == np.zeros(1)


def test_get_params(ts_data):
    clf = ProphetRegressor()
    assert clf.get_params() == {
        "changepoint_prior_scale": 0.05,
        "date_index": None,
        "seasonality_prior_scale": 10,
        "holidays_prior_scale": 10,
        "seasonality_mode": "additive",
        "stan_backend": "CMDSTANPY",
    }


def test_fit_predict_ts_with_X_index(ts_data):
    X, y = ts_data
    assert isinstance(X.index, pd.DatetimeIndex)

    p_clf = prophet.Prophet(uncertainty_samples=False, changepoint_prior_scale=2.0)
    prophet_df = ProphetRegressor.build_prophet_df(X=X, y=y, date_column="ds")

    p_clf.fit(prophet_df)
    y_pred_p = p_clf.predict(prophet_df)["yhat"]

    clf = ProphetRegressor(uncertainty_samples=False, changepoint_prior_scale=2.0)
    clf.fit(X, y)
    y_pred = clf.predict(X)

    assert (y_pred == y_pred_p).all()


def test_fit_predict_ts_with_y_index(ts_data):
    X, y = ts_data
    X = X.reset_index(drop=True)
    assert isinstance(y.index, pd.DatetimeIndex)

    p_clf = prophet.Prophet(uncertainty_samples=False, changepoint_prior_scale=2.0)
    prophet_df = ProphetRegressor.build_prophet_df(X=X, y=y, date_column="ds")

    p_clf.fit(prophet_df)
    y_pred_p = p_clf.predict(prophet_df)["yhat"]

    clf = ProphetRegressor(uncertainty_samples=False, changepoint_prior_scale=2.0)
    clf.fit(X, y)
    y_pred = clf.predict(X, y)

    assert (y_pred == y_pred_p).all()


def test_fit_predict_ts_no_X(ts_data):
    y = pd.Series(
        range(1, 32), name="dates", index=pd.date_range("2020-10-01", "2020-10-31")
    )

    p_clf = prophet.Prophet(uncertainty_samples=False, changepoint_prior_scale=2.0)
    prophet_df = ProphetRegressor.build_prophet_df(
        X=pd.DataFrame(), y=y, date_column="ds"
    )
    p_clf.fit(prophet_df)
    y_pred_p = p_clf.predict(prophet_df)["yhat"]

    clf = ProphetRegressor(uncertainty_samples=False, changepoint_prior_scale=2.0)
    clf.fit(X=None, y=y)
    y_pred = clf.predict(X=None, y=y)

    assert (y_pred == y_pred_p).all()


def test_fit_predict_date_col(ts_data):
    X = pd.DataFrame(
        {
            "features": range(100),
            "these_dates": pd.date_range("1/1/21", periods=100),
            "more_dates": pd.date_range("7/4/1987", periods=100),
        }
    )
    y = pd.Series(np.random.randint(1, 5, 100), name="y")

    clf = ProphetRegressor(
        date_index="these_dates", uncertainty_samples=False, changepoint_prior_scale=2.0
    )
    clf.fit(X, y)
    y_pred = clf.predict(X)

    p_clf = prophet.Prophet(uncertainty_samples=False, changepoint_prior_scale=2.0)
    prophet_df = ProphetRegressor.build_prophet_df(X=X, y=y, date_column="these_dates")

    p_clf.fit(prophet_df)
    y_pred_p = p_clf.predict(prophet_df)["yhat"]

    assert (y_pred == y_pred_p).all()


def test_fit_predict_no_date_col_or_index(ts_data):
    X, y = ts_data
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    assert not isinstance(X.index, pd.DatetimeIndex)
    assert not isinstance(y.index, pd.DatetimeIndex)

    clf = ProphetRegressor()
    with pytest.raises(
        ValueError,
        match="Prophet estimator requires input data X to have a datetime column",
    ):
        clf.fit(X, y)
