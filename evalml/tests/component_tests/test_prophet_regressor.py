import numpy as np
import pandas as pd
import pytest

from evalml.model_family import ModelFamily
from evalml.pipelines.components import ProphetRegressor
from evalml.problem_types import ProblemTypes

pytestmark = [
    pytest.mark.skip_during_conda,
]


def test_model_family():
    assert ProphetRegressor.model_family == ModelFamily.PROPHET


@pytest.fixture(scope="module")
def prophet():
    import prophet

    return prophet


def test_cmdstanpy_backend(prophet):

    m = prophet.Prophet(stan_backend="CMDSTANPY")
    assert m.stan_backend.get_type() == "CMDSTANPY"


def test_problem_types():
    assert set(ProphetRegressor.supported_problem_types) == {
        ProblemTypes.TIME_SERIES_REGRESSION,
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
        "time_index": None,
        "holidays_prior_scale": 10,
        "interval_width": 0.8,
        "mcmc_samples": 5,
        "seasonality_mode": "additive",
        "seasonality_prior_scale": 10,
        "uncertainty_samples": 0,
        "stan_backend": "CMDSTANPY",
    }


def test_feature_importance(ts_data):
    X, _, y = ts_data()
    clf = ProphetRegressor(
        time_index="date",
        uncertainty_samples=False,
        changepoint_prior_scale=2.0,
    )
    clf.fit(X, y)
    assert clf.feature_importance == np.zeros(1)


def test_get_params():
    clf = ProphetRegressor()
    assert clf.get_params() == {
        "changepoint_prior_scale": 0.05,
        "time_index": None,
        "seasonality_prior_scale": 10,
        "holidays_prior_scale": 10,
        "seasonality_mode": "additive",
        "stan_backend": "CMDSTANPY",
    }


@pytest.mark.parametrize("index_status", [None, "wrong_column"])
def test_build_prophet_df_time_index_errors(index_status, ts_data):
    X, _, y = ts_data()

    if index_status is None:
        with pytest.raises(ValueError, match="time_index cannot be None!"):
            ProphetRegressor.build_prophet_df(X, y, index_status)
    elif index_status == "wrong_column":
        with pytest.raises(
            ValueError,
            match=f"Column {index_status} was not found in X!",
        ):
            ProphetRegressor.build_prophet_df(X, y, index_status)


@pytest.mark.parametrize("drop_index", [None, "X", "y", "both"])
def test_fit_predict_ts(ts_data, drop_index, prophet):
    X, _, y = ts_data()
    if drop_index is None:
        assert isinstance(X.index, pd.DatetimeIndex)
        assert isinstance(y.index, pd.DatetimeIndex)
    elif drop_index == "X":
        X = X.reset_index(drop=True)
        assert not isinstance(X.index, pd.DatetimeIndex)
        assert isinstance(y.index, pd.DatetimeIndex)
    elif drop_index == "y":
        y = y.reset_index(drop=True)
        assert isinstance(X.index, pd.DatetimeIndex)
        assert not isinstance(y.index, pd.DatetimeIndex)
    elif drop_index == "both":
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        assert not isinstance(X.index, pd.DatetimeIndex)
        assert not isinstance(y.index, pd.DatetimeIndex)

    prophet_df = ProphetRegressor.build_prophet_df(X=X, y=y, time_index="date")
    p_clf = prophet.Prophet(uncertainty_samples=False, changepoint_prior_scale=2.0)
    p_clf.fit(prophet_df)
    y_pred_p = p_clf.predict(prophet_df)["yhat"]

    clf = ProphetRegressor(
        time_index="date",
        uncertainty_samples=False,
        changepoint_prior_scale=2.0,
    )
    clf.fit(X, y)
    y_pred = clf.predict(X)
    np.array_equal(y_pred_p.values, y_pred.values)
