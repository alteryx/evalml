from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pytest import importorskip

from evalml.model_family import ModelFamily
from evalml.pipelines.components import ExponentialSmoothingRegressor
from evalml.problem_types import ProblemTypes

sktime_exp = importorskip(
    "sktime.forecasting.exp_smoothing", reason="Skipping test because sktime not installed"
)
forecasting = importorskip(
    "sktime.forecasting.base", reason="Skipping test because sktime not installed"
)


def test_model_family():
    assert ExponentialSmoothingRegressor().model_family == ModelFamily.EXPONENTIAL_SMOOTHING


def test_problem_types():
    assert set(ExponentialSmoothingRegressor().supported_problem_types) == {
        ProblemTypes.TIME_SERIES_REGRESSION
    }


def test_model_instance(ts_data):
    X, y = ts_data
    clf = ExponentialSmoothingRegressor()
    fitted = clf.fit(X, y)
    assert isinstance(fitted, ExponentialSmoothingRegressor)


def test_feature_importance(ts_data):
    X, y = ts_data
    clf = ExponentialSmoothingRegressor()
    with patch.object(clf, "_component_obj"):
        clf.fit(X, y)
        assert clf.feature_importance == np.zeros(1)


def test_fit_predict_ts_with_datetime_in_X_column(
    ts_data_seasonal_train, ts_data_seasonal_test
):
    X, y = ts_data_seasonal_train
    X_test, y_test = ts_data_seasonal_test
    assert isinstance(X.index, pd.DatetimeIndex)
    assert isinstance(y.index, pd.DatetimeIndex)

    fh = forecasting.ForecastingHorizon(
        [i + 1 for i in range(len(y_test))], is_relative=True
    )

    m_clf = ExponentialSmoothingRegressor(forecast_horizon=fh)
    m_clf.fit(X=X, y=y)
    y_pred = m_clf.predict(X=X_test)

    X["Sample"] = pd.date_range(start="1/1/2016", periods=25)

    dt_clf = ExponentialSmoothingRegressor(forecast_horizon=fh)
    dt_clf.fit(X=X, y=y)
    y_pred_dt = dt_clf.predict(X=X_test)

    assert isinstance(y_pred_dt, pd.Series)
    pd.testing.assert_series_equal(y_pred, y_pred_dt)


def test_fit_predict_ts_with_only_datetime_column_in_X(
    ts_data_seasonal_train, ts_data_seasonal_test
):
    X, y = ts_data_seasonal_train
    X_test, y_test = ts_data_seasonal_test
    assert isinstance(X.index, pd.DatetimeIndex)
    assert isinstance(y.index, pd.DatetimeIndex)

    fh_ = forecasting.ForecastingHorizon(
        [i + 1 for i in range(len(y_test))], is_relative=True
    )

    a_clf = sktime_exp.ExponentialSmoothing()
    clf = a_clf.fit(y=y)
    y_pred_sk = clf.predict(fh=fh_)

    X = X.drop(["features"], axis=1)

    m_clf = ExponentialSmoothingRegressor(forecast_horizon=fh_)
    m_clf.fit(X=X, y=y)
    y_pred = m_clf.predict(X=X_test)

    assert (y_pred_sk == y_pred).all()


def test_fit_predict_ts_with_X_and_y_index_out_of_sample(
    ts_data_seasonal_train, ts_data_seasonal_test
):
    X, y = ts_data_seasonal_train
    X_test, y_test = ts_data_seasonal_test
    assert isinstance(X.index, pd.DatetimeIndex)
    assert isinstance(y.index, pd.DatetimeIndex)

    fh_ = forecasting.ForecastingHorizon(
        [i + 1 for i in range(len(y_test))], is_relative=True
    )

    a_clf = sktime_exp.ExponentialSmoothing()
    clf = a_clf.fit(X=X, y=y)
    y_pred_sk = clf.predict(fh=fh_, X=X_test)

    m_clf = ExponentialSmoothingRegressor(forecast_horizon=fh_)
    m_clf.fit(X=X, y=y)
    y_pred = m_clf.predict(X=X_test)

    assert (y_pred_sk == y_pred).all()


def test_fit_predict_ts_with_X_and_y_index(ts_data_seasonal_train):
    X, y = ts_data_seasonal_train
    assert isinstance(X.index, pd.DatetimeIndex)
    assert isinstance(y.index, pd.DatetimeIndex)


    fh_ = forecasting.ForecastingHorizon(
        [i + 1 for i in range(len(y))], is_relative=True
    )

    a_clf = sktime_exp.ExponentialSmoothing()
    clf = a_clf.fit(X=X, y=y)
    y_pred_sk = clf.predict(fh=fh_, X=X)

    m_clf = ExponentialSmoothingRegressor(forecast_horizon=fh_)
    m_clf.fit(X=X, y=y)
    y_pred = m_clf.predict(X=X)

    assert (y_pred_sk == y_pred).all()


@patch("sktime.forecasting.base._sktime.BaseForecaster.predict")
@patch("sktime.forecasting.base._sktime.BaseForecaster.fit")
def test_predict_ts_X_error(mock_sktime_fit, mock_sktime_predict, ts_data):
    X, y = ts_data

    mock_sktime_predict.side_effect = ValueError("Sktime value error")

    m_clf = ExponentialSmoothingRegressor()
    clf_ = m_clf.fit(X=X, y=y)
    with pytest.raises(ValueError, match="Sktime value error"):
        clf_.predict(y=y)


def test_fit_ts_with_not_X_not_y_index(ts_data):
    X, y = ts_data
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    assert not isinstance(y.index, pd.DatetimeIndex)
    assert not isinstance(X.index, pd.DatetimeIndex)

    clf = ExponentialSmoothingRegressor()
    with pytest.raises(
        ValueError,
        match="If not it will look for the datetime column in the index of X or y.",
    ):
        clf.fit(X=X, y=y)


def test_predict_ts_with_not_X_index(ts_data):
    X, y = ts_data
    X = X.reset_index(drop=True)
    assert not isinstance(X.index, pd.DatetimeIndex)

    m_clf = ExponentialSmoothingRegressor()
    clf_ = m_clf.fit(X=X, y=y)
    with pytest.raises(
        ValueError,
        match="If not it will look for the datetime column in the index of X.",
    ):
        clf_.predict(X)


def test_fit_ts_without_y(ts_data):
    X, y = ts_data

    clf = ExponentialSmoothingRegressor()
    with pytest.raises(ValueError, match="Exponential Smoothing Regressor requires y as input."):
        clf.fit(X=X)


def test_fit_predict_ts_no_X_out_of_sample(
    ts_data_seasonal_train, ts_data_seasonal_test
):
    X, y = ts_data_seasonal_train
    X_test, y_test = ts_data_seasonal_test

    fh_ = forecasting.ForecastingHorizon(
        [i + 1 for i in range(len(y_test))], is_relative=True
    )

    a_clf = sktime_exp.ExponentialSmoothing()
    a_clf.fit(y=y)
    y_pred_sk = a_clf.predict(fh=fh_)

    m_clf = ExponentialSmoothingRegressor(d=None, forecast_horizon=fh_)
    m_clf.fit(X=None, y=y)
    y_pred = m_clf.predict(X=None, y=y_test)

    assert (y_pred_sk == y_pred).all()
