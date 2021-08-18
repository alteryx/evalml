from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pytest import importorskip

from evalml.model_family import ModelFamily
from evalml.pipelines.components import ARIMARegressor
from evalml.problem_types import ProblemTypes

sktime_arima = importorskip(
    "sktime.forecasting.arima", reason="Skipping test because sktime not installed"
)
forecasting = importorskip(
    "sktime.forecasting.base", reason="Skipping test because sktime not installed"
)


def test_model_family():
    assert ARIMARegressor.model_family == ModelFamily.ARIMA


def test_problem_types():
    assert set(ARIMARegressor.supported_problem_types) == {
        ProblemTypes.TIME_SERIES_REGRESSION
    }


def test_model_instance(ts_data):
    X, y = ts_data
    clf = ARIMARegressor()
    fitted = clf.fit(X, y)
    assert isinstance(fitted, ARIMARegressor)


def test_get_dates_fit_and_predict(ts_data):
    X, y = ts_data
    clf = ARIMARegressor()
    date_col, X_ = clf._get_dates(X, y)
    assert isinstance(date_col, pd.DatetimeIndex)
    assert X_.equals(X)


def test_match_indices(ts_data):
    X, y = ts_data
    date_index = pd.date_range("2020-10-02", "2020-11-01")
    clf = ARIMARegressor()
    X_, y_ = clf._match_indices(X, y, date_index)
    assert isinstance(X_.index, pd.DatetimeIndex)
    assert isinstance(y_.index, pd.DatetimeIndex)
    assert X_.index.equals(y_.index)
    assert X_.index.equals(date_index)


@pytest.mark.parametrize("predict", [True, False])
@pytest.mark.parametrize("dates_shape", [0, 1, 2])
def test_format_dates(predict, dates_shape, ts_data):
    X, y = ts_data
    date_index = pd.date_range("2020-10-02", "2020-11-01")
    if dates_shape == 1:
        date_index = pd.DataFrame(date_index)
    elif dates_shape == 2:
        date_index = pd.DataFrame(data={"a": date_index, "b": date_index})

    clf = ARIMARegressor()

    if predict:
        if dates_shape != 2:
            X_, y_, fh_ = clf._format_dates(date_index, X, y, True)
            assert X_.index.equals(y_.index)
            assert isinstance(fh_, forecasting.ForecastingHorizon)
        elif dates_shape == 2:
            with pytest.raises(ValueError, match="Found 2 columns"):
                clf._format_dates(date_index, X, y, True)
    else:
        if dates_shape != 2:
            X_, y_, _ = clf._format_dates(date_index, X, y, False)
            assert X_.index.equals(y_.index)
            assert _ is None
        elif dates_shape == 2:
            with pytest.raises(ValueError, match="Found 2 columns"):
                clf._format_dates(date_index, X, y, False)


def test_feature_importance(ts_data):
    X, y = ts_data
    clf = ARIMARegressor()
    with patch.object(clf, "_component_obj"):
        clf.fit(X, y)
        assert clf.feature_importance == np.zeros(1)


def test_fit_predict_ts_with_datetime_in_X_column(
    ts_data_seasonal_train, ts_data_seasonal_test
):
    X, y = ts_data_seasonal_train
    X_test, _ = ts_data_seasonal_test
    assert isinstance(X.index, pd.DatetimeIndex)
    assert isinstance(y.index, pd.DatetimeIndex)

    m_clf = ARIMARegressor(d=None)
    m_clf.fit(X=X, y=y)
    y_pred = m_clf.predict(X=X_test)

    X["Sample"] = pd.date_range(start="1/1/2016", periods=25)

    dt_clf = ARIMARegressor(d=None)
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

    a_clf = sktime_arima.AutoARIMA()
    clf = a_clf.fit(y=y)
    y_pred_sk = clf.predict(fh=fh_)

    X = X.drop(["features"], axis=1)

    m_clf = ARIMARegressor(d=None)
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

    a_clf = sktime_arima.AutoARIMA()
    clf = a_clf.fit(X=X, y=y)
    y_pred_sk = clf.predict(fh=fh_, X=X_test)

    m_clf = ARIMARegressor(d=None)
    m_clf.fit(X=X, y=y)
    y_pred = m_clf.predict(X=X_test)

    assert (y_pred_sk == y_pred).all()


@patch(
    "evalml.pipelines.components.estimators.regressors.arima_regressor.ARIMARegressor._format_dates"
)
@patch(
    "evalml.pipelines.components.estimators.regressors.arima_regressor.ARIMARegressor._get_dates"
)
def test_fit_predict_ts_with_X_and_y_index(
    mock_get_dates,
    mock_format_dates,
    ts_data_seasonal_train,
):
    X, y = ts_data_seasonal_train
    assert isinstance(X.index, pd.DatetimeIndex)
    assert isinstance(y.index, pd.DatetimeIndex)

    mock_get_dates.return_value = (X.index, X)
    mock_format_dates.return_value = (X, y, None)

    fh_ = forecasting.ForecastingHorizon(
        [i + 1 for i in range(len(y))], is_relative=True
    )

    a_clf = sktime_arima.AutoARIMA()
    clf = a_clf.fit(X=X, y=y)
    y_pred_sk = clf.predict(fh=fh_, X=X)

    m_clf = ARIMARegressor(d=None)
    m_clf.fit(X=X, y=y)
    mock_format_dates.return_value = (X, y, fh_)
    y_pred = m_clf.predict(X=X)

    assert (y_pred_sk == y_pred).all()


@patch(
    "evalml.pipelines.components.estimators.regressors.arima_regressor.ARIMARegressor._format_dates"
)
@patch(
    "evalml.pipelines.components.estimators.regressors.arima_regressor.ARIMARegressor._get_dates"
)
def test_fit_predict_ts_with_X_not_y_index(
    mock_get_dates, mock_format_dates, ts_data_seasonal_train
):
    X, y = ts_data_seasonal_train
    assert isinstance(X.index, pd.DatetimeIndex)
    assert isinstance(y.index, pd.DatetimeIndex)

    mock_get_dates.return_value = (X.index, X)
    mock_format_dates.return_value = (X, y, None)

    fh_ = forecasting.ForecastingHorizon(
        [i + 1 for i in range(len(y))], is_relative=True
    )

    a_clf = sktime_arima.AutoARIMA()
    clf = a_clf.fit(X=X, y=y)
    y_pred_sk = clf.predict(fh=fh_, X=X)

    y = y.reset_index(drop=True)
    assert not isinstance(y.index, pd.DatetimeIndex)

    m_clf = ARIMARegressor(d=None)
    clf_ = m_clf.fit(X=X, y=y)
    mock_format_dates.return_value = (X, y, fh_)
    y_pred = clf_.predict(X=X)

    assert (y_pred_sk == y_pred).all()


@patch(
    "evalml.pipelines.components.estimators.regressors.arima_regressor.ARIMARegressor._format_dates"
)
@patch(
    "evalml.pipelines.components.estimators.regressors.arima_regressor.ARIMARegressor._get_dates"
)
def test_fit_predict_ts_with_y_not_X_index(
    mock_get_dates, mock_format_dates, ts_data_seasonal_train
):
    X, y = ts_data_seasonal_train

    mock_get_dates.return_value = (y.index, X)
    mock_format_dates.return_value = (X, y, None)

    fh_ = forecasting.ForecastingHorizon(
        [i + 1 for i in range(len(y))], is_relative=True
    )

    a_clf = sktime_arima.AutoARIMA()
    clf = a_clf.fit(X=X, y=y)
    y_pred_sk = clf.predict(fh=fh_, X=X)

    X_no_ind = X.reset_index(drop=True)
    assert isinstance(y.index, pd.DatetimeIndex)
    assert not isinstance(X_no_ind.index, pd.DatetimeIndex)

    m_clf = ARIMARegressor(d=None)
    clf_ = m_clf.fit(X=X_no_ind, y=y)
    mock_format_dates.return_value = (X, y, fh_)
    y_pred = clf_.predict(X=X, y=y)

    assert (y_pred_sk == y_pred).all()


def test_predict_ts_without_X_error(ts_data):
    X, y = ts_data

    m_clf = ARIMARegressor()
    clf_ = m_clf.fit(X=X, y=y)
    with pytest.raises(
        ValueError, match="If X was passed to the fit method of the ARIMARegressor"
    ):
        clf_.predict(y=y)


@patch("sktime.forecasting.base._sktime.BaseForecaster.predict")
@patch("sktime.forecasting.base._sktime.BaseForecaster.fit")
def test_predict_ts_X_error(mock_sktime_fit, mock_sktime_predict, ts_data):
    X, y = ts_data

    mock_sktime_predict.side_effect = ValueError("Sktime value error")

    m_clf = ARIMARegressor()
    clf_ = m_clf.fit(X=X, y=y)
    with pytest.raises(ValueError, match="Sktime value error"):
        clf_.predict(y=y)


def test_fit_ts_with_not_X_not_y_index(ts_data):
    X, y = ts_data
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    assert not isinstance(y.index, pd.DatetimeIndex)
    assert not isinstance(X.index, pd.DatetimeIndex)

    clf = ARIMARegressor()
    with pytest.raises(
        ValueError,
        match="If not it will look for the datetime column in the index of X or y.",
    ):
        clf.fit(X=X, y=y)


def test_predict_ts_with_not_X_index(ts_data):
    X, y = ts_data
    X = X.reset_index(drop=True)
    assert not isinstance(X.index, pd.DatetimeIndex)

    m_clf = ARIMARegressor()
    clf_ = m_clf.fit(X=X, y=y)
    with pytest.raises(
        ValueError,
        match="If not it will look for the datetime column in the index of X.",
    ):
        clf_.predict(X)


def test_fit_ts_without_y(ts_data):
    X, y = ts_data

    clf = ARIMARegressor()
    with pytest.raises(ValueError, match="ARIMA Regressor requires y as input."):
        clf.fit(X=X)


def test_fit_predict_ts_no_X_out_of_sample(
    ts_data_seasonal_train, ts_data_seasonal_test
):
    X, y = ts_data_seasonal_train
    X_test, y_test = ts_data_seasonal_test

    fh_ = forecasting.ForecastingHorizon(
        [i + 1 for i in range(len(y_test))], is_relative=True
    )

    a_clf = sktime_arima.AutoARIMA()
    a_clf.fit(y=y)
    y_pred_sk = a_clf.predict(fh=fh_)

    m_clf = ARIMARegressor(d=None)
    m_clf.fit(X=None, y=y)
    y_pred = m_clf.predict(X=None, y=y_test)

    assert (y_pred_sk == y_pred).all()


@pytest.mark.parametrize("X_none", [True, False])
def test_fit_predict_date_index_named_out_of_sample(
    X_none, ts_data_seasonal_train, ts_data_seasonal_test
):
    X, y = ts_data_seasonal_train
    X_test, y_test = ts_data_seasonal_test

    fh_ = forecasting.ForecastingHorizon(
        [i + 1 for i in range(len(y_test))], is_relative=True
    )

    a_clf = sktime_arima.AutoARIMA()
    if X_none:
        clf = a_clf.fit(y=y)
        y_pred_sk = clf.predict(fh=fh_)
    else:
        clf = a_clf.fit(X=X, y=y)
        y_pred_sk = clf.predict(fh=fh_, X=X_test)

    X = X.reset_index()
    assert not isinstance(X.index, pd.DatetimeIndex)
    m_clf = ARIMARegressor(date_index="index", d=None)
    if X_none:
        m_clf.fit(X=None, y=y)
        y_pred = m_clf.predict(X=None, y=y_test)
    else:
        m_clf.fit(X=X, y=y)
        y_pred = m_clf.predict(X=X_test, y=y_test)

    assert (y_pred_sk == y_pred).all()


@pytest.mark.parametrize("freq_num", ["1", "2"])
@pytest.mark.parametrize("freq_str", ["T", "M", "Y"])
def test_different_time_units_out_of_sample(freq_str, freq_num):
    datetime_ = pd.date_range("1/1/1870", periods=20, freq=freq_num + freq_str)

    X = pd.DataFrame(range(20), index=datetime_)
    y = pd.Series(np.sin(np.linspace(-8 * np.pi, 8 * np.pi, 20)), index=datetime_)

    fh_ = forecasting.ForecastingHorizon(
        [i + 1 for i in range(len(y[15:]))], is_relative=True
    )

    a_clf = sktime_arima.AutoARIMA(start_p=2, start_q=2, max_p=2, max_q=2)
    clf = a_clf.fit(X=X[:15], y=y[:15])
    y_pred_sk = clf.predict(fh=fh_, X=X[15:])

    m_clf = ARIMARegressor(start_p=2, start_q=2, max_p=2, max_q=2, d=None)
    m_clf.fit(X=X[:15], y=y[:15])
    y_pred = m_clf.predict(X=X[15:], y=y[15:])

    pd.testing.assert_series_equal(y_pred_sk, y_pred)
