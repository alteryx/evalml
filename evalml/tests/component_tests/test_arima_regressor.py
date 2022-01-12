from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from evalml.model_family import ModelFamily
from evalml.pipelines.components import ARIMARegressor
from evalml.problem_types import ProblemTypes

pytestmark = [
    pytest.mark.noncore_dependency,
    pytest.mark.skip_during_conda,
    pytest.mark.skip_if_39,
]


@pytest.fixture(scope="module")
def sktime_arima():
    from sktime.forecasting import arima as sktime_arima

    return sktime_arima


@pytest.fixture(scope="module")
def forecasting():
    from sktime.forecasting import base as forecasting

    return forecasting


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


def test_fit_ts_without_y(ts_data):
    X, y = ts_data

    clf = ARIMARegressor()
    with pytest.raises(ValueError, match="ARIMA Regressor requires y as input."):
        clf.fit(X=X)


@pytest.mark.parametrize("train_features_index_dt", [True, False])
@pytest.mark.parametrize("train_target_index_dt", [True, False])
@pytest.mark.parametrize(
    "train_none, no_features, datetime_feature",
    [
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (False, False, False),
    ],
)
def test_remove_datetime(
    train_features_index_dt,
    train_target_index_dt,
    train_none,
    datetime_feature,
    no_features,
    get_ts_X_y,
):
    X_train, _, y_train = get_ts_X_y(
        train_features_index_dt,
        train_target_index_dt,
        train_none,
        datetime_feature,
        no_features,
        test_features_index_dt=False,
    )

    if not train_none:
        if train_features_index_dt:
            assert isinstance(X_train.index, pd.DatetimeIndex)
        else:
            assert not isinstance(X_train.index, pd.DatetimeIndex)
        if datetime_feature:
            assert X_train.select_dtypes(include=["datetime64"]).shape[1] == 1
    if train_target_index_dt:
        assert isinstance(y_train.index, pd.DatetimeIndex)
    else:
        assert not isinstance(y_train.index, pd.DatetimeIndex)

    clf = ARIMARegressor()
    X_train_no_dt = clf._remove_datetime(X_train, features=True)
    y_train_no_dt = clf._remove_datetime(y_train)

    if train_none:
        assert X_train_no_dt is None
    else:
        assert not isinstance(X_train_no_dt.index, pd.DatetimeIndex)
        if no_features:
            assert X_train_no_dt.shape[1] == 0
        if datetime_feature:
            assert X_train_no_dt.select_dtypes(include=["datetime64"]).shape[1] == 0

    assert not isinstance(y_train_no_dt.index, pd.DatetimeIndex)


def test_match_indices(get_ts_X_y):
    X_train, _, y_train = get_ts_X_y(
        train_features_index_dt=False,
        train_target_index_dt=False,
        train_none=False,
        datetime_feature=False,
        no_features=False,
        test_features_index_dt=False,
    )

    assert not X_train.index.equals(y_train.index)

    clf = ARIMARegressor()
    X_, y_ = clf._match_indices(X_train, y_train)
    assert X_.index.equals(y_.index)


def test_set_forecast(get_ts_X_y):
    from sktime.forecasting.base import ForecastingHorizon

    _, X_test, _ = get_ts_X_y(
        train_features_index_dt=False,
        train_target_index_dt=False,
        train_none=False,
        datetime_feature=False,
        no_features=False,
        test_features_index_dt=False,
    )

    clf = ARIMARegressor()
    fh_ = clf._set_forecast(X_test)
    assert isinstance(fh_, ForecastingHorizon)
    assert len(fh_) == len(X_test)
    assert fh_.is_relative


def test_feature_importance(ts_data):
    X, y = ts_data
    clf = ARIMARegressor()
    with patch.object(clf, "_component_obj"):
        clf.fit(X, y)
        assert clf.feature_importance == np.zeros(1)


@pytest.mark.parametrize(
    "train_none, train_features_index_dt, "
    "train_target_index_dt, no_features, "
    "datetime_feature, test_features_index_dt",
    [
        (True, False, False, False, False, False),
        (False, True, True, False, False, True),
        (False, True, True, False, False, False),
    ],
)
def test_fit_predict(
    train_features_index_dt,
    train_target_index_dt,
    train_none,
    no_features,
    datetime_feature,
    test_features_index_dt,
    get_ts_X_y,
):
    from sktime.forecasting.arima import AutoARIMA
    from sktime.forecasting.base import ForecastingHorizon

    X_train, X_test, y_train = get_ts_X_y(
        train_features_index_dt,
        train_target_index_dt,
        train_none,
        datetime_feature,
        no_features,
        test_features_index_dt,
    )

    fh_ = ForecastingHorizon([i + 1 for i in range(len(X_test))], is_relative=True)

    a_clf = AutoARIMA()
    clf = a_clf.fit(X=X_train, y=y_train)
    y_pred_sk = clf.predict(fh=fh_, X=X_test)

    m_clf = ARIMARegressor(d=None)
    m_clf.fit(X=X_train, y=y_train)
    y_pred = m_clf.predict(X=X_test)

    assert (y_pred_sk.values == y_pred.values).all()
    assert y_pred.index.equals(X_test.index)


@pytest.mark.parametrize(
    "train_none, train_features_index_dt, "
    "train_target_index_dt, no_features, "
    "datetime_feature, test_features_index_dt",
    [
        (False, False, False, False, False, False),
        (False, True, False, False, False, True),
        (False, False, True, False, True, False),
        (False, False, True, True, False, False),
        (False, True, True, True, False, False),
        (False, True, True, False, True, False),
    ],
)
def test_fit_predict_sk_failure(
    train_features_index_dt,
    train_target_index_dt,
    train_none,
    no_features,
    datetime_feature,
    test_features_index_dt,
    get_ts_X_y,
):
    from sktime.forecasting.arima import AutoARIMA

    X_train, X_test, y_train = get_ts_X_y(
        train_features_index_dt,
        train_target_index_dt,
        train_none,
        datetime_feature,
        no_features,
        test_features_index_dt,
    )

    a_clf = AutoARIMA()
    with pytest.raises(Exception):
        a_clf.fit(X=X_train, y=y_train)

    m_clf = ARIMARegressor(d=None)
    m_clf.fit(X=X_train, y=y_train)
    y_pred = m_clf.predict(X=X_test)
    assert isinstance(y_pred, pd.Series)
    assert len(y_pred) == 10
    assert y_pred.index.equals(X_test.index)


@pytest.mark.parametrize("freq_num", ["1", "2"])
@pytest.mark.parametrize("freq_str", ["T", "M", "Y"])
def test_different_time_units_out_of_sample(
    freq_str, freq_num, sktime_arima, forecasting
):
    from sktime.forecasting.arima import AutoARIMA
    from sktime.forecasting.base import ForecastingHorizon

    datetime_ = pd.date_range("1/1/1870", periods=20, freq=freq_num + freq_str)

    X = pd.DataFrame(range(20), index=datetime_)
    y = pd.Series(np.sin(np.linspace(-8 * np.pi, 8 * np.pi, 20)), index=datetime_)

    fh_ = ForecastingHorizon([i + 1 for i in range(len(y[15:]))], is_relative=True)

    a_clf = AutoARIMA()
    clf = a_clf.fit(X=X[:15], y=y[:15])
    y_pred_sk = clf.predict(fh=fh_, X=X[15:])

    m_clf = ARIMARegressor(d=None)
    m_clf.fit(X=X[:15], y=y[:15])
    y_pred = m_clf.predict(X=X[15:])

    assert (y_pred_sk.values == y_pred.values).all()
    assert y_pred.index.equals(X[15:].index)


def test_arima_supports_boolean_features():
    X = pd.DataFrame({"dates": pd.date_range("2021-01-01", periods=10)})
    X.ww.init()
    X.ww["bool_1"] = (
        pd.Series([True, False])
        .sample(n=10, replace=True, random_state=0)
        .reset_index(drop=True)
    )
    X.ww["bool_2"] = (
        pd.Series([True, False])
        .sample(n=10, replace=True, random_state=1)
        .reset_index(drop=True)
    )
    y = pd.Series(range(10))

    ar = ARIMARegressor(time_index="dates")

    ar._component_obj = MagicMock()
    ar.fit(X, y)

    pd.testing.assert_series_equal(
        ar._component_obj.fit.call_args[1]["X"]["bool_1"], X["bool_1"].astype(float)
    )
    pd.testing.assert_series_equal(
        ar._component_obj.fit.call_args[1]["X"]["bool_2"], X["bool_2"].astype(float)
    )

    # Test that non-mocked predict does not error or produce NaNs
    ar = ARIMARegressor(time_index="dates")
    ar.fit(X, y)
    preds = ar.predict(X)
    assert not preds.isna().any()
