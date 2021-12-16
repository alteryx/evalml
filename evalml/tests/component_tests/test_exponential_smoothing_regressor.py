from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from evalml.model_family import ModelFamily
from evalml.pipelines.components import ExponentialSmoothingRegressor
from evalml.problem_types import ProblemTypes

pytestmark = [
    pytest.mark.noncore_dependency,
    pytest.mark.skip_during_conda,
    pytest.mark.skip_if_39,
]


@pytest.fixture(scope="module")
def sktime_arima():
    from sktime.forecasting import exp_smoothing as sktime_exp

    return sktime_arima


@pytest.fixture(scope="module")
def forecasting():
    from sktime.forecasting import base as forecasting

    return forecasting


def test_model_family():
    assert (
        ExponentialSmoothingRegressor().model_family
        == ModelFamily.EXPONENTIAL_SMOOTHING
    )


def test_problem_types():
    assert set(ExponentialSmoothingRegressor.supported_problem_types) == {
        ProblemTypes.TIME_SERIES_REGRESSION
    }


def test_model_instance(ts_data):
    X, y = ts_data
    clf = ExponentialSmoothingRegressor()
    fitted = clf.fit(X, y)
    assert isinstance(fitted, ExponentialSmoothingRegressor)


def test_fit_ts_without_y(ts_data):
    X, y = ts_data

    clf = ExponentialSmoothingRegressor()
    with pytest.raises(
        ValueError, match="Exponential Smoothing Regressor requires y as input."
    ):
        clf.fit(X=X)


@pytest.fixture
def get_X_y():
    def _get_X_y(
        train_features_index_dt,
        train_target_index_dt,
        train_none,
        datetime_feature,
        no_features,
        test_features_index_dt,
    ):
        X = pd.DataFrame(index=[i + 1 for i in range(50)])
        dates = pd.date_range("1/1/21", periods=50)
        feature = [1, 5, 2] * 10 + [3, 1] * 10
        y = pd.Series([1, 2, 3, 4, 5, 6, 5, 4, 3, 2] * 5)

        X_train = X[:40]
        X_test = X[40:]
        y_train = y[:40]

        if train_features_index_dt:
            X_train.index = dates[:40]
        if train_target_index_dt:
            y_train.index = dates[:40]
        if test_features_index_dt:
            X_test.index = dates[40:]
        if not no_features:
            X_train["Feature"] = feature[:40]
            X_test["Feature"] = feature[40:]
            if datetime_feature:
                X_train["Dates"] = dates[:40]
                X_test["Dates"] = dates[40:]
        if train_none:
            X_train = None

        return X_train, X_test, y_train

    return _get_X_y


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
    get_X_y,
):
    X_train, _, y_train = get_X_y(
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

    clf = ExponentialSmoothingRegressor()
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


def test_match_indices(get_X_y):
    X_train, _, y_train = get_X_y(
        train_features_index_dt=False,
        train_target_index_dt=False,
        train_none=False,
        datetime_feature=False,
        no_features=False,
        test_features_index_dt=False,
    )

    assert not X_train.index.equals(y_train.index)

    clf = ExponentialSmoothingRegressor()
    X_, y_ = clf._match_indices(X_train, y_train)
    assert X_.index.equals(y_.index)


def test_set_forecast(get_X_y):
    from sktime.forecasting.base import ForecastingHorizon

    _, X_test, _ = get_X_y(
        train_features_index_dt=False,
        train_target_index_dt=False,
        train_none=False,
        datetime_feature=False,
        no_features=False,
        test_features_index_dt=False,
    )

    clf = ExponentialSmoothingRegressor()
    fh_ = clf._set_forecast(X_test)
    assert isinstance(fh_, ForecastingHorizon)
    assert len(fh_) == len(X_test)
    assert fh_.is_relative


def test_feature_importance(ts_data):
    X, y = ts_data
    clf = ExponentialSmoothingRegressor()
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
    get_X_y,
):
    from sktime.forecasting.base import ForecastingHorizon
    from sktime.forecasting.exp_smoothing import ExponentialSmoothing

    X_train, X_test, y_train = get_X_y(
        train_features_index_dt,
        train_target_index_dt,
        train_none,
        datetime_feature,
        no_features,
        test_features_index_dt,
    )

    fh_ = ForecastingHorizon([i + 1 for i in range(len(X_test))], is_relative=True)

    sk_clf = ExponentialSmoothing()
    clf = sk_clf.fit(X=X_train, y=y_train)
    y_pred_sk = clf.predict(fh=fh_, X=X_test)

    m_clf = ExponentialSmoothingRegressor()
    m_clf.fit(X=X_train, y=y_train)
    y_pred = m_clf.predict(X=X_test)

    assert (y_pred_sk.values == y_pred.values).all()
    assert y_pred.index.equals(X_test.index)
