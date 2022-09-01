from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from evalml.model_family import ModelFamily
from evalml.pipelines.components import ExponentialSmoothingRegressor
from evalml.problem_types import ProblemTypes

pytestmark = [
    pytest.mark.skip_during_conda,
]


def test_model_family():
    assert (
        ExponentialSmoothingRegressor().model_family
        == ModelFamily.EXPONENTIAL_SMOOTHING
    )


def test_problem_types():
    assert set(ExponentialSmoothingRegressor.supported_problem_types) == {
        ProblemTypes.TIME_SERIES_REGRESSION,
    }


def test_model_instance(ts_data):
    X, _, y = ts_data()
    regressor = ExponentialSmoothingRegressor()
    fitted = regressor.fit(X, y)
    assert isinstance(fitted, ExponentialSmoothingRegressor)


def test_fit_ts_without_y(ts_data):
    X, _, _ = ts_data()

    regressor = ExponentialSmoothingRegressor()
    with pytest.raises(
        ValueError,
        match="Exponential Smoothing Regressor requires y as input.",
    ):
        regressor.fit(X=X)


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
@patch("sktime.forecasting.exp_smoothing.ExponentialSmoothing.fit")
def test_remove_datetime(
    mock_fit,
    train_features_index_dt,
    train_target_index_dt,
    train_none,
    datetime_feature,
    no_features,
    ts_data,
):
    X_train, _, y_train = ts_data(
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

    regressor = ExponentialSmoothingRegressor()
    regressor.fit(X_train, y_train)

    y_train_removed = mock_fit.call_args.kwargs.get("y", None)
    if y_train_removed is not None:
        assert not isinstance(y_train_removed.index, pd.DatetimeIndex)


def test_set_forecast(ts_data):
    from sktime.forecasting.base import ForecastingHorizon

    _, X_test, _ = ts_data(
        train_features_index_dt=False,
        train_target_index_dt=False,
        train_none=False,
        datetime_feature=False,
        no_features=False,
        test_features_index_dt=False,
    )

    regressor = ExponentialSmoothingRegressor()
    fh_ = regressor._set_forecast(X_test)
    assert isinstance(fh_, ForecastingHorizon)
    assert len(fh_) == len(X_test)
    assert fh_.is_relative


def test_feature_importance(ts_data):
    X, _, y = ts_data()
    regressor = ExponentialSmoothingRegressor()
    with patch.object(regressor, "_component_obj"):
        regressor.fit(X, y)
        pd.testing.assert_series_equal(
            regressor.feature_importance,
            pd.Series(np.zeros(1)),
        )


@pytest.mark.parametrize(
    "train_none, train_features_index_dt, "
    "train_target_index_dt, no_features, "
    "datetime_feature, test_features_index_dt",
    [
        (True, False, False, False, False, False),
        (False, True, True, False, False, True),
        (False, True, True, False, False, False),
        (True, False, True, True, True, False),
    ],
)
def test_fit_predict(
    train_features_index_dt,
    train_target_index_dt,
    train_none,
    no_features,
    datetime_feature,
    test_features_index_dt,
    ts_data,
):
    from sktime.forecasting.base import ForecastingHorizon
    from sktime.forecasting.exp_smoothing import ExponentialSmoothing

    X_train, X_test, y_train = ts_data(
        train_features_index_dt,
        train_target_index_dt,
        train_none,
        datetime_feature,
        no_features,
        test_features_index_dt,
    )

    fh_ = ForecastingHorizon([i + 1 for i in range(len(X_test))], is_relative=True)

    sk_clf = ExponentialSmoothing()
    regressor = sk_clf.fit(X=X_train, y=y_train)
    y_pred_sk = regressor.predict(fh=fh_, X=X_test)

    m_clf = ExponentialSmoothingRegressor()
    m_clf.fit(X=X_train, y=y_train)
    y_pred = m_clf.predict(X=X_test)

    assert (y_pred_sk.values == y_pred.values).all()
    assert y_pred.index.equals(X_test.index)


def test_predict_no_X_in_fit(
    ts_data,
):
    from sktime.forecasting.base import ForecastingHorizon
    from sktime.forecasting.exp_smoothing import ExponentialSmoothing

    X_train, X_test, y_train = ts_data(
        train_features_index_dt=False,
        train_target_index_dt=True,
        train_none=True,
        datetime_feature=False,
        no_features=True,
        test_features_index_dt=False,
    )

    fh_ = ForecastingHorizon([i + 1 for i in range(len(X_test))], is_relative=True)

    sk_clf = ExponentialSmoothing()
    regressor = sk_clf.fit(X=X_train, y=y_train)
    y_pred_sk = regressor.predict(fh=fh_)

    m_clf = ExponentialSmoothingRegressor()
    m_clf.fit(X=None, y=y_train)
    y_pred = m_clf.predict(X=X_test)

    assert (y_pred_sk.values == y_pred.values).all()
