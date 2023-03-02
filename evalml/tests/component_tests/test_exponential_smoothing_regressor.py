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


@pytest.mark.parametrize(
    "nullable_y_ltype",
    ["IntegerNullable", "AgeNullable", "BooleanNullable"],
)
def test_handle_nullable_types(
    nullable_type_test_data,
    nullable_type_target,
    nullable_y_ltype,
):
    y = nullable_type_target(ltype=nullable_y_ltype, has_nans=False)
    X = nullable_type_test_data(has_nans=False)
    X = X.ww.select(include=["numeric", "Boolean", "BooleanNullable"])

    comp = ExponentialSmoothingRegressor()

    X_d, y_d = comp._handle_nullable_types(X, y)
    comp.fit(X_d, y_d)
    comp.predict(X_d)


@pytest.mark.parametrize(
    "nullable_y_ltype",
    ["IntegerNullable", "AgeNullable", "BooleanNullable"],
)
@pytest.mark.parametrize(
    "handle_incompatibility",
    [
        True,
        pytest.param(
            False,
            marks=pytest.mark.xfail(strict=True, raises=ValueError),
        ),
    ],
)
def test_exponential_smoothing_regressor_nullable_type_incompatibility(
    nullable_type_target,
    nullable_type_test_data,
    handle_incompatibility,
    nullable_y_ltype,
):
    """Testing that the nullable type incompatibility that caused us to add handling for ExponentialSmoothingRegressor
    is still present in sktime's ForecastingHorizon component. If this test is causing the test suite to fail
    because the code below no longer raises the expected ValueError, we should confirm that the nullable
    types now work for our use case and remove the nullable type handling logic from ExponentialSmoothingRegressor.
    """
    from sktime.forecasting.base import ForecastingHorizon
    from sktime.forecasting.exp_smoothing import ExponentialSmoothing

    y = nullable_type_target(ltype=nullable_y_ltype, has_nans=False)
    X = nullable_type_test_data(has_nans=False)
    X = X.ww.select(include=["numeric", "Boolean", "BooleanNullable"])

    if handle_incompatibility:
        comp = ExponentialSmoothingRegressor()
        X, y = comp._handle_nullable_types(X, y)

    X_train = X.ww.iloc[:10, :]
    X_test = X.ww.iloc[10:, :]
    y_train = y[:10]

    fh_ = ForecastingHorizon([i + 1 for i in range(len(X_test))], is_relative=True)

    sk_comp = ExponentialSmoothing()
    sk_comp.fit(X=X_train, y=y_train)
    sk_comp.predict(fh=fh_)
