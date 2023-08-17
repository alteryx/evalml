from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from evalml.model_family import ModelFamily
from evalml.pipelines.components import VARMAXRegressor
from evalml.preprocessing import split_data
from evalml.problem_types import ProblemTypes

pytestmark = [
    pytest.mark.skip_during_conda,
]


def test_model_family():
    assert VARMAXRegressor.model_family == ModelFamily.VARMAX


def test_problem_types():
    assert set(VARMAXRegressor.supported_problem_types) == {
        ProblemTypes.MULTISERIES_TIME_SERIES_REGRESSION,
    }


def test_model_instance(ts_multiseries_data):
    X, _, y = ts_multiseries_data(n_series=2)
    clf = VARMAXRegressor()
    fitted = clf.fit(X, y)
    assert isinstance(fitted, VARMAXRegressor)


def test_fit_ts_without_y(ts_multiseries_data):
    X, _, _ = ts_multiseries_data(n_series=2)

    clf = VARMAXRegressor()
    with pytest.raises(ValueError, match="VARMAX Regressor requires y as input."):
        clf.fit(X=X)


@patch("sktime.forecasting.varmax.VARMAX.fit")
def test_remove_datetime_feature(
    mock_fit,
    ts_multiseries_data,
):
    X_train, _, y_train = ts_multiseries_data(
        datetime_feature=True,
    )

    clf = VARMAXRegressor(use_covariates=True)
    clf.fit(X_train, y_train)

    assert "date" not in mock_fit.call_args.kwargs["X"]
    assert (
        mock_fit.call_args.kwargs["X"].select_dtypes(include=["datetime64"]).shape[1]
        == 0
    )


def test_set_forecast(ts_multiseries_data):
    from sktime.forecasting.base import ForecastingHorizon

    X, X_test, _ = ts_multiseries_data(
        train_features_index_dt=False,
        train_target_index_dt=False,
        train_none=False,
        datetime_feature=False,
        no_features=False,
        test_features_index_dt=False,
    )

    clf = VARMAXRegressor()
    clf.last_X_index = X.index[-1]
    fh_ = clf._set_forecast_horizon(X_test)
    assert isinstance(fh_, ForecastingHorizon)
    assert len(fh_) == len(X_test)
    assert fh_.is_relative


def test_feature_importance(ts_multiseries_data):
    X, _, y = ts_multiseries_data(n_series=2)
    clf = VARMAXRegressor()
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
    ],
)
@pytest.mark.parametrize("use_covariates", [True, False])
def test_fit_predict(
    train_features_index_dt,
    train_target_index_dt,
    train_none,
    no_features,
    datetime_feature,
    test_features_index_dt,
    use_covariates,
    ts_multiseries_data,
):
    from sktime.forecasting.base import ForecastingHorizon
    from sktime.forecasting.varmax import VARMAX

    X_train, X_test, y_train = ts_multiseries_data(
        train_features_index_dt,
        train_target_index_dt,
        train_none,
        datetime_feature,
        no_features,
        test_features_index_dt,
    )

    fh_ = ForecastingHorizon([i + 1 for i in range(len(X_test))], is_relative=True)

    a_clf = VARMAX(maxiter=10)
    if use_covariates:
        clf = a_clf.fit(X=X_train, y=y_train)
        y_pred_sk = clf.predict(fh=fh_, X=X_test)
    else:
        clf = a_clf.fit(y=y_train)
        y_pred_sk = clf.predict(fh=fh_)

    m_clf = VARMAXRegressor(maxiter=10, use_covariates=use_covariates)
    m_clf.fit(X=X_train, y=y_train)
    y_pred = m_clf.predict(X=X_test)
    np.testing.assert_almost_equal(y_pred_sk.values, y_pred.values)
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
    ts_multiseries_data,
):
    from sktime.forecasting.varmax import VARMAX

    X_train, X_test, y_train = ts_multiseries_data(
        train_features_index_dt,
        train_target_index_dt,
        train_none,
        datetime_feature,
        no_features,
        test_features_index_dt,
        match_indices=False,
    )

    a_clf = VARMAX(maxiter=10)
    # Each parameter combo generates a dataset combo that will raise an error to the underlying sktime component.
    with pytest.raises(ValueError):
        a_clf.fit(X=X_train, y=y_train)

    m_clf = VARMAXRegressor()
    m_clf.fit(X=X_train, y=y_train)
    y_pred = m_clf.predict(X=X_test)
    assert isinstance(y_pred, pd.DataFrame)
    assert len(y_pred) == 10
    assert y_pred.index.equals(X_test.index)


@pytest.mark.parametrize("freq_num", ["1", "2"])
@pytest.mark.parametrize("freq_str", ["T", "M", "Y"])
@pytest.mark.parametrize("use_covariates", [True, False])
def test_different_time_units_out_of_sample(
    freq_str,
    freq_num,
    ts_multiseries_data,
    use_covariates,
):
    from sktime.forecasting.base import ForecastingHorizon
    from sktime.forecasting.varmax import VARMAX

    X, _, y = ts_multiseries_data(freq=freq_num + freq_str, datetime_feature=False)
    fh_ = ForecastingHorizon([i + 1 for i in range(len(y[15:]))], is_relative=True)

    a_clf = VARMAX(maxiter=10)
    if use_covariates:
        clf = a_clf.fit(X=X[:15], y=y[:15])
        y_pred_sk = clf.predict(fh=fh_, X=X[15:])
    else:
        clf = a_clf.fit(y=y[:15])
        y_pred_sk = clf.predict(fh=fh_)

    m_clf = VARMAXRegressor(use_covariates=use_covariates)
    m_clf.fit(X=X[:15], y=y[:15])
    y_pred = m_clf.predict(X=X[15:])

    np.testing.assert_almost_equal(y_pred_sk.values, y_pred.values)
    assert y_pred.index.equals(X[15:].index)


def test_varmax_supports_boolean_features():
    from sktime.forecasting.varmax import VARMAX

    X = pd.DataFrame(
        {
            "dates": pd.date_range("2021-01-01", periods=10),
            "bool_1": pd.Series([True, False])
            .sample(n=10, replace=True, random_state=0)
            .reset_index(drop=True),
            "bool_2": pd.Series([True, False])
            .sample(n=10, replace=True, random_state=1)
            .reset_index(drop=True),
        },
    )
    X.ww.init()
    y = pd.DataFrame({"target_1": np.random.rand(10), "target_2": np.random.rand(10)})

    vx = VARMAXRegressor(time_index="dates", use_covariates=True)

    with patch.object(VARMAX, "fit") as mock_fit:
        vx.fit(X, y)
        pd.testing.assert_series_equal(
            mock_fit.call_args[1]["X"]["bool_1"],
            X["bool_1"].astype(float),
        )
        pd.testing.assert_series_equal(
            mock_fit.call_args[1]["X"]["bool_2"],
            X["bool_2"].astype(float),
        )

    vx.fit(X, y)
    preds = vx.predict(X)
    assert all(preds.isna().eq(False))


@patch("sktime.forecasting.varmax.VARMAX.fit")
@patch("sktime.forecasting.varmax.VARMAX.predict")
def test_varmax_regressor_respects_use_covariates(
    mock_predict,
    mock_fit,
    ts_multiseries_data,
):
    X_train, X_test, y_train = ts_multiseries_data(n_series=2)
    clf = VARMAXRegressor(use_covariates=False)

    mock_returned = pd.DataFrame({"lower": [1] * 10, "upper": [2] * 10})
    mock_returned = pd.concat({0.95: mock_returned}, axis=1)
    mock_returned = pd.concat({"Coverage": mock_returned}, axis=1)
    mock_predict.return_value = mock_returned

    clf.fit(X_train, y_train)
    clf.predict(X_test)
    mock_fit.assert_called_once()
    assert "X" not in mock_fit.call_args.kwargs
    assert "y" in mock_fit.call_args.kwargs
    mock_predict.assert_called_once()
    assert "X" not in mock_predict.call_args.kwargs


@patch("sktime.forecasting.varmax.VARMAX.fit")
def test_varmax_regressor_X_datetime_only(mock_fit, multiseries_ts_data_unstacked):
    X, y = multiseries_ts_data_unstacked
    X.ww.init()
    X = X.ww.select(include=["Datetime"])

    clf = VARMAXRegressor(use_covariates=True)
    clf.fit(X, y)

    assert "X" not in mock_fit.call_args.kwargs


def test_varmax_regressor_can_forecast_arbitrary_dates_no_covariates(
    ts_multiseries_data,
):
    X, _, y = ts_multiseries_data(n_series=2)
    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        problem_type="time series regression",
        test_size=0.2,
        random_seed=0,
    )

    X_test_last_5 = X_test.tail(5)

    varmax = VARMAXRegressor(use_covariates=False)
    varmax.fit(X_train, y_train)

    pd.testing.assert_frame_equal(
        varmax.predict(X_test).tail(5),
        varmax.predict(X_test_last_5),
    )


@pytest.mark.parametrize("use_covariates", [True, False])
def test_varmax_regressor_can_forecast_arbitrary_dates_past_holdout(
    use_covariates,
    ts_multiseries_data,
):
    X, _, y = ts_multiseries_data(
        train_features_index_dt=False,
        train_target_index_dt=False,
    )

    # Create a training and testing set that are not continuous
    X_train, X_test, y_train, _ = split_data(
        X,
        y,
        problem_type="time series regression",
        test_size=0.2,
        random_seed=0,
    )
    X_train, _, y_train, _ = split_data(
        X_train,
        y_train,
        problem_type="time series regression",
        test_size=0.4,
        random_seed=0,
    )

    varmax = VARMAXRegressor(use_covariates=use_covariates)
    varmax.fit(X_train, y_train)

    varmax.predict(X_test)


@pytest.mark.parametrize("use_X_train", [True, False])
@pytest.mark.parametrize("coverages", [None, [0.85], [0.95, 0.90, 0.85]])
@pytest.mark.parametrize("use_covariates", [True, False])
def test_varmax_regressor_prediction_intervals(
    use_covariates,
    coverages,
    use_X_train,
    ts_multiseries_data,
):
    X_train, X_test, y_train = ts_multiseries_data(no_features=not use_covariates)

    clf = VARMAXRegressor(use_covariates=use_covariates)
    clf.fit(X=X_train if use_X_train else None, y=y_train)

    # Check we are not using exogenous variables if use_covariates=False, even if X_test is passed.
    if not use_covariates or not use_X_train:
        with patch.object(
            clf._component_obj._fitted_forecaster,
            "simulate",
        ) as mock_simulate:
            clf.get_prediction_intervals(X_test, None, coverages)
            assert mock_simulate.call_args[1]["exog"] is None

    results_coverage = clf.get_prediction_intervals(X_test, None, coverages)
    predictions = clf.predict(X_test)

    series_id_targets = list(results_coverage.keys())
    for series in series_id_targets:
        series_results_coverage = results_coverage[series]
        conf_ints = list(series_results_coverage.keys())
        data = list(series_results_coverage.values())

        assert len(conf_ints) == (len(coverages) if coverages is not None else 1) * 2
        assert len(data) == (len(coverages) if coverages is not None else 1) * 2

        for interval in coverages if coverages is not None else [0.95]:
            conf_int_lower = f"{interval}_lower"
            conf_int_upper = f"{interval}_upper"

            assert (series_results_coverage[conf_int_upper] > predictions[series]).all()
            assert (predictions[series] > series_results_coverage[conf_int_lower]).all()
