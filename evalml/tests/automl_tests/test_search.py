from unittest.mock import patch

import pandas as pd
import pytest

from evalml.automl import AutoMLSearch, search
from evalml.automl.automl_algorithm import DefaultAlgorithm
from evalml.utils import infer_feature_types


@patch("evalml.data_checks.default_data_checks.DefaultDataChecks.validate")
@patch("evalml.automl.AutoMLSearch.search")
def test_search(
    mock_automl_search,
    mock_data_checks_validate,
    X_y_binary,
    dummy_data_check_validate_output_warnings,
):
    X, y = X_y_binary
    mock_data_checks_validate.return_value = dummy_data_check_validate_output_warnings
    automl, data_check_results = search(X_train=X, y_train=y, problem_type="binary")
    assert isinstance(automl, AutoMLSearch)
    assert data_check_results is dummy_data_check_validate_output_warnings
    mock_data_checks_validate.assert_called_once()
    data, target = (
        mock_data_checks_validate.call_args[0][0],
        mock_data_checks_validate.call_args[1]["y"],
    )
    pd.testing.assert_frame_equal(data, infer_feature_types(X))
    pd.testing.assert_series_equal(target, infer_feature_types(y))
    mock_automl_search.assert_called_once()


@patch("evalml.data_checks.default_data_checks.DefaultDataChecks.validate")
@patch("evalml.automl.AutoMLSearch.search")
def test_search_data_check_error(
    mock_automl_search,
    mock_data_checks_validate,
    X_y_binary,
    dummy_data_check_validate_output_errors,
):
    X, y = X_y_binary
    mock_data_checks_validate.return_value = dummy_data_check_validate_output_errors
    automl, data_check_results = search(X_train=X, y_train=y, problem_type="binary")
    assert automl is None
    assert data_check_results == dummy_data_check_validate_output_errors
    mock_data_checks_validate.assert_called_once()
    data, target = (
        mock_data_checks_validate.call_args[0][0],
        mock_data_checks_validate.call_args[1]["y"],
    )
    pd.testing.assert_frame_equal(data, infer_feature_types(X))
    pd.testing.assert_series_equal(target, infer_feature_types(y))


def test_n_splits_passed_to_ts_splitting_data_check():
    X = pd.DataFrame(pd.date_range("1/1/21", periods=100), columns=["date"])
    y = pd.Series(0 if i < 40 else 1 for i in range(100))

    problem_config = {
        "gap": 1,
        "max_delay": 1,
        "forecast_horizon": 1,
        "time_index": "date",
    }
    _, data_checks = search(
        X_train=X,
        y_train=y,
        problem_configuration=problem_config,
        problem_type="time series binary",
        n_splits=4,
    )
    assert len(data_checks[0]["details"]["invalid_splits"]) == 4


@pytest.mark.parametrize(
    "problem_config",
    [None, "missing_time_index", "missing_other_index"],
)
def test_search_data_check_error_timeseries(problem_config):
    X, y = pd.DataFrame({"features": range(30)}), pd.Series(range(30))
    problem_configuration = None

    dates = pd.date_range("2021-01-01", periods=29).append(
        pd.date_range("2021-01-31", periods=1),
    )
    X["dates"] = dates

    if not problem_config:
        problem_configuration = None
    elif problem_config == "missing_time_index":
        problem_configuration = {"gap": 4}
    elif problem_config == "missing_other_index":
        problem_configuration = {"time_index": "dates", "max_delay": 2, "gap": 2}

    with pytest.raises(
        ValueError,
        match="problem_configuration must be a dict containing",
    ):
        search(
            X_train=X,
            y_train=y,
            problem_type="time series regression",
            problem_configuration=problem_configuration,
        )


@patch("evalml.data_checks.default_data_checks.DefaultDataChecks.validate")
@patch("evalml.automl.AutoMLSearch.search")
def test_search_args(mock_automl_search, mock_data_checks_validate, X_y_binary):
    X, y = X_y_binary
    automl, _ = search(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_time=42,
        patience=3,
        tolerance=0.5,
        mode="fast",
    )
    assert automl.max_time == 42
    assert automl.patience == 3
    assert automl.tolerance == 0.5
    assert automl.max_batches == 4
    assert isinstance(automl.automl_algorithm, DefaultAlgorithm)

    automl, _ = search(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_time=42,
        patience=3,
        tolerance=0.5,
        mode="long",
    )
    assert automl.max_time == 42
    assert automl.patience == 3
    assert automl.tolerance == 0.5
    assert automl.max_batches == 999
    assert isinstance(automl.automl_algorithm, DefaultAlgorithm)

    automl, _ = search(
        X_train=X,
        y_train=y,
        problem_type="binary",
        mode="long",
    )

    assert automl.max_batches == 6
    assert isinstance(automl.automl_algorithm, DefaultAlgorithm)

    with pytest.raises(ValueError):
        search(
            X_train=X,
            y_train=y,
            problem_type="binary",
            max_time=42,
            patience=3,
            tolerance=0.5,
            mode="everything",
        )
