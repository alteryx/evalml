from unittest.mock import patch

import pandas as pd
import pytest

from evalml.automl import AutoMLSearch, search
from evalml.automl.automl_algorithm import DefaultAlgorithm
from evalml.utils import infer_feature_types


@patch("evalml.data_checks.default_data_checks.DefaultDataChecks.validate")
@patch("evalml.automl.AutoMLSearch.search")
def test_search(mock_automl_search, mock_data_checks_validate, X_y_binary):
    X, y = X_y_binary
    # this doesn't exactly match the data check results schema but its enough to trigger the error in search()
    data_check_results_expected = {"warnings": ["Warning 1", "Warning 2"]}
    mock_data_checks_validate.return_value = data_check_results_expected
    automl, data_check_results = search(X_train=X, y_train=y, problem_type="binary")
    assert isinstance(automl, AutoMLSearch)
    assert data_check_results is data_check_results_expected
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
    mock_automl_search, mock_data_checks_validate, X_y_binary
):
    X, y = X_y_binary
    # this doesn't exactly match the data check results schema but its enough to trigger the error in search()
    data_check_results_expected = {"errors": ["Error 1", "Error 2"]}
    mock_data_checks_validate.return_value = data_check_results_expected
    automl, data_check_results = search(X_train=X, y_train=y, problem_type="binary")
    assert automl is None
    assert data_check_results == data_check_results_expected
    mock_data_checks_validate.assert_called_once()
    data, target = (
        mock_data_checks_validate.call_args[0][0],
        mock_data_checks_validate.call_args[1]["y"],
    )
    pd.testing.assert_frame_equal(data, infer_feature_types(X))
    pd.testing.assert_series_equal(target, infer_feature_types(y))


@pytest.mark.parametrize(
    "problem_config", [None, "missing_date_index", "has_date_index"]
)
def test_search_data_check_error_timeseries(problem_config):
    X, y = pd.DataFrame({"features": range(30)}), pd.Series(range(30))
    problem_configuration = None

    dates = pd.date_range("2021-01-01", periods=29).append(
        pd.date_range("2021-01-31", periods=1)
    )
    X["dates"] = dates

    if problem_config == "missing_date_index":
        problem_configuration = {"gap": 4}
        with pytest.raises(
            ValueError,
            match="date_index has to be passed in problem_configuration.",
        ):
            search(
                X_train=X,
                y_train=y,
                problem_type="time series regression",
                problem_configuration=problem_configuration,
            )
    elif not problem_config:
        with pytest.raises(
            ValueError,
            match="the problem_configuration parameter must be specified.",
        ):
            search(
                X_train=X,
                y_train=y,
                problem_type="time series regression",
                problem_configuration=problem_configuration,
            )
    else:
        problem_configuration = {"date_index": "dates"}
        automl, data_check_results = search(
            X_train=X,
            y_train=y,
            problem_type="time series regression",
            problem_configuration=problem_configuration,
        )
        assert len(data_check_results["warnings"]) == 2
        assert len(data_check_results["errors"]) == 1


@patch("evalml.data_checks.default_data_checks.DefaultDataChecks.validate")
@patch("evalml.automl.AutoMLSearch.search")
def test_search_args(mock_automl_search, mock_data_checks_validate, X_y_binary):
    X, y = X_y_binary
    automl, data_check_results = search(
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
    assert automl.max_batches == 3
    assert isinstance(automl._automl_algorithm, DefaultAlgorithm)

    automl, data_check_results = search(
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
    assert automl.max_batches is None
    assert isinstance(automl._automl_algorithm, DefaultAlgorithm)

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
