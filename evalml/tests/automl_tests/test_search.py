from unittest.mock import patch

from evalml.automl import AutoMLSearch, search
from evalml.utils import infer_feature_types


@patch('evalml.data_checks.default_data_checks.DefaultDataChecks.validate')
@patch('evalml.automl.AutoMLSearch.search')
def test_search(mock_automl_search, mock_data_checks_validate, X_y_binary):
    X, y = X_y_binary
    # this doesn't exactly match the data check results schema but its enough to trigger the error in search()
    data_check_results_expected = {'warnings': ['Warning 1', 'Warning 2']}
    mock_data_checks_validate.return_value = data_check_results_expected
    automl, data_check_results = search(X_train=X, y_train=y, problem_type='binary')
    assert isinstance(automl, AutoMLSearch)
    assert data_check_results is data_check_results_expected
    mock_data_checks_validate.assert_called_once()
    mock_data_checks_validate.assert_called_with(infer_feature_types(X), y=infer_feature_types(y))
    mock_automl_search.assert_called_once()


@patch('evalml.data_checks.default_data_checks.DefaultDataChecks.validate')
@patch('evalml.automl.AutoMLSearch.search')
def test_search_data_check_error(mock_automl_search, mock_data_checks_validate, X_y_binary):
    X, y = X_y_binary
    # this doesn't exactly match the data check results schema but its enough to trigger the error in search()
    data_check_results_expected = {'errors': ['Error 1', 'Error 2']}
    mock_data_checks_validate.return_value = data_check_results_expected
    automl, data_check_results = search(X_train=X, y_train=y, problem_type='binary')
    assert automl is None
    assert data_check_results == data_check_results_expected
    mock_data_checks_validate.assert_called_once()
    mock_data_checks_validate.assert_called_with(infer_feature_types(X), y=infer_feature_types(y))
    mock_automl_search.assert_not_called()


@patch('evalml.data_checks.default_data_checks.DefaultDataChecks.validate')
@patch('evalml.automl.AutoMLSearch.search')
def test_search_kwargs(mock_automl_search, mock_data_checks_validate, X_y_binary):
    X, y = X_y_binary
    automl, data_check_results = search(X_train=X, y_train=y, problem_type='binary', max_iterations=42)
    assert automl.max_iterations == 42
