from unittest.mock import patch

import pytest

from evalml import AutoMLSearch
from evalml.exceptions import ObjectiveNotFoundError


def test_init_problem_type_error():
    with pytest.raises(ValueError, match=r"choose one of \(binary, multiclass, regression\) as problem_type"):
        AutoMLSearch()

    with pytest.raises(ValueError, match=r"choose one of \(binary, multiclass, regression\) as problem_type"):
        AutoMLSearch(problem_type='multi')


def test_init_objective():
    defaults = {'multiclass': 'Log Loss Multiclass', 'binary': 'Log Loss Binary', 'regression': 'R2'}
    for problem_type in defaults:
        error_automl = AutoMLSearch(problem_type=problem_type)
        assert error_automl.objective.name == defaults[problem_type]

    with pytest.raises(ObjectiveNotFoundError, match="Could not find the specified objective."):
        AutoMLSearch(problem_type='binary', objective='binary')


@patch('evalml.automl.auto_search_base.AutoSearchBase.search')
def test_checks_at_search_time(mock_search, dummy_regression_pipeline, X_y_multi):
    X, y = X_y_multi

    error_text = "in search, problem_type mismatches label type."
    mock_search.side_effect = ValueError(error_text)

    error_automl = AutoMLSearch(problem_type='regression', objective="R2")
    with pytest.raises(ValueError, match=error_text):
        error_automl.search(X, y)

    error_text = "in search, problem_type mismatches allowed_pipelines."
    mock_search.side_effect = ValueError(error_text)

    allowed_pipelines = [dummy_regression_pipeline.__class__]
    error_automl = AutoMLSearch(problem_type='binary', allowed_pipelines=allowed_pipelines)
    with pytest.raises(ValueError, match=error_text):
        error_automl.search(X, y)
