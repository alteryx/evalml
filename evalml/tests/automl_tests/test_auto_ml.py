import pytest

from evalml import AutoMLSearch
from evalml.exceptions import ObjectiveNotFoundError


def test_init_problem_type_error():
    with pytest.raises(ValueError, match="choose one of (binary, multiclass, regression) as problem_type"):
        AutoMLSearch()
    
    with pytest.raises(ValueError, match= "choose one of (binary, multiclass, regression) as problem_type"):
        AutoMLSearch(problem_type='multi')


def test_init_objective():
    defaults = {'multiclass':'log_loss_multi', 'binary':'log_loss_binary', 'regression':'R2'}
    for problem_type in defaults:
        automl = AutoMLSearch(problem_type=problem_type)
        assert automl.objective.value == defaults[problem_type]

    with pytest.raises(ObjectiveNotFoundError, match="Could not find the specified objective."):
        AutoMLSearch(problem_type='binary', objective='binary')
