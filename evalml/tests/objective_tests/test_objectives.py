import numpy as np
import pandas as pd
import pytest

from evalml.exceptions import ObjectiveNotFoundError
from evalml.objectives import Precision, get_objective, get_objectives
from evalml.objectives.objective_base import ObjectiveBase
from evalml.problem_types import ProblemTypes


def test_create_custom_objective():
    class MockEmptyObjective(ObjectiveBase):
        def objective_function(self, y_true, y_predicted, X=None):
            pass

    with pytest.raises(TypeError):
        MockEmptyObjective()

    class MockNoObjectiveFunctionObjective(ObjectiveBase):
        name = "Mock objective without objective function"
        problem_type = ProblemTypes.BINARY

    with pytest.raises(TypeError):
        MockNoObjectiveFunctionObjective()


def test_get_objective():
    assert isinstance(get_objective('precision'), Precision)
    assert isinstance(get_objective(Precision()), Precision)

    with pytest.raises(TypeError):
        get_objective(None)
    with pytest.raises(ObjectiveNotFoundError):
        get_objective('this is not a valid objective')
    with pytest.raises(ObjectiveNotFoundError):
        get_objective(1)


def test_get_objectives_types():

    assert len(get_objectives(ProblemTypes.MULTICLASS)) == 13
    assert len(get_objectives(ProblemTypes.BINARY)) == 7
    assert len(get_objectives(ProblemTypes.REGRESSION)) == 7


def test_objective_output_type(X_y):
    _, y_np = X_y
    print(y_np)
    assert isinstance(y_np, np.ndarray)
    all_objectives = (get_objectives(ProblemTypes.REGRESSION) +
                      get_objectives(ProblemTypes.BINARY) +
                      get_objectives(ProblemTypes.MULTICLASS))
    for objective in all_objectives:
        print('Testing objective {}'.format(objective.name))
        expected_value = 1.0 if objective.greater_is_better else 0.0
        np.testing.assert_almost_equal(objective.score(y_np, y_np), expected_value)
        np.testing.assert_almost_equal(objective.score(pd.Series(y_np), pd.Series(y_np)), expected_value)
