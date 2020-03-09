import pytest

from evalml.exceptions import ObjectiveNotFoundError
from evalml.objectives.objective_base import ObjectiveBase
from evalml.problem_types import ProblemTypes

from evalml.objectives import (  # PrecisionMacro,; PrecisionMicro,
    Precision,
    get_objective,
    get_objectives
)


def test_create_custom_objective():
    class MockEmptyObjective(ObjectiveBase):
        def objective_function(self, y_predicted, y_true, X=None):
            pass

    with pytest.raises(NameError):
        MockEmptyObjective()

    class MockNoObjectiveFunctionObjective(ObjectiveBase):
        name = "Mock objective without objective function"
        problem_type = ProblemTypes.BINARY

    with pytest.raises(TypeError):  # TODO: add match
        MockNoObjectiveFunctionObjective()


def test_get_objective():
    assert isinstance(get_objective('precision'), Precision)
    assert isinstance(get_objective(Precision()), Precision)

    with pytest.raises(ObjectiveNotFoundError):
        get_objective('this is not a valid objective')
    with pytest.raises(ObjectiveNotFoundError):
        get_objective(1)


def test_get_objectives_types():
    assert len(get_objectives(ProblemTypes.MULTICLASS)) == 14
    assert len(get_objectives(ProblemTypes.BINARY)) == 6
    assert len(get_objectives(ProblemTypes.REGRESSION)) == 6
