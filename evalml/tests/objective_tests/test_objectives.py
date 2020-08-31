import numpy as np
import pandas as pd
import pytest

from evalml.automl import AutoMLSearch
from evalml.exceptions import ObjectiveNotFoundError
from evalml.objectives import (
    BinaryClassificationObjective,
    CostBenefitMatrix,
    MulticlassClassificationObjective,
    RegressionObjective,
    get_objective,
    get_objectives
)
from evalml.objectives.objective_base import ObjectiveBase
from evalml.problem_types import ProblemTypes
from evalml.utils.gen_utils import _get_subclasses

_not_allowed_in_automl = AutoMLSearch._objectives_not_allowed_in_automl

binary_objectives = [obj() for obj in get_objectives(ProblemTypes.BINARY) if obj not in _not_allowed_in_automl]
multiclass_objectives = [obj() for obj in get_objectives(ProblemTypes.MULTICLASS) if obj not in _not_allowed_in_automl]
regression_objectives = [obj() for obj in get_objectives(ProblemTypes.REGRESSION) if obj not in _not_allowed_in_automl]


def test_create_custom_objective():
    class MockEmptyObjective(ObjectiveBase):
        def objective_function(self, y_true, y_predicted, X=None):
            """Docstring for mock objective function"""

    with pytest.raises(TypeError):
        MockEmptyObjective()

    class MockNoObjectiveFunctionObjective(ObjectiveBase):
        problem_type = ProblemTypes.BINARY

    with pytest.raises(TypeError):
        MockNoObjectiveFunctionObjective()


@pytest.mark.parametrize("obj", _get_subclasses(ObjectiveBase))
def test_get_objective_works_for_names_of_defined_objectives(obj):
    assert get_objective(obj.name) == obj
    assert get_objective(obj.name.lower()) == obj

    args = []
    if obj == CostBenefitMatrix:
        args = [0] * 4
    assert isinstance(get_objective(obj(*args)), obj)


def test_get_objective_does_raises_error_for_incorrect_name_or_random_class():

    class InvalidObjective:
        pass

    obj = InvalidObjective()

    with pytest.raises(ObjectiveNotFoundError):
        get_objective(obj)

    with pytest.raises(ObjectiveNotFoundError):
        get_objective("log loss")


def test_get_objective_return_instance_does_not_work_for_some_objectives():

    with pytest.raises(TypeError, match="In get_objective, cannot pass in return_instance=True for Cost Benefit Matrix"):
        get_objective("Cost Benefit Matrix", return_instance=True)


def test_get_objective_does_not_work_for_none_type():
    with pytest.raises(TypeError, match="Objective parameter cannot be NoneType"):
        get_objective(None)


def test_get_objectives_types():

    assert len(get_objectives(ProblemTypes.MULTICLASS)) == 16
    assert len(get_objectives(ProblemTypes.BINARY)) == 11
    assert len(get_objectives(ProblemTypes.REGRESSION)) == 9


def test_objective_outputs(X_y_binary, X_y_multi):
    _, y_binary_np = X_y_binary
    assert isinstance(y_binary_np, np.ndarray)
    _, y_multi_np = X_y_multi
    assert isinstance(y_multi_np, np.ndarray)
    y_true_multi_np = y_multi_np
    y_pred_multi_np = y_multi_np
    # convert to a simulated predicted probability, which must range between 0 and 1
    classes = np.unique(y_multi_np)
    y_pred_proba_multi_np = np.concatenate([(y_multi_np == val).astype(float).reshape(-1, 1) for val in classes], axis=1)

    all_objectives = binary_objectives + regression_objectives + multiclass_objectives

    for objective in all_objectives:
        print('Testing objective {}'.format(objective.name))
        expected_value = 1.0 if objective.greater_is_better else 0.0
        if isinstance(objective, (RegressionObjective, BinaryClassificationObjective)):
            np.testing.assert_almost_equal(objective.score(y_binary_np, y_binary_np), expected_value)
            np.testing.assert_almost_equal(objective.score(pd.Series(y_binary_np), pd.Series(y_binary_np)), expected_value)
        if isinstance(objective, MulticlassClassificationObjective):
            y_predicted = y_pred_multi_np
            y_predicted_pd = pd.Series(y_predicted)
            if objective.score_needs_proba:
                y_predicted = y_pred_proba_multi_np
                y_predicted_pd = pd.DataFrame(y_predicted)
            np.testing.assert_almost_equal(objective.score(y_true_multi_np, y_predicted), expected_value)
            np.testing.assert_almost_equal(objective.score(pd.Series(y_true_multi_np), y_predicted_pd), expected_value)
