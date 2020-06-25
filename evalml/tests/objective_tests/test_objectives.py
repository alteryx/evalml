import numpy as np
import pandas as pd
import pytest

from evalml.exceptions import ObjectiveNotFoundError
from evalml.objectives import (
    BinaryClassificationObjective,
    MulticlassClassificationObjective,
    Precision,
    RegressionObjective,
    get_default_objective,
    get_objective,
    get_objectives
)
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


def test_objective_outputs(X_y, X_y_multi):
    _, y_binary_np = X_y
    assert isinstance(y_binary_np, np.ndarray)
    _, y_multi_np = X_y_multi
    assert isinstance(y_multi_np, np.ndarray)
    y_true_multi_np = y_multi_np
    y_pred_multi_np = y_multi_np
    # convert to a simulated predicted probability, which must range between 0 and 1
    classes = np.unique(y_multi_np)
    y_pred_proba_multi_np = np.concatenate([(y_multi_np == val).astype(float).reshape(-1, 1) for val in classes], axis=1)

    all_objectives = (get_objectives(ProblemTypes.REGRESSION) +
                      get_objectives(ProblemTypes.BINARY) +
                      get_objectives(ProblemTypes.MULTICLASS))
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


def test_default_objective():
    correct_matches = {ProblemTypes.MULTICLASS: 'Log Loss Multiclass',
                       ProblemTypes.BINARY: 'Log Loss Binary',
                       ProblemTypes.REGRESSION: 'R2'}
    for problem_type in correct_matches:
        assert get_default_objective(problem_type).name == correct_matches[problem_type]
        assert get_default_objective(problem_type.name).name == correct_matches[problem_type]

    with pytest.raises(KeyError, match="does not exist"):
        get_default_objective('fake_class')
