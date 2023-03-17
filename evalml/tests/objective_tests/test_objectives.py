import numpy as np
import pandas as pd
import pytest
from woodwork.logical_types import BooleanNullable

from evalml.exceptions import ObjectiveCreationError, ObjectiveNotFoundError
from evalml.objectives import (
    BinaryClassificationObjective,
    CostBenefitMatrix,
    LogLossBinary,
    MulticlassClassificationObjective,
    RegressionObjective,
    get_all_objective_names,
    get_core_objective_names,
    get_core_objectives,
    get_non_core_objectives,
    get_objective,
    get_optimization_objectives,
    get_ranking_objectives,
    ranking_only_objectives,
)
from evalml.objectives.fraud_cost import FraudCost
from evalml.objectives.objective_base import ObjectiveBase
from evalml.objectives.standard_metrics import MAPE
from evalml.objectives.utils import _all_objectives_dict
from evalml.problem_types import ProblemTypes


def test_create_custom_objective():
    class MockEmptyObjective(ObjectiveBase):
        def objective_function(self, y_true, y_predicted, X=None):
            """Docstring for mock objective function."""

    with pytest.raises(TypeError):
        MockEmptyObjective()

    class MockNoObjectiveFunctionObjective(ObjectiveBase):
        problem_type = ProblemTypes.BINARY

    with pytest.raises(TypeError):
        MockNoObjectiveFunctionObjective()


@pytest.mark.parametrize("obj", _all_objectives_dict().values())
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

    with pytest.raises(TypeError):
        get_objective(obj)

    with pytest.raises(ObjectiveNotFoundError):
        get_objective("log loss")


def test_get_objective_return_instance_does_not_work_for_some_objectives():
    with pytest.raises(
        ObjectiveCreationError,
        match="In get_objective, cannot pass in return_instance=True for Cost Benefit Matrix",
    ):
        get_objective("Cost Benefit Matrix", return_instance=True)

    cbm = CostBenefitMatrix(0, 0, 0, 0)
    assert get_objective(cbm) == cbm


def test_get_objective_does_not_work_for_none_type():
    with pytest.raises(TypeError, match="Objective parameter cannot be NoneType"):
        get_objective(None)


def test_get_objective_kwargs():
    obj = get_objective(
        "cost benefit matrix",
        return_instance=True,
        true_positive=0,
        true_negative=0,
        false_positive=0,
        false_negative=0,
    )
    assert isinstance(obj, CostBenefitMatrix)


def test_can_get_only_core_and_all_objective_names():
    all_objective_names = get_all_objective_names()
    core_objective_names = get_core_objective_names()
    assert set(all_objective_names).difference(core_objective_names) == {
        c.name.lower() for c in get_non_core_objectives()
    }


def test_get_core_objectives_types():
    assert len(get_core_objectives(ProblemTypes.MULTICLASS)) == 13
    assert len(get_core_objectives(ProblemTypes.BINARY)) == 8
    assert len(get_core_objectives(ProblemTypes.REGRESSION)) == 7
    assert len(get_core_objectives(ProblemTypes.TIME_SERIES_REGRESSION)) == 7


def test_get_optimization_objectives_types():
    assert len(get_optimization_objectives(ProblemTypes.MULTICLASS)) == 13
    assert len(get_optimization_objectives(ProblemTypes.BINARY)) == 8
    assert len(get_optimization_objectives(ProblemTypes.REGRESSION)) == 7
    assert len(get_optimization_objectives(ProblemTypes.TIME_SERIES_REGRESSION)) == 7


def test_get_ranking_objectives_types():
    assert len(get_ranking_objectives(ProblemTypes.MULTICLASS)) == 16
    assert len(get_ranking_objectives(ProblemTypes.BINARY)) == 9
    assert len(get_ranking_objectives(ProblemTypes.REGRESSION)) == 9
    assert len(get_ranking_objectives(ProblemTypes.TIME_SERIES_REGRESSION)) == 10


def test_optimization_excludes_ranking():
    objs = get_optimization_objectives(ProblemTypes.BINARY)
    for obj in objs:
        assert obj.__class__ not in ranking_only_objectives()


def test_get_time_series_objectives_types(time_series_objectives):
    assert len(time_series_objectives) == 10


def test_objective_outputs(
    X_y_binary,
    X_y_multi,
):
    _, y_binary = X_y_binary
    y_binary_np = y_binary.values
    _, y_multi = X_y_multi
    y_multi_np = y_multi.values
    y_true_multi_np = y_multi_np
    y_pred_multi_np = y_multi_np
    # convert to a simulated predicted probability, which must range between 0 and 1
    classes = np.unique(y_multi_np)
    y_pred_proba_multi_np = np.concatenate(
        [(y_multi_np == val).astype(float).reshape(-1, 1) for val in classes],
        axis=1,
    )

    all_objectives = (
        get_core_objectives("binary")
        + get_core_objectives("multiclass")
        + get_core_objectives("regression")
    )

    for objective in all_objectives:
        print("Testing objective {}".format(objective.name))
        expected_value = 1.0 if objective.greater_is_better else 0.0
        if isinstance(objective, (RegressionObjective, BinaryClassificationObjective)):
            np.testing.assert_almost_equal(
                objective.score(y_binary_np, y_binary_np),
                expected_value,
            )
            np.testing.assert_almost_equal(
                objective.score(pd.Series(y_binary_np), pd.Series(y_binary_np)),
                expected_value,
            )
        if isinstance(objective, MulticlassClassificationObjective):
            y_predicted = y_pred_multi_np
            y_predicted_pd = pd.Series(y_predicted)
            if objective.score_needs_proba:
                y_predicted = y_pred_proba_multi_np
                y_predicted_pd = pd.DataFrame(y_predicted)
            np.testing.assert_almost_equal(
                objective.score(y_true_multi_np, y_predicted),
                expected_value,
            )
            np.testing.assert_almost_equal(
                objective.score(pd.Series(y_true_multi_np), y_predicted_pd),
                expected_value,
            )


def test_is_defined_for_problem_type():
    assert LogLossBinary.is_defined_for_problem_type(ProblemTypes.BINARY)
    assert LogLossBinary.is_defined_for_problem_type("binary")
    assert not LogLossBinary.is_defined_for_problem_type(ProblemTypes.MULTICLASS)


@pytest.mark.parametrize("obj", _all_objectives_dict().values())
def test_get_objectives_all_expected_ranges(obj):
    assert len(obj.expected_range) == 2


@pytest.mark.parametrize("obj", [obj for obj in _all_objectives_dict().values()])
@pytest.mark.parametrize(
    "nullable_y_true_ltype",
    ["BooleanNullable", "IntegerNullable", "AgeNullable"],
)
def test_objectives_support_nullable_types(
    nullable_y_true_ltype,
    obj,
    nullable_type_target,
):
    y_true = nullable_type_target(ltype=nullable_y_true_ltype, has_nans=False)
    y_pred = pd.Series([0, 1, 0, 1, 1] * 4, dtype="int64")
    X = None

    # Instantiate objective with any needed parameters
    if obj == CostBenefitMatrix:
        obj = obj(
            true_positive=10,
            true_negative=-1,
            false_positive=-7,
            false_negative=-2,
        )
    else:
        obj = obj()

    # Make any changes to data needed for objective
    if isinstance(obj, FraudCost):
        # FraudCost needs an "amount" column
        X = pd.DataFrame({"amount": [100, 5, 250, 89] * 5})
    elif isinstance(obj, MAPE):
        if isinstance(y_true.ww.logical_type, BooleanNullable):
            pytest.skip("MAPE doesn't support inputs containing 0")
        # Replace numeric inputs containing 0
        y_true = y_true.ww.replace({0: 10})
        y_pred = y_pred.replace({0: 10})

    score = obj.score(y_true=y_true, y_predicted=y_pred, X=X)
    assert not pd.isna(score)
