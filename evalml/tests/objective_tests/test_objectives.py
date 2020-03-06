import pytest

from evalml.exceptions import ObjectiveNotFoundError
from evalml.problem_types import ProblemTypes

from evalml.objectives import (  # PrecisionMacro,; PrecisionMicro,
    Precision,
    get_objective,
    get_objectives
)


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


# todo: use to test logic?
# def test_binary_average(X_y):
#     X, y = X_y

#     objective = Precision()
#     pipeline = LogisticRegressionBinaryPipeline(penalty='l2', C=1.0, impute_strategy='mean', number_features=0)
#     pipeline.fit(X, y, objective)
#     y_pred = pipeline.predict(X, objective=objective)

#     # assert Precision().score(y, y_pred) == PrecisionMicro().score(y, y_pred)
#     assert Precision().score(y, y_pred) == PrecisionMacro().score(y, y_pred)
