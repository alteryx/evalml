from evalml import pipelines
from evalml.automl.automl_algorithm import EvalMLAlgorithm
from evalml.problem_types import ProblemTypes
from evalml.model_family import ModelFamily

import pytest


def test_evalml_algorithm_init(X_y_binary):
    X, y = X_y_binary
    problem_type = "binary"
    sampler_name = "Undersampler"

    algo = EvalMLAlgorithm(X, y, problem_type, sampler_name)

    assert algo.problem_type == problem_type
    assert algo._sampler_name == sampler_name
    assert algo.pipeline_number == 0
    assert algo.batch_number == 0
    assert algo.allowed_pipelines == []


def test_evalml_algorithm_custom_hyperparameters_error(X_y_binary):
    X, y = X_y_binary
    problem_type = "binary"
    sampler_name = "Undersampler"

    custom_hyperparameters = [
        {"Imputer": {"numeric_imput_strategy": ["median"]}},
        {"One Hot Encoder": {"value1": ["value2"]}},
    ]

    with pytest.raises(
        ValueError, match="If custom_hyperparameters provided, must be of type dict"
    ):
        EvalMLAlgorithm(
            X,
            y,
            problem_type,
            sampler_name,
            custom_hyperparameters=custom_hyperparameters,
        )


@pytest.mark.parametrize(
    "automl_type",
    [ProblemTypes.BINARY, ProblemTypes.MULTICLASS, ProblemTypes.REGRESSION],
)
def test_evalml_algorithm_first_batch(
    automl_type,
    X_y_binary,
    X_y_multi,
    X_y_regression,):

    if automl_type == ProblemTypes.BINARY:
        X, y = X_y_binary
    elif automl_type == ProblemTypes.MULTICLASS:
        X, y = X_y_multi
    elif automl_type == ProblemTypes.REGRESSION:
        X, y = X_y_regression

    problem_type = "binary"
    sampler_name = "Undersampler"
    algo = EvalMLAlgorithm(X, y, problem_type, sampler_name)
    first_batch = algo.next_batch()
    model_families = [ModelFamily.LINEAR_MODEL, ModelFamily.RANDOM_FOREST]
    assert len(first_batch) == 2
    for pipeline in first_batch:
        assert pipeline.model_family in model_families
    