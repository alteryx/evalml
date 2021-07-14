from unittest.mock import patch

import numpy as np
import pytest

from evalml.automl.automl_algorithm import EvalMLAlgorithm
from evalml.model_family import ModelFamily
from evalml.pipelines.components import (
    ElasticNetClassifier,
    ElasticNetRegressor,
)

from evalml.problem_types import ProblemTypes


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


def add_result(algo, batch):
    scores = np.arange(0, len(batch))
    for score, pipeline in zip(scores, batch):
        algo.add_result(score, pipeline, {"id": algo.pipeline_number})


@patch("evalml.pipelines.components.FeatureSelector.get_names")
@pytest.mark.parametrize(
    "automl_type",
    [ProblemTypes.BINARY, ProblemTypes.MULTICLASS, ProblemTypes.REGRESSION],
)
def test_evalml_algorithm_short(
    mock_get_names,
    automl_type,
    X_y_binary,
    X_y_multi,
    X_y_regression,
):

    if automl_type == ProblemTypes.BINARY:
        X, y = X_y_binary
        fs = "RF Classifier Select From Model"
    elif automl_type == ProblemTypes.MULTICLASS:
        X, y = X_y_multi
        fs = "RF Classifier Select From Model"
    elif automl_type == ProblemTypes.REGRESSION:
        X, y = X_y_regression
        fs = "RF Regressor Select From Model"
    mock_get_names.return_value = ["0", "1", "2"]
    problem_type = automl_type
    sampler_name = None
    algo = EvalMLAlgorithm(X, y, problem_type, sampler_name)
    naive_model_families = [ModelFamily.LINEAR_MODEL, ModelFamily.RANDOM_FOREST]

    first_batch = algo.next_batch()
    assert len(first_batch) == 2
    for pipeline in first_batch:
        assert pipeline.model_family in naive_model_families

    add_result(algo, first_batch)

    second_batch = algo.next_batch()
    assert len(second_batch) == 2
    for pipeline in second_batch:
        assert pipeline.model_family in naive_model_families
        assert pipeline.get_component(fs)

    add_result(algo, second_batch)
    assert algo._selected_cols == ["0", "1", "2"]

    final_batch = algo.next_batch()
    # assert len(first_batch) == 2
    for pipeline in final_batch:
        if not isinstance(
            pipeline.estimator, (ElasticNetClassifier, ElasticNetRegressor)
        ):
            assert pipeline.model_family not in naive_model_families
        select = pipeline.get_component("Select Columns Transformer")
        assert select.parameters["columns"] == algo._selected_cols

    # add_result(algo, second_batch)
    # assert algo._selected_cols == ["0", "1", "2"]
