import pandas as pd
import pytest
from itertools import product

from evalml.tests.model_understanding_tests.prediction_explanations_tests.test_algorithms import (
    interpretable_estimators,
    all_problems,
)

from evalml.model_family.model_family import ModelFamily
from evalml.model_understanding.graphs import force_plot
from evalml.pipelines.utils import make_pipeline
from evalml.problem_types.problem_types import ProblemTypes


@pytest.mark.parametrize("estimator,problem_type,n_points_to_explain",
                         product(interpretable_estimators, all_problems, [[0], [0, 1, 2, 3, 4]]))
def test_plot(estimator, problem_type, n_points_to_explain, X_y_binary, X_y_multi, X_y_regression,
              helper_functions):

    if problem_type not in estimator.supported_problem_types:
        pytest.skip("Skipping because estimator and pipeline are not compatible.")

    if problem_type == ProblemTypes.MULTICLASS and estimator.model_family == ModelFamily.CATBOOST:
        pytest.skip("Skipping Catboost for multiclass problems.")

    if problem_type == ProblemTypes.BINARY:
        training_data, y = X_y_binary
        is_binary = True
    elif problem_type == ProblemTypes.MULTICLASS:
        training_data, y = X_y_multi
        is_binary = False
    else:
        training_data, y = X_y_regression

    training_data = pd.DataFrame(training_data)

    pipeline_class = make_pipeline(training_data, y, estimator, problem_type)
    pipeline = helper_functions.safe_init_pipeline_with_njobs_1(pipeline_class)

    pipeline.fit(training_data, y)

    force_plot(pipeline, n_points_to_explain, training_data)


@pytest.mark.parametrize("rows_to_explain", ([0], [0, 1, 2, 3, 4]))
def test_my_foce_plot(rows_to_explain, helper_functions):
    import shap
    from evalml.pipelines.components.estimators import LightGBMRegressor
    training_data, y = shap.datasets.boston()
    pipeline_class = make_pipeline(training_data, y, LightGBMRegressor, problem_type="regression")
    pipeline = helper_functions.safe_init_pipeline_with_njobs_1(pipeline_class)
    pipeline.fit(training_data, y)
    shap.initjs()

    force_plot(pipeline, rows_to_explain, training_data, matplotlib=False)
