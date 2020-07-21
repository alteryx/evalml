from itertools import product
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from evalml.model_family.model_family import ModelFamily
from evalml.pipelines import (
    BinaryClassificationPipeline,
    MeanBaselineRegressionPipeline,
    ModeBaselineBinaryPipeline,
    ModeBaselineMulticlassPipeline,
    MulticlassClassificationPipeline,
    RegressionPipeline
)
from evalml.pipelines.components import (
    CatBoostClassifier,
    LinearRegressor,
    RandomForestClassifier,
    XGBoostClassifier,
    XGBoostRegressor
)
from evalml.pipelines.components.utils import _all_estimators_used_in_search
from evalml.pipelines.explanations._algorithms import (
    _compute_shap_values,
    _normalize_values
)
from evalml.pipelines.utils import make_pipeline
from evalml.problem_types.problem_types import ProblemTypes


def make_test_pipeline(estimator, base_class):

    class Pipeline(base_class):
        component_graph = [estimator]

    return Pipeline


baseline_message = "You passed in a baseline pipeline. These are simple enough that SHAP values are not needed."
xg_boost_message = "SHAP values cannot currently be computed for xgboost models."
catboost_message = "SHAP values cannot currently be computed for catboost models for multiclass problems."
datatype_message = "Unknown shap_values datatype <class 'unittest.mock.MagicMock'>!"
data_message = "You must pass in a value for parameter 'training_data' when the pipeline does not have a tree-based estimator. Current estimator model family is Linear."


@pytest.mark.parametrize("pipeline,exception,match", [(MeanBaselineRegressionPipeline, ValueError, baseline_message),
                                                      (ModeBaselineBinaryPipeline, ValueError, baseline_message),
                                                      (ModeBaselineMulticlassPipeline, ValueError, baseline_message),
                                                      (make_test_pipeline(CatBoostClassifier, MulticlassClassificationPipeline), NotImplementedError, catboost_message),
                                                      (make_test_pipeline(XGBoostClassifier, BinaryClassificationPipeline), NotImplementedError, xg_boost_message),
                                                      (make_test_pipeline(XGBoostClassifier, MulticlassClassificationPipeline), NotImplementedError, xg_boost_message),
                                                      (make_test_pipeline(XGBoostRegressor, RegressionPipeline), NotImplementedError, xg_boost_message),
                                                      (make_test_pipeline(RandomForestClassifier, BinaryClassificationPipeline), ValueError, datatype_message),
                                                      (make_test_pipeline(LinearRegressor, RegressionPipeline), ValueError, data_message)])
@patch("evalml.pipelines.explanations._algorithms.shap.TreeExplainer")
def test_value_errors_raised(mock_tree_explainer, pipeline, exception, match):

    with pytest.raises(exception, match=match):
        _ = _compute_shap_values(pipeline({}), pd.DataFrame(np.random.random((2, 16))))


N_CLASSES_BINARY = 2
N_CLASSES_MULTICLASS = 3
N_FEATURES = 20


def check_classification(shap_values, is_binary, n_points_to_explain):
    """Checks whether the SHAP values computed for a classifier match our expectations."""
    assert isinstance(shap_values, list), "For binary classification, returned values must be a list"
    assert all(isinstance(class_values, dict) for class_values in shap_values), "Not all list elements are lists!"
    if is_binary:
        assert len(shap_values) == N_CLASSES_BINARY, "A dictionary should be returned for each class!"
    else:
        assert len(shap_values) == N_CLASSES_MULTICLASS, "A dictionary should be returned for each class!"
    assert all(len(values) == N_FEATURES for values in shap_values), "A SHAP value must be computed for every feature!"
    for class_values in shap_values:
        assert all(isinstance(feature, list) for feature in class_values.values()), "Every value in the dict must be a list!"
        assert all(len(v) == n_points_to_explain for v in class_values.values()), "A SHAP value must be computed for every data point to explain!"


def check_regression(shap_values, n_points_to_explain):
    """Checks whether the SHAP values computed for a regressor match our expectations."""
    assert isinstance(shap_values, dict), "For regression, returned values must be a dictionary!"
    assert len(shap_values) == N_FEATURES, "A SHAP value should be computed for every feature!"
    assert all(isinstance(feature, list) for feature in shap_values.values()), "Every value in the dict must be a list!"
    assert all(len(v) == n_points_to_explain for v in shap_values.values()), "A SHAP value must be computed for every data point to explain!"


def not_xgboost_or_baseline(estimator):
    """Filter out xgboost and baselines for next test since they are not supported."""
    return estimator.model_family not in {ModelFamily.XGBOOST, ModelFamily.BASELINE}


def calculate_shap_for_test(training_data, y, pipeline_class, n_points_to_explain):
    """Helper function to compute the SHAP values for n_points_to_explain for a given pipeline."""
    pipeline = pipeline_class({}, random_state=0)
    points_to_explain = training_data[:n_points_to_explain]
    pipeline.fit(training_data, y)
    return _compute_shap_values(pipeline, points_to_explain, training_data)


interpretable_estimators = filter(not_xgboost_or_baseline, _all_estimators_used_in_search)
all_problems = [ProblemTypes.REGRESSION, ProblemTypes.BINARY, ProblemTypes.MULTICLASS]
all_n_points_to_explain = [1, 5]


@pytest.mark.parametrize("estimator, problem_type,n_points_to_explain",
                         product(interpretable_estimators, all_problems, all_n_points_to_explain))
def test_shap(estimator, problem_type, n_points_to_explain, X_y_binary, X_y_multi, X_y_regression):

    if problem_type not in estimator.supported_problem_types:
        pytest.skip("Skipping because estimator and pipeline are not compatible.")

    if problem_type == ProblemTypes.MULTICLASS and estimator.model_family == ModelFamily.CATBOOST:
        pytest.skip("Skipping Catboost for multiclass problems.")

    if problem_type == ProblemTypes.BINARY:
        training_data, y = X_y_binary
        pipeline_class = make_pipeline(training_data, y, estimator, problem_type)
        shap_values = calculate_shap_for_test(training_data, y, pipeline_class, n_points_to_explain)

        # CatBoostClassifiers on binary problems only output one value
        if isinstance(shap_values, dict):
            check_regression(shap_values, n_points_to_explain=n_points_to_explain)
        else:
            check_classification(shap_values, True, n_points_to_explain=n_points_to_explain)
    elif problem_type == ProblemTypes.MULTICLASS:
        training_data, y = X_y_multi
        pipeline_class = make_pipeline(training_data, y, estimator, problem_type)
        shap_values = calculate_shap_for_test(training_data, y, pipeline_class, n_points_to_explain)
        check_classification(shap_values, False, n_points_to_explain)
    else:
        training_data, y = X_y_regression
        pipeline_class = make_pipeline(training_data, y, estimator, problem_type)
        shap_values = calculate_shap_for_test(training_data, y, pipeline_class, n_points_to_explain)
        check_regression(shap_values, n_points_to_explain)


@pytest.mark.parametrize("values,match", [(1, "^Unsupported data type for _normalize_values"),
                                          ({"a": [10.00001, 9.9999], "b": [10.00001, 9.9999]},
                                           "^Cannot normalize values where curr_min and curr_max are almost equal"),
                                          ([{"a": [5, 5, 5], "b": [5, 5, 5]}] * 2,
                                           "^Cannot normalize values where curr_min and curr_max are almost equal")])
def test_normalize_values_exceptions(values, match):

    with pytest.raises(ValueError, match=match):
        _normalize_values(values)


@pytest.mark.parametrize("values,answer", [({"a": [-0.5, 0, 0.5], "b": [0.1, -0.6, 0.2]},
                                            {"a": [-0.8181, 0.0909, 1.0], "b": [0.2727, -1.0, 0.4545]}),
                                           ([{"a": [-0.5, 0, 0.5], "b": [0.1, -0.6, 0.2]}] * 2,
                                            [{"a": [-0.8181, 0.0909, 1.0], "b": [0.2727, -1.0, 0.4545]}] * 2),
                                           ({"a": [0, 0]}, {"a": [0, 0]}),
                                           ([{"a": [0]}] * 10, [{"a": [0]}] * 10),
                                           ({"a": [5], "b": [20], "c": [22]},
                                            {"a": [-1], "b": [0.7647], "c": [1.0]})])
def test_normalize_values(values, answer):

    def check_equal_dicts(normalized, answer):
        assert set(normalized.keys()) == set(answer)
        for key in normalized:
            np.testing.assert_almost_equal(normalized[key], answer[key], decimal=4)

    normalized = _normalize_values(values)
    if isinstance(normalized, dict):
        check_equal_dicts(normalized, answer)

    else:
        assert len(normalized) == len(answer)
        for values, correct in zip(normalized, answer):
            check_equal_dicts(values, correct)
