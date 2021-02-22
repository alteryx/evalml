from itertools import product
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from evalml.model_family.model_family import ModelFamily
from evalml.model_understanding.prediction_explanations._algorithms import (
    _compute_shap_values,
    _create_dictionary,
    _normalize_shap_values
)
from evalml.pipelines import (
    BinaryClassificationPipeline,
    MeanBaselineRegressionPipeline,
    ModeBaselineBinaryPipeline,
    ModeBaselineMulticlassPipeline,
    MulticlassClassificationPipeline,
    RegressionPipeline,
    TimeSeriesBaselineRegressionPipeline
)
from evalml.pipelines.components import (
    CatBoostClassifier,
    LinearRegressor,
    RandomForestClassifier,
    XGBoostClassifier,
    XGBoostRegressor
)
from evalml.pipelines.components.utils import _all_estimators_used_in_search
from evalml.pipelines.utils import make_pipeline
from evalml.problem_types.problem_types import ProblemTypes


def make_test_pipeline(estimator, base_class):
    """Make an estimator-only pipeline.

    This is helps test the exceptions raised in _compute_shap_values without having to use make_pipeline
    (which needs training data to be passed in).
    """

    class Pipeline(base_class):
        component_graph = [estimator]
        name = estimator.name

    return Pipeline


baseline_message = "You passed in a baseline pipeline. These are simple enough that SHAP values are not needed."
xg_boost_message = "SHAP values cannot currently be computed for xgboost models."
catboost_message = "SHAP values cannot currently be computed for catboost models for multiclass problems."
datatype_message = "^Unknown shap_values datatype"
data_message = "You must pass in a value for parameter 'training_data' when the pipeline does not have a tree-based estimator. Current estimator model family is Linear."


@pytest.mark.parametrize("pipeline,exception,match", [(MeanBaselineRegressionPipeline, ValueError, baseline_message),
                                                      (ModeBaselineBinaryPipeline, ValueError, baseline_message),
                                                      (ModeBaselineMulticlassPipeline, ValueError, baseline_message),
                                                      (TimeSeriesBaselineRegressionPipeline, ValueError, baseline_message),
                                                      (make_test_pipeline(CatBoostClassifier, MulticlassClassificationPipeline), NotImplementedError, catboost_message),
                                                      (make_test_pipeline(XGBoostClassifier, BinaryClassificationPipeline), NotImplementedError, xg_boost_message),
                                                      (make_test_pipeline(XGBoostClassifier, MulticlassClassificationPipeline), NotImplementedError, xg_boost_message),
                                                      (make_test_pipeline(XGBoostRegressor, RegressionPipeline), NotImplementedError, xg_boost_message),
                                                      (make_test_pipeline(RandomForestClassifier, BinaryClassificationPipeline), ValueError, datatype_message),
                                                      (make_test_pipeline(LinearRegressor, RegressionPipeline), ValueError, data_message)])
@patch("evalml.model_understanding.prediction_explanations._algorithms.shap.TreeExplainer")
def test_value_errors_raised(mock_tree_explainer, pipeline, exception, match):

    if "xgboost" in pipeline.name.lower():
        pytest.importorskip("xgboost", "Skipping test because xgboost is not installed.")
    if "catboost" in pipeline.name.lower():
        pytest.importorskip("catboost", "Skipping test because catboost is not installed.")

    with pytest.raises(exception, match=match):
        _ = _compute_shap_values(pipeline({"pipeline": {"gap": 1, "max_delay": 1}}), pd.DataFrame(np.random.random((2, 16))))


def test_create_dictionary_exception():
    with pytest.raises(ValueError, match="SHAP values must be stored in a numpy array!"):
        _create_dictionary([1, 2, 3], ["a", "b", "c"])


N_CLASSES_BINARY = 2
N_CLASSES_MULTICLASS = 3
N_FEATURES = 20


def calculate_shap_for_test(training_data, y, pipeline, n_points_to_explain):
    """Helper function to compute the SHAP values for n_points_to_explain for a given pipeline."""
    points_to_explain = training_data[:n_points_to_explain]
    pipeline.fit(training_data, y)
    return _compute_shap_values(pipeline, pd.DataFrame(points_to_explain), training_data)


interpretable_estimators = [e for e in _all_estimators_used_in_search() if e.model_family not in {ModelFamily.XGBOOST, ModelFamily.BASELINE}]
all_problems = [ProblemTypes.REGRESSION, ProblemTypes.BINARY, ProblemTypes.MULTICLASS]
all_n_points_to_explain = [1, 5]


@pytest.mark.parametrize("estimator,problem_type,n_points_to_explain",
                         product(interpretable_estimators, all_problems, all_n_points_to_explain))
def test_shap(estimator, problem_type, n_points_to_explain, X_y_binary, X_y_multi, X_y_regression,
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

    pipeline_class = make_pipeline(training_data, y, estimator, problem_type)
    pipeline = helper_functions.safe_init_pipeline_with_njobs_1(pipeline_class)
    shap_values = calculate_shap_for_test(training_data, y, pipeline, n_points_to_explain)

    if problem_type in [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]:
        assert isinstance(shap_values, list), "For binary classification, returned values must be a list"
        assert all(isinstance(class_values, dict) for class_values in shap_values), "Not all list elements are lists!"
        if is_binary:
            assert len(shap_values) == N_CLASSES_BINARY, "A dictionary should be returned for each class!"
        else:
            assert len(shap_values) == N_CLASSES_MULTICLASS, "A dictionary should be returned for each class!"
        assert all(
            len(values) == N_FEATURES for values in shap_values), "A SHAP value must be computed for every feature!"
        for class_values in shap_values:
            assert all(isinstance(feature, list) for feature in
                       class_values.values()), "Every value in the dict must be a list!"
            assert all(len(v) == n_points_to_explain for v in
                       class_values.values()), "A SHAP value must be computed for every data point to explain!"
    elif problem_type == ProblemTypes.REGRESSION:
        assert isinstance(shap_values, dict), "For regression, returned values must be a dictionary!"
        assert len(shap_values) == N_FEATURES, "A SHAP value should be computed for every feature!"
        assert all(isinstance(feature, list) for feature in shap_values.values()), "Every value in the dict must be a list!"
        assert all(len(v) == n_points_to_explain for v in
                   shap_values.values()), "A SHAP value must be computed for every data point to explain!"


def test_normalize_values_exceptions():

    with pytest.raises(ValueError, match="^Unsupported data type for _normalize_shap_values"):
        _normalize_shap_values(1)


@pytest.mark.parametrize("values,answer", [({"a": [-0.5, 0, 0.5], "b": [0.1, -0.6, 0.2]},
                                            {"a": [-0.5 / 0.6, 0, 0.5 / 0.7], "b": [0.1 / 0.6, -1.0, 0.2 / 0.7]}),
                                           ([{"a": [-0.5, 0, 0.5], "b": [0.1, -0.6, 0.2]}] * 2,
                                            [{"a": [-0.5 / 0.6, 0, 0.5 / 0.7], "b": [0.1 / 0.6, -1.0, 0.2 / 0.7]}] * 2),
                                           ({"a": [0, 0]}, {"a": [0, 0]}),
                                           ([{"a": [0]}] * 10, [{"a": [0]}] * 10),
                                           ({"a": [5], "b": [20], "c": [-22]},
                                            {"a": [5 / 47], "b": [20 / 47], "c": [-22 / 47]}),
                                           ({"a": [5], "b": [-5]}, {"a": [0.5], "b": [-0.5]}),
                                           ({0: [5], "b": [-5]}, {0: [0.5], "b": [-0.5]}),
                                           ({"a": [-0.5, 0, 0.5], 1: [0.1, -0.6, 0.2]},
                                            {"a": [-0.5 / 0.6, 0, 0.5 / 0.7], 1: [0.1 / 0.6, -1.0, 0.2 / 0.7]})
                                           ])
def test_normalize_values(values, answer):

    def check_equal_dicts(normalized, answer):
        assert set(normalized.keys()) == set(answer)
        for key in normalized:
            np.testing.assert_almost_equal(normalized[key], answer[key], decimal=4)

    normalized = _normalize_shap_values(values)
    if isinstance(normalized, dict):
        check_equal_dicts(normalized, answer)

    else:
        assert len(normalized) == len(answer)
        for values, correct in zip(normalized, answer):
            check_equal_dicts(values, correct)
