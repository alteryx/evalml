import warnings
from itertools import product
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from evalml.model_family.model_family import ModelFamily
from evalml.model_understanding.prediction_explanations._algorithms import (
    _aggregate_explainer_values,
    _compute_lime_values,
    _compute_shap_values,
    _create_dictionary,
    _normalize_explainer_values,
)
from evalml.pipelines import (
    BinaryClassificationPipeline,
    MulticlassClassificationPipeline,
    RegressionPipeline,
    TimeSeriesRegressionPipeline,
)
from evalml.pipelines.components import (
    BaselineClassifier,
    BaselineRegressor,
    LinearRegressor,
    RandomForestClassifier,
    TimeSeriesBaselineEstimator,
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
        custom_name = estimator.name

        def __init__(self, parameters, random_seed=0):
            super().__init__(
                self.component_graph,
                parameters=parameters,
                custom_name=self.custom_name,
                random_seed=random_seed,
            )

    return Pipeline


baseline_message = "You passed in a baseline pipeline. These are simple enough that SHAP values are not needed."
baseline_lime_message = "You passed in a baseline pipeline. These are simple enough that LIME values are not needed."
datatype_message = "^Unknown shap_values datatype"
data_message = "You must pass in a value for parameter 'training_data' when the pipeline does not have a tree-based estimator. Current estimator model family is Linear."


@pytest.mark.parametrize(
    "pipeline,exception,match,algorithm",
    [
        (
            make_test_pipeline(BaselineRegressor, RegressionPipeline),
            ValueError,
            baseline_message,
            "shap",
        ),
        (
            make_test_pipeline(BaselineClassifier, BinaryClassificationPipeline),
            ValueError,
            baseline_message,
            "shap",
        ),
        (
            make_test_pipeline(BaselineClassifier, MulticlassClassificationPipeline),
            ValueError,
            baseline_message,
            "shap",
        ),
        (
            make_test_pipeline(
                TimeSeriesBaselineEstimator,
                TimeSeriesRegressionPipeline,
            ),
            ValueError,
            baseline_message,
            "shap",
        ),
        (
            make_test_pipeline(RandomForestClassifier, BinaryClassificationPipeline),
            ValueError,
            datatype_message,
            "shap",
        ),
        (
            make_test_pipeline(LinearRegressor, RegressionPipeline),
            ValueError,
            data_message,
            "shap",
        ),
        (
            make_test_pipeline(BaselineRegressor, RegressionPipeline),
            ValueError,
            baseline_lime_message,
            "lime",
        ),
        (
            make_test_pipeline(BaselineClassifier, BinaryClassificationPipeline),
            ValueError,
            baseline_lime_message,
            "lime",
        ),
        (
            make_test_pipeline(BaselineClassifier, MulticlassClassificationPipeline),
            ValueError,
            baseline_lime_message,
            "lime",
        ),
        (
            make_test_pipeline(
                TimeSeriesBaselineEstimator,
                TimeSeriesRegressionPipeline,
            ),
            ValueError,
            baseline_lime_message,
            "lime",
        ),
    ],
)
@patch(
    "evalml.model_understanding.prediction_explanations._algorithms.shap.TreeExplainer",
)
def test_explainer_value_errors_raised(
    mock_tree_explainer,
    pipeline,
    exception,
    match,
    algorithm,
):
    pipeline = pipeline(
        {
            "pipeline": {
                "time_index": "foo",
                "gap": 1,
                "max_delay": 1,
                "forecast_horizon": 1,
            },
        },
    )
    with pytest.raises(exception, match=match):
        if algorithm == "shap":
            _ = _compute_shap_values(pipeline, pd.DataFrame(np.random.random((2, 16))))
        else:
            _ = _compute_lime_values(
                pipeline,
                pd.DataFrame(np.random.random((2, 16))),
                0,
            )


def test_create_dictionary_exception():
    with pytest.raises(
        ValueError,
        match="Explainer values must be stored in a numpy array!",
    ):
        _create_dictionary([1, 2, 3], ["a", "b", "c"])


N_CLASSES_BINARY = 2
N_CLASSES_MULTICLASS = 3
N_FEATURES = 20


def calculate_shap_for_test(training_data, y, pipeline, n_points_to_explain):
    """Helper function to compute the SHAP values for n_points_to_explain for a given pipeline."""
    points_to_explain = training_data[:n_points_to_explain]
    pipeline.fit(training_data, y)
    shap_values, expected_value = _compute_shap_values(
        pipeline,
        pd.DataFrame(points_to_explain),
        training_data,
    )
    return shap_values


def calculate_lime_for_test(training_data, y, pipeline, index_to_explain):
    """Helper function to compute the LIME values for n_points_to_explain for a given pipeline."""
    pipeline.fit(training_data, y)
    lime_values = _compute_lime_values(
        pipeline,
        training_data,
        index_to_explain,
    )
    return lime_values


interpretable_estimators = [
    e
    for e in _all_estimators_used_in_search()
    if e.model_family != ModelFamily.BASELINE
]
all_problems = [ProblemTypes.REGRESSION, ProblemTypes.BINARY, ProblemTypes.MULTICLASS]
all_n_points_to_explain = [1, 5]
algorithms = ["shap", "lime"]


@pytest.mark.parametrize(
    "estimator,problem_type,n_points_to_explain,algorithm",
    product(
        interpretable_estimators,
        all_problems,
        all_n_points_to_explain,
        algorithms,
    ),
)
def test_explainers(
    estimator,
    problem_type,
    n_points_to_explain,
    algorithm,
    X_y_binary,
    X_y_multi,
    X_y_regression,
    helper_functions,
):
    if problem_type not in estimator.supported_problem_types:
        pytest.skip("Skipping because estimator and pipeline are not compatible.")

    if problem_type == ProblemTypes.BINARY:
        training_data, y = X_y_binary
        is_binary = True
    elif problem_type == ProblemTypes.MULTICLASS:
        training_data, y = X_y_multi
        is_binary = False
    else:
        training_data, y = X_y_regression

    parameters = {estimator.name: {"n_jobs": 1}}
    try:
        pipeline = make_pipeline(
            training_data,
            y,
            estimator,
            problem_type,
            parameters=parameters,
        )
    except ValueError:
        pipeline = make_pipeline(training_data, y, estimator, problem_type)

    if algorithm == "shap":
        explainer_values = calculate_shap_for_test(
            training_data,
            y,
            pipeline,
            n_points_to_explain,
        )
    else:
        explainer_values = calculate_lime_for_test(training_data, y, pipeline, 0)

    if problem_type in [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]:
        assert isinstance(
            explainer_values,
            list,
        ), "For binary classification, returned values must be a list"
        assert all(
            isinstance(class_values, dict) for class_values in explainer_values
        ), "Not all list elements are lists!"
        if is_binary:
            assert (
                len(explainer_values) == N_CLASSES_BINARY
            ), "A dictionary should be returned for each class!"
        else:
            assert (
                len(explainer_values) == N_CLASSES_MULTICLASS
            ), "A dictionary should be returned for each class!"
        assert all(
            len(values) == N_FEATURES for values in explainer_values
        ), f"A {algorithm.upper()} value must be computed for every feature!"
        for class_values in explainer_values:
            assert all(
                isinstance(feature, list) for feature in class_values.values()
            ), "Every value in the dict must be a list!"
            if algorithm == "shap":
                assert all(
                    len(v) == n_points_to_explain for v in class_values.values()
                ), f"A SHAP value must be computed for every data point to explain!"
    elif problem_type == ProblemTypes.REGRESSION:
        assert isinstance(
            explainer_values,
            dict,
        ), "For regression, returned values must be a dictionary!"
        assert (
            len(explainer_values) == N_FEATURES
        ), f"A {algorithm.upper()} value should be computed for every feature!"
        assert all(
            isinstance(feature, list) for feature in explainer_values.values()
        ), "Every value in the dict must be a list!"
        if algorithm == "shap":
            assert all(
                len(v) == n_points_to_explain for v in explainer_values.values()
            ), f"A SHAP value must be computed for every data point to explain!"


@pytest.mark.parametrize("data_type", ["ww", "np"])
def test_lime_xgboost(data_type, X_y_multi):

    from evalml.pipelines.components import XGBoostClassifier

    X, y = X_y_multi
    if data_type == "np":
        X, y = X.values, y.values

    xgboost_pipeline = make_pipeline(X, y, XGBoostClassifier, "multiclass")
    xgboost_values = calculate_lime_for_test(X, y, xgboost_pipeline, 1)

    assert len(xgboost_values) == len(set(y))


@patch("evalml.model_understanding.prediction_explanations._algorithms.logger")
@patch("shap.TreeExplainer")
def test_compute_shap_values_catches_shap_tree_warnings(
    mock_tree_explainer,
    mock_debug,
    X_y_binary,
    caplog,
):
    X, y = X_y_binary
    pipeline = BinaryClassificationPipeline(["Random Forest Classifier"])

    def raise_warning_from_shap(estimator, feature_perturbation):
        warnings.warn("Shap raised a warning!")
        mock = MagicMock()
        mock.shap_values.return_value = np.zeros(10)
        return mock

    mock_tree_explainer.side_effect = raise_warning_from_shap

    _ = _compute_shap_values(pipeline, pd.DataFrame(X))
    mock_debug.debug.assert_called_with(
        "_compute_shap_values TreeExplainer: Shap raised a warning!",
    )


@patch("sklearn.linear_model.LogisticRegression.predict_proba")
def test_compute_shap_values_absolute_probs(mock_predict_proba, X_y_binary):
    X, y = X_y_binary
    pipeline = BinaryClassificationPipeline(["Logistic Regression Classifier"])
    mock_predict_proba.side_effect = [
        pd.DataFrame({0: [0.0] + [0.4] * 99, 1: [1.0] + [0.6] * 99}),
        pd.DataFrame({0: [0.0] + [0.4] * 99, 1: [1.0] + [0.6] * 99}),
        pd.DataFrame({0: [0.0] + [0.4] * 208799, 1: [1.0] + [0.6] * 208799}),
    ]

    shap_values = calculate_shap_for_test(X, y, pipeline, 1)
    mock_predict_proba.assert_called()

    for feat in shap_values[0].values():
        assert not np.isnan(feat).any()


def test_normalize_values_exceptions():

    with pytest.raises(
        ValueError,
        match="^Unsupported data type for _normalize_explainer_values",
    ):
        _normalize_explainer_values(1)


def check_equal_dicts(normalized, answer):
    assert set(normalized.keys()) == set(answer)
    for key in normalized:
        np.testing.assert_almost_equal(normalized[key], answer[key], decimal=4)


@pytest.mark.parametrize(
    "values,answer",
    [
        (
            {"a": [-0.5, 0, 0.5], "b": [0.1, -0.6, 0.2]},
            {"a": [-0.5 / 0.6, 0, 0.5 / 0.7], "b": [0.1 / 0.6, -1.0, 0.2 / 0.7]},
        ),
        (
            [{"a": [-0.5, 0, 0.5], "b": [0.1, -0.6, 0.2]}] * 2,
            [{"a": [-0.5 / 0.6, 0, 0.5 / 0.7], "b": [0.1 / 0.6, -1.0, 0.2 / 0.7]}] * 2,
        ),
        ({"a": [0, 0]}, {"a": [0, 0]}),
        ([{"a": [0]}] * 10, [{"a": [0]}] * 10),
        (
            {"a": [5], "b": [20], "c": [-22]},
            {"a": [5 / 47], "b": [20 / 47], "c": [-22 / 47]},
        ),
        ({"a": [5], "b": [-5]}, {"a": [0.5], "b": [-0.5]}),
        ({0: [5], "b": [-5]}, {0: [0.5], "b": [-0.5]}),
        (
            {"a": [-0.5, 0, 0.5], 1: [0.1, -0.6, 0.2]},
            {"a": [-0.5 / 0.6, 0, 0.5 / 0.7], 1: [0.1 / 0.6, -1.0, 0.2 / 0.7]},
        ),
    ],
)
def test_normalize_values(values, answer):

    normalized = _normalize_explainer_values(values)
    if isinstance(normalized, dict):
        check_equal_dicts(normalized, answer)

    else:
        assert len(normalized) == len(answer)
        for values, correct in zip(normalized, answer):
            check_equal_dicts(values, correct)


@pytest.mark.parametrize(
    "values,provenance,answer",
    [
        (
            {"a_0": [-0.5, 0, 0.5], "a_1": [1, 1, 2], "b": [0.1, -0.6, 0.2]},
            {"a": ["a_0", "a_1"]},
            {"a": [0.5, 1, 2.5], "b": [0.1, -0.6, 0.2]},
        ),
        (
            [
                {"a_0": [0.5, 1.0, 2.0], "a_1": [1.2, 1.5, 0.6], "b": [0.5, 0.2, 0.5]},
                {"a_0": [-0.5, 0, 0.5], "a_1": [1, 1, 2], "b": [0.1, -0.6, 0.2]},
            ],
            {"a": ["a_0", "a_1"], "c": ["c_1", "c_2"]},
            [
                {"a": [1.7, 2.5, 2.6], "b": [0.5, 0.2, 0.5]},
                {"a": [0.5, 1, 2.5], "b": [0.1, -0.6, 0.2]},
            ],
        ),
    ],
)
def test_aggregate_values(values, provenance, answer):
    aggregated = _aggregate_explainer_values(values, provenance)

    if isinstance(aggregated, dict):
        check_equal_dicts(aggregated, answer)
    else:
        assert len(aggregated) == len(answer)
        for values, correct in zip(aggregated, answer):
            check_equal_dicts(values, correct)
