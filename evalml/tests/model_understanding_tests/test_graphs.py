import os
import warnings
from collections import OrderedDict
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import woodwork as ww
from sklearn.exceptions import NotFittedError, UndefinedMetricWarning
from sklearn.preprocessing import label_binarize
from skopt.space import Real

from evalml.model_understanding.graphs import (
    binary_objective_vs_threshold,
    calculate_permutation_importance,
    confusion_matrix,
    decision_tree_data_from_estimator,
    decision_tree_data_from_pipeline,
    get_linear_coefficients,
    get_prediction_vs_actual_data,
    get_prediction_vs_actual_over_time_data,
    graph_binary_objective_vs_threshold,
    graph_confusion_matrix,
    graph_partial_dependence,
    graph_permutation_importance,
    graph_precision_recall_curve,
    graph_prediction_vs_actual,
    graph_prediction_vs_actual_over_time,
    graph_roc_curve,
    graph_t_sne,
    normalize_confusion_matrix,
    precision_recall_curve,
    roc_curve,
    t_sne,
    visualize_decision_tree
)
from evalml.objectives import CostBenefitMatrix
from evalml.pipelines import (
    BinaryClassificationPipeline,
    DecisionTreeRegressor,
    ElasticNetRegressor,
    LinearRegressor,
    MulticlassClassificationPipeline,
    RegressionPipeline
)
from evalml.problem_types import ProblemTypes
from evalml.utils import (
    _convert_woodwork_types_wrapper,
    get_random_state,
    infer_feature_types
)


@pytest.fixture
def test_pipeline():
    class TestPipeline(BinaryClassificationPipeline):
        component_graph = ['Simple Imputer', 'One Hot Encoder', 'Standard Scaler', 'Logistic Regression Classifier']

        hyperparameters = {
            "penalty": ["l2"],
            "C": Real(.01, 10),
            "impute_strategy": ["mean", "median", "most_frequent"],
        }

        def __init__(self, parameters):
            super().__init__(parameters=parameters)

    return TestPipeline(parameters={"Logistic Regression Classifier": {"n_jobs": 1}})


@pytest.mark.parametrize("data_type", ['np', 'pd', 'ww'])
def test_confusion_matrix(data_type, make_data_type):
    y_true = np.array([2, 0, 2, 2, 0, 1, 1, 0, 2])
    y_predicted = np.array([0, 0, 2, 2, 0, 2, 1, 1, 1])
    y_true = make_data_type(data_type, y_true)
    y_predicted = make_data_type(data_type, y_predicted)

    conf_mat = confusion_matrix(y_true, y_predicted, normalize_method=None)
    conf_mat_expected = np.array([[2, 1, 0], [0, 1, 1], [1, 1, 2]])
    assert np.array_equal(conf_mat_expected, conf_mat.to_numpy())
    assert isinstance(conf_mat, pd.DataFrame)

    conf_mat = confusion_matrix(y_true, y_predicted, normalize_method='all')
    conf_mat_expected = conf_mat_expected / 9.0
    assert np.array_equal(conf_mat_expected, conf_mat.to_numpy())
    assert isinstance(conf_mat, pd.DataFrame)

    conf_mat = confusion_matrix(y_true, y_predicted, normalize_method='true')
    conf_mat_expected = np.array([[2 / 3.0, 1 / 3.0, 0], [0, 0.5, 0.5], [0.25, 0.25, 0.5]])
    assert np.array_equal(conf_mat_expected, conf_mat.to_numpy())
    assert isinstance(conf_mat, pd.DataFrame)

    conf_mat = confusion_matrix(y_true, y_predicted, normalize_method='pred')
    conf_mat_expected = np.array([[2 / 3.0, 1 / 3.0, 0], [0, 1 / 3.0, 1 / 3.0], [1 / 3.0, 1 / 3.0, 2 / 3.0]])
    assert np.allclose(conf_mat_expected, conf_mat.to_numpy(), equal_nan=True)
    assert isinstance(conf_mat, pd.DataFrame)

    with pytest.raises(ValueError, match='Invalid value provided'):
        conf_mat = confusion_matrix(y_true, y_predicted, normalize_method='Invalid Option')


@pytest.mark.parametrize("data_type", ['ww', 'np', 'pd'])
def test_normalize_confusion_matrix(data_type, make_data_type):
    conf_mat = np.array([[2, 3, 0], [0, 1, 1], [1, 0, 2]])
    conf_mat = make_data_type(data_type, conf_mat)

    conf_mat_normalized = normalize_confusion_matrix(conf_mat)
    assert all(conf_mat_normalized.sum(axis=1) == 1.0)
    assert isinstance(conf_mat_normalized, pd.DataFrame)

    conf_mat_normalized = normalize_confusion_matrix(conf_mat, 'pred')
    for col_sum in conf_mat_normalized.sum(axis=0):
        assert col_sum == 1.0 or col_sum == 0.0

    conf_mat_normalized = normalize_confusion_matrix(conf_mat, 'all')
    assert conf_mat_normalized.sum().sum() == 1.0

    # testing with named pd.DataFrames
    conf_mat_df = pd.DataFrame()
    conf_mat_df["col_1"] = [0, 1, 2]
    conf_mat_df["col_2"] = [0, 0, 3]
    conf_mat_df["col_3"] = [2, 0, 0]
    conf_mat_normalized = normalize_confusion_matrix(conf_mat_df)
    assert all(conf_mat_normalized.sum(axis=1) == 1.0)
    assert list(conf_mat_normalized.columns) == ['col_1', 'col_2', 'col_3']

    conf_mat_normalized = normalize_confusion_matrix(conf_mat_df, 'pred')
    for col_sum in conf_mat_normalized.sum(axis=0):
        assert col_sum == 1.0 or col_sum == 0.0

    conf_mat_normalized = normalize_confusion_matrix(conf_mat_df, 'all')
    assert conf_mat_normalized.sum().sum() == 1.0


@pytest.mark.parametrize("data_type", ['ww', 'np', 'pd'])
def test_normalize_confusion_matrix_error(data_type, make_data_type):
    conf_mat = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    conf_mat = make_data_type(data_type, conf_mat)

    warnings.simplefilter('default', category=RuntimeWarning)

    with pytest.raises(ValueError, match='Invalid value provided for "normalize_method": invalid option'):
        normalize_confusion_matrix(conf_mat, normalize_method='invalid option')
    with pytest.raises(ValueError, match='Invalid value provided'):
        normalize_confusion_matrix(conf_mat, normalize_method=None)

    with pytest.raises(ValueError, match="Sum of given axis is 0"):
        normalize_confusion_matrix(conf_mat, 'true')
    with pytest.raises(ValueError, match="Sum of given axis is 0"):
        normalize_confusion_matrix(conf_mat, 'pred')
    with pytest.raises(ValueError, match="Sum of given axis is 0"):
        normalize_confusion_matrix(conf_mat, 'all')
