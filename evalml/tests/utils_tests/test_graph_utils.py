import warnings

import numpy as np
import pandas as pd
import pytest
from unittest.mock import   patch

from evalml import AutoMLSearch
from evalml.objectives import CostBenefitMatrix
from evalml.utils.graph_utils import (
    confusion_matrix,
    cost_benefit_thresholds,
    graph_cost_benefit_thresholds,
    normalize_confusion_matrix
)


@pytest.mark.parametrize("data_type", ['np', 'pd'])
def test_confusion_matrix(data_type):
    y_true = [2, 0, 2, 2, 0, 1]
    y_predicted = [0, 0, 2, 2, 0, 2]
    if data_type == 'pd':
        y_true = pd.Series(y_true)
        y_predicted = pd.Series(y_predicted)
    conf_mat = confusion_matrix(y_true, y_predicted, normalize_method=None)
    conf_mat_expected = np.array([[2, 0, 0], [0, 0, 1], [1, 0, 2]])
    assert np.array_equal(conf_mat_expected, conf_mat)
    assert isinstance(conf_mat, pd.DataFrame)
    conf_mat = confusion_matrix(y_true, y_predicted, normalize_method='true')
    conf_mat_expected = np.array([[1, 0, 0], [0, 0, 1], [1 / 3.0, 0, 2 / 3.0]])
    assert np.array_equal(conf_mat_expected, conf_mat)
    assert isinstance(conf_mat, pd.DataFrame)
    conf_mat = confusion_matrix(y_true, y_predicted, normalize_method='pred')
    conf_mat_expected = np.array([[2 / 3.0, np.nan, 0], [0, np.nan, 1 / 3.0], [1 / 3.0, np.nan, 2 / 3.0]])
    assert np.allclose(conf_mat_expected, conf_mat, equal_nan=True)
    assert isinstance(conf_mat, pd.DataFrame)
    conf_mat = confusion_matrix(y_true, y_predicted, normalize_method='all')
    conf_mat_expected = np.array([[1 / 3.0, 0, 0], [0, 0, 1 / 6.0], [1 / 6.0, 0, 1 / 3.0]])
    assert np.array_equal(conf_mat_expected, conf_mat)
    assert isinstance(conf_mat, pd.DataFrame)
    with pytest.raises(ValueError, match='Invalid value provided'):
        conf_mat = confusion_matrix(y_true, y_predicted, normalize_method='Invalid Option')


@pytest.mark.parametrize("data_type", ['np', 'pd'])
def test_normalize_confusion_matrix(data_type):
    conf_mat = np.array([[2, 3, 0], [0, 1, 1], [1, 0, 2]])
    if data_type == 'pd':
        conf_mat = pd.DataFrame(conf_mat)
    conf_mat_normalized = normalize_confusion_matrix(conf_mat)
    assert all(conf_mat_normalized.sum(axis=1) == 1.0)
    assert isinstance(conf_mat_normalized, type(conf_mat))

    conf_mat_normalized = normalize_confusion_matrix(conf_mat, 'pred')
    for col_sum in conf_mat_normalized.sum(axis=0):
        assert col_sum == 1.0 or col_sum == 0.0

    conf_mat_normalized = normalize_confusion_matrix(conf_mat, 'all')
    assert conf_mat_normalized.sum().sum() == 1.0

    # testing with pd.DataFrames
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


def test_normalize_confusion_matrix_error():
    conf_mat = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    warnings.simplefilter('default', category=RuntimeWarning)

    with pytest.raises(ValueError, match='Invalid value provided'):
        normalize_confusion_matrix(conf_mat, normalize_method='invalid option')
    with pytest.raises(ValueError, match='Invalid value provided'):
        normalize_confusion_matrix(conf_mat, normalize_method=None)

    with pytest.raises(ValueError, match="Sum of given axis is 0"):
        normalize_confusion_matrix(conf_mat, 'true')
    with pytest.raises(ValueError, match="Sum of given axis is 0"):
        normalize_confusion_matrix(conf_mat, 'pred')
    with pytest.raises(ValueError, match="Sum of given axis is 0"):
        normalize_confusion_matrix(conf_mat, 'all')


def test_cost_benefit_thresholds(X_y_binary, logistic_regression_binary_pipeline_class):
    X, y = X_y_binary
    cbm = CostBenefitMatrix(true_positive_cost=1, true_negative_cost=-1,
                            false_positive_cost=-7, false_negative_cost=-2)
    pipeline = logistic_regression_binary_pipeline_class(parameters={})
    pipeline.fit(X, y)
    cost_benefit_df = cost_benefit_thresholds(pipeline, X, y, cbm)
    assert list(cost_benefit_df.columns) == ['thresholds', 'costs']
    assert cost_benefit_df.shape == (101, 2)


@patch('evalml.pipelines.BinaryClassificationPipeline.predict_proba')
@patch('evalml.objectives.CostBenefitMatrix.decision_function')
@patch('evalml.objectives.CostBenefitMatrix.objective_function')
def test_cost_benefit_thresholds_steps(mock_obj_function, mock_decision_function, mock_predict_proba,
                                       X_y_binary, logistic_regression_binary_pipeline_class):
    X, y = X_y_binary
    cbm = CostBenefitMatrix(true_positive_cost=1, true_negative_cost=-1,
                            false_positive_cost=-7, false_negative_cost=-2)
    pipeline = logistic_regression_binary_pipeline_class(parameters={})
    mock_obj_function.return_value = 0.2
    cost_benefit_df = cost_benefit_thresholds(pipeline, X, y, cbm, steps=234)
    mock_predict_proba.assert_called()
    mock_decision_function.assert_called()
    assert list(cost_benefit_df.columns) == ['thresholds', 'costs']
    assert cost_benefit_df.shape == (235, 2)


@patch('evalml.utils.graph_utils.cost_benefit_thresholds')
def test_graph_cost_benefit_thresholds(mock_cb_thresholds, X_y_binary, logistic_regression_binary_pipeline_class):
    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    X, y = X_y_binary
    pipeline = logistic_regression_binary_pipeline_class(parameters={})
    cbm = CostBenefitMatrix(true_positive_cost=1, true_negative_cost=-1,
                            false_positive_cost=-7, false_negative_cost=-2)

    mock_cb_thresholds.return_value = pd.DataFrame({'thresholds': [0, 0.5, 1.0],
    'costs': [100, -20, 5]})

    figure = graph_cost_benefit_thresholds(pipeline, X, y, cbm)
    assert isinstance(figure, go.Figure)
    data = figure.data[0]
    assert not np.any(np.isnan(data['x']))
    assert not np.any(np.isnan(data['y']))
