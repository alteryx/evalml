import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import label_binarize
from skopt.space import Real

from evalml.demos import load_breast_cancer
from evalml.model_understanding.graphs import (
    binary_objective_vs_threshold,
    calculate_permutation_importance,
    confusion_matrix,
    graph_binary_objective_vs_threshold,
    graph_confusion_matrix,
    graph_partial_dependence,
    graph_permutation_importance,
    graph_precision_recall_curve,
    graph_roc_curve,
    normalize_confusion_matrix,
    partial_dependence,
    precision_recall_curve,
    roc_curve
)
from evalml.objectives import CostBenefitMatrix, get_objectives
from evalml.pipelines import BinaryClassificationPipeline
from evalml.problem_types import ProblemTypes


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

    return TestPipeline(parameters={})


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


@pytest.fixture
def binarized_ys(X_y_multi):
    _, y_true = X_y_multi
    rs = np.random.RandomState(42)
    y_tr = label_binarize(y_true, classes=[0, 1, 2])
    y_pred_proba = y_tr * rs.random(y_tr.shape)
    return y_true, y_tr, y_pred_proba


def test_precision_recall_curve_return_type():
    y_true = np.array([0, 0, 1, 1])
    y_predict_proba = np.array([0.1, 0.4, 0.35, 0.8])
    precision_recall_curve_data = precision_recall_curve(y_true, y_predict_proba)
    assert isinstance(precision_recall_curve_data['precision'], np.ndarray)
    assert isinstance(precision_recall_curve_data['recall'], np.ndarray)
    assert isinstance(precision_recall_curve_data['thresholds'], np.ndarray)
    assert isinstance(precision_recall_curve_data['auc_score'], float)


def test_precision_recall_curve():
    y_true = np.array([0, 0, 1, 1])
    y_predict_proba = np.array([0.1, 0.4, 0.35, 0.8])
    precision_recall_curve_data = precision_recall_curve(y_true, y_predict_proba)

    precision = precision_recall_curve_data.get('precision')
    recall = precision_recall_curve_data.get('recall')
    thresholds = precision_recall_curve_data.get('thresholds')

    precision_expected = np.array([0.66666667, 0.5, 1, 1])
    recall_expected = np.array([1, 0.5, 0.5, 0])
    thresholds_expected = np.array([0.35, 0.4, 0.8])

    np.testing.assert_almost_equal(precision_expected, precision, decimal=5)
    np.testing.assert_almost_equal(recall_expected, recall, decimal=5)
    np.testing.assert_almost_equal(thresholds_expected, thresholds, decimal=5)


@pytest.mark.parametrize("data_type", ['np', 'pd'])
def test_graph_precision_recall_curve(X_y_binary, data_type):
    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    X, y_true = X_y_binary
    if data_type == 'pd':
        X = pd.DataFrame(X)
        y_true = pd.Series(y_true)
    rs = np.random.RandomState(42)
    y_pred_proba = y_true * rs.random(y_true.shape)
    fig = graph_precision_recall_curve(y_true, y_pred_proba)
    assert isinstance(fig, type(go.Figure()))

    fig_dict = fig.to_dict()
    assert fig_dict['layout']['title']['text'] == 'Precision-Recall'
    assert len(fig_dict['data']) == 1

    precision_recall_curve_data = precision_recall_curve(y_true, y_pred_proba)
    assert np.array_equal(fig_dict['data'][0]['x'], precision_recall_curve_data['recall'])
    assert np.array_equal(fig_dict['data'][0]['y'], precision_recall_curve_data['precision'])
    assert fig_dict['data'][0]['name'] == 'Precision-Recall (AUC {:06f})'.format(precision_recall_curve_data['auc_score'])


def test_graph_precision_recall_curve_title_addition(X_y_binary):
    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    X, y_true = X_y_binary
    rs = np.random.RandomState(42)
    y_pred_proba = y_true * rs.random(y_true.shape)
    fig = graph_precision_recall_curve(y_true, y_pred_proba, title_addition='with added title text')
    assert isinstance(fig, type(go.Figure()))
    fig_dict = fig.to_dict()
    assert fig_dict['layout']['title']['text'] == 'Precision-Recall with added title text'


@pytest.mark.parametrize("data_type", ['np', 'pd'])
def test_roc_curve_binary(data_type):
    y_true = np.array([1, 1, 0, 0])
    y_predict_proba = np.array([0.1, 0.4, 0.35, 0.8])
    if data_type == 'pd':
        y_true = pd.Series(y_true)
        y_predict_proba = pd.DataFrame(y_predict_proba)
    roc_curve_data = roc_curve(y_true, y_predict_proba)
    fpr_rates = roc_curve_data.get('fpr_rates')
    tpr_rates = roc_curve_data.get('tpr_rates')
    thresholds = roc_curve_data.get('thresholds')
    auc_score = roc_curve_data.get('auc_score')
    fpr_expected = np.array([0, 0.5, 0.5, 1, 1])
    tpr_expected = np.array([0, 0, 0.5, 0.5, 1])
    thresholds_expected = np.array([1.8, 0.8, 0.4, 0.35, 0.1])
    assert np.array_equal(fpr_expected, fpr_rates)
    assert np.array_equal(tpr_expected, tpr_rates)
    assert np.array_equal(thresholds_expected, thresholds)
    assert auc_score == pytest.approx(0.25, 1e-5)
    assert isinstance(roc_curve_data['fpr_rates'], np.ndarray)
    assert isinstance(roc_curve_data['tpr_rates'], np.ndarray)
    assert isinstance(roc_curve_data['thresholds'], np.ndarray)


@pytest.mark.parametrize("data_type", ['np', 'pd'])
def test_graph_roc_curve_binary(X_y_binary, data_type):
    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    X, y_true = X_y_binary
    rs = np.random.RandomState(42)
    y_pred_proba = y_true * rs.random(y_true.shape)
    if data_type == 'pd':
        y_true = pd.Series(y_true)
        y_pred_proba = pd.DataFrame(y_pred_proba)
    fig = graph_roc_curve(y_true, y_pred_proba)
    assert isinstance(fig, type(go.Figure()))
    fig_dict = fig.to_dict()
    assert fig_dict['layout']['title']['text'] == 'Receiver Operating Characteristic'
    assert len(fig_dict['data']) == 2
    roc_curve_data = roc_curve(y_true, y_pred_proba)
    assert np.array_equal(fig_dict['data'][0]['x'], roc_curve_data['fpr_rates'])
    assert np.array_equal(fig_dict['data'][0]['y'], roc_curve_data['tpr_rates'])
    assert fig_dict['data'][0]['name'] == 'Class 1 (AUC {:06f})'.format(roc_curve_data['auc_score'])
    assert np.array_equal(fig_dict['data'][1]['x'], np.array([0, 1]))
    assert np.array_equal(fig_dict['data'][1]['y'], np.array([0, 1]))
    assert fig_dict['data'][1]['name'] == 'Trivial Model (AUC 0.5)'


def test_graph_roc_curve_nans():
    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    one_val_y_zero = np.array([0])
    with pytest.warns(UndefinedMetricWarning):
        fig = graph_roc_curve(one_val_y_zero, one_val_y_zero)
    assert isinstance(fig, type(go.Figure()))
    fig_dict = fig.to_dict()
    assert np.array_equal(fig_dict['data'][0]['x'], np.array([0., 1.]))
    assert np.allclose(fig_dict['data'][0]['y'], np.array([np.nan, np.nan]), equal_nan=True)
    fig1 = graph_roc_curve(np.array([np.nan, 1, 1, 0, 1]), np.array([0, 0, 0.5, 0.1, 0.9]))
    fig2 = graph_roc_curve(np.array([1, 0, 1, 0, 1]), np.array([0, np.nan, 0.5, 0.1, 0.9]))
    assert fig1 == fig2


def test_graph_roc_curve_multiclass(binarized_ys):
    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    y_true, y_tr, y_pred_proba = binarized_ys
    fig = graph_roc_curve(y_true, y_pred_proba)
    assert isinstance(fig, type(go.Figure()))
    fig_dict = fig.to_dict()
    assert fig_dict['layout']['title']['text'] == 'Receiver Operating Characteristic'
    assert len(fig_dict['data']) == 4
    for i in range(3):
        roc_curve_data = roc_curve(y_tr[:, i], y_pred_proba[:, i])
        assert np.array_equal(fig_dict['data'][i]['x'], roc_curve_data['fpr_rates'])
        assert np.array_equal(fig_dict['data'][i]['y'], roc_curve_data['tpr_rates'])
        assert fig_dict['data'][i]['name'] == 'Class {name} (AUC {:06f})'.format(roc_curve_data['auc_score'], name=i + 1)
    assert np.array_equal(fig_dict['data'][3]['x'], np.array([0, 1]))
    assert np.array_equal(fig_dict['data'][3]['y'], np.array([0, 1]))
    assert fig_dict['data'][3]['name'] == 'Trivial Model (AUC 0.5)'

    with pytest.raises(ValueError, match='Number of custom class names does not match number of classes'):
        graph_roc_curve(y_true, y_pred_proba, custom_class_names=['one', 'two'])


def test_graph_roc_curve_multiclass_custom_class_names(binarized_ys):
    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    y_true, y_tr, y_pred_proba = binarized_ys
    custom_class_names = ['one', 'two', 'three']
    fig = graph_roc_curve(y_true, y_pred_proba, custom_class_names=custom_class_names)
    assert isinstance(fig, type(go.Figure()))
    fig_dict = fig.to_dict()
    assert fig_dict['layout']['title']['text'] == 'Receiver Operating Characteristic'
    for i in range(3):
        roc_curve_data = roc_curve(y_tr[:, i], y_pred_proba[:, i])
        assert np.array_equal(fig_dict['data'][i]['x'], roc_curve_data['fpr_rates'])
        assert np.array_equal(fig_dict['data'][i]['y'], roc_curve_data['tpr_rates'])
        assert fig_dict['data'][i]['name'] == 'Class {name} (AUC {:06f})'.format(roc_curve_data['auc_score'], name=custom_class_names[i])
    assert np.array_equal(fig_dict['data'][3]['x'], np.array([0, 1]))
    assert np.array_equal(fig_dict['data'][3]['y'], np.array([0, 1]))
    assert fig_dict['data'][3]['name'] == 'Trivial Model (AUC 0.5)'


def test_graph_roc_curve_title_addition(X_y_binary):
    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    X, y_true = X_y_binary
    rs = np.random.RandomState(42)
    y_pred_proba = y_true * rs.random(y_true.shape)
    fig = graph_roc_curve(y_true, y_pred_proba, title_addition='with added title text')
    assert isinstance(fig, type(go.Figure()))
    fig_dict = fig.to_dict()
    assert fig_dict['layout']['title']['text'] == 'Receiver Operating Characteristic with added title text'


@pytest.mark.parametrize("data_type", ['np', 'pd'])
def test_graph_confusion_matrix_default(X_y_binary, data_type):
    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    X, y_true = X_y_binary
    rs = np.random.RandomState(42)
    y_pred = np.round(y_true * rs.random(y_true.shape)).astype(int)
    if data_type == 'pd':
        y_true = pd.Series(y_true)
        y_pred = pd.Series(y_pred)
    fig = graph_confusion_matrix(y_true, y_pred)
    assert isinstance(fig, type(go.Figure()))
    fig_dict = fig.to_dict()
    assert fig_dict['layout']['title']['text'] == 'Confusion matrix, normalized using method "true"'
    assert fig_dict['layout']['xaxis']['title']['text'] == 'Predicted Label'
    assert np.all(fig_dict['layout']['xaxis']['tickvals'] == np.array([0, 1]))
    assert fig_dict['layout']['yaxis']['title']['text'] == 'True Label'
    assert np.all(fig_dict['layout']['yaxis']['tickvals'] == np.array([0, 1]))
    assert fig_dict['layout']['yaxis']['autorange'] == 'reversed'
    heatmap = fig_dict['data'][0]
    conf_mat = confusion_matrix(y_true, y_pred, normalize_method='true')
    conf_mat_unnormalized = confusion_matrix(y_true, y_pred, normalize_method=None)
    assert np.array_equal(heatmap['x'], conf_mat.columns)
    assert np.array_equal(heatmap['y'], conf_mat.columns)
    assert np.array_equal(heatmap['z'], conf_mat)
    assert np.array_equal(heatmap['customdata'], conf_mat_unnormalized)
    assert heatmap['hovertemplate'] == '<b>True</b>: %{y}<br><b>Predicted</b>: %{x}<br><b>Normalized Count</b>: %{z}<br><b>Raw Count</b>: %{customdata} <br><extra></extra>'


def test_graph_confusion_matrix_norm_disabled(X_y_binary):
    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    X, y_true = X_y_binary
    rs = np.random.RandomState(42)
    y_pred = np.round(y_true * rs.random(y_true.shape)).astype(int)
    fig = graph_confusion_matrix(y_true, y_pred, normalize_method=None)
    assert isinstance(fig, type(go.Figure()))
    fig_dict = fig.to_dict()
    assert fig_dict['layout']['title']['text'] == 'Confusion matrix'
    assert fig_dict['layout']['xaxis']['title']['text'] == 'Predicted Label'
    assert np.all(fig_dict['layout']['xaxis']['tickvals'] == np.array([0, 1]))
    assert fig_dict['layout']['yaxis']['title']['text'] == 'True Label'
    assert np.all(fig_dict['layout']['yaxis']['tickvals'] == np.array([0, 1]))
    assert fig_dict['layout']['yaxis']['autorange'] == 'reversed'
    heatmap = fig_dict['data'][0]
    conf_mat = confusion_matrix(y_true, y_pred, normalize_method=None)
    conf_mat_normalized = confusion_matrix(y_true, y_pred, normalize_method='true')
    assert np.array_equal(heatmap['x'], conf_mat.columns)
    assert np.array_equal(heatmap['y'], conf_mat.columns)
    assert np.array_equal(heatmap['z'], conf_mat)
    assert np.array_equal(heatmap['customdata'], conf_mat_normalized)
    assert heatmap['hovertemplate'] == '<b>True</b>: %{y}<br><b>Predicted</b>: %{x}<br><b>Raw Count</b>: %{z}<br><b>Normalized Count</b>: %{customdata} <br><extra></extra>'


def test_graph_confusion_matrix_title_addition(X_y_binary):
    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    X, y_true = X_y_binary
    rs = np.random.RandomState(42)
    y_pred = np.round(y_true * rs.random(y_true.shape)).astype(int)
    fig = graph_confusion_matrix(y_true, y_pred, title_addition='with added title text')
    assert isinstance(fig, type(go.Figure()))
    fig_dict = fig.to_dict()
    assert fig_dict['layout']['title']['text'] == 'Confusion matrix with added title text, normalized using method "true"'


def test_get_permutation_importance_invalid_objective(X_y_regression, linear_regression_pipeline_class):
    X, y = X_y_regression
    pipeline = linear_regression_pipeline_class(parameters={}, random_state=np.random.RandomState(42))
    with pytest.raises(ValueError, match=f"Given objective 'MCC Multiclass' cannot be used with '{pipeline.name}'"):
        calculate_permutation_importance(pipeline, X, y, "mcc_multi")


@pytest.mark.parametrize("data_type", ['np', 'pd'])
def test_get_permutation_importance_binary(X_y_binary, data_type, logistic_regression_binary_pipeline_class):
    X, y = X_y_binary
    if data_type == 'pd':
        X = pd.DataFrame(X)
        y = pd.Series(y)
    pipeline = logistic_regression_binary_pipeline_class(parameters={}, random_state=np.random.RandomState(42))
    pipeline.fit(X, y)
    for objective in get_objectives(ProblemTypes.BINARY):
        permutation_importance = calculate_permutation_importance(pipeline, X, y, objective)
        assert list(permutation_importance.columns) == ["feature", "importance"]
        assert not permutation_importance.isnull().all().all()


def test_get_permutation_importance_multiclass(X_y_multi, logistic_regression_multiclass_pipeline_class):
    X, y = X_y_multi
    pipeline = logistic_regression_multiclass_pipeline_class(parameters={}, random_state=np.random.RandomState(42))
    pipeline.fit(X, y)
    for objective in get_objectives(ProblemTypes.MULTICLASS):
        permutation_importance = calculate_permutation_importance(pipeline, X, y, objective)
        assert list(permutation_importance.columns) == ["feature", "importance"]
        assert not permutation_importance.isnull().all().all()


def test_get_permutation_importance_regression(X_y_regression, linear_regression_pipeline_class):
    X, y = X_y_regression
    pipeline = linear_regression_pipeline_class(parameters={}, random_state=np.random.RandomState(42))
    pipeline.fit(X, y)
    for objective in get_objectives(ProblemTypes.REGRESSION):
        permutation_importance = calculate_permutation_importance(pipeline, X, y, objective)
        assert list(permutation_importance.columns) == ["feature", "importance"]
        assert not permutation_importance.isnull().all().all()


def test_get_permutation_importance_correlated_features(logistic_regression_binary_pipeline_class):
    y = pd.Series([1, 0, 1, 1])
    X = pd.DataFrame()
    X["correlated"] = y * 2
    X["not correlated"] = [-1, -1, -1, 0]
    y = y.astype(bool)
    pipeline = logistic_regression_binary_pipeline_class(parameters={}, random_state=np.random.RandomState(42))
    pipeline.fit(X, y)
    importance = calculate_permutation_importance(pipeline, X, y, objective="log_loss_binary", random_state=0)
    assert list(importance.columns) == ["feature", "importance"]
    assert not importance.isnull().all().all()
    correlated_importance_val = importance["importance"][importance.index[importance["feature"] == "correlated"][0]]
    not_correlated_importance_val = importance["importance"][importance.index[importance["feature"] == "not correlated"][0]]
    assert correlated_importance_val > not_correlated_importance_val


def test_graph_permutation_importance(X_y_binary, test_pipeline):
    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    X, y = X_y_binary
    clf = test_pipeline
    clf.fit(X, y)
    fig = graph_permutation_importance(test_pipeline, X, y, "log_loss_binary")
    assert isinstance(fig, go.Figure)
    fig_dict = fig.to_dict()
    assert fig_dict['layout']['title']['text'] == "Permutation Importance<br><sub>"\
                                                  "The relative importance of each input feature's overall "\
                                                  "influence on the pipelines' predictions, computed using the "\
                                                  "permutation importance algorithm.</sub>"
    assert len(fig_dict['data']) == 1

    perm_importance_data = calculate_permutation_importance(clf, X, y, "log_loss_binary")
    assert np.array_equal(fig_dict['data'][0]['x'][::-1], perm_importance_data['importance'].values)
    assert np.array_equal(fig_dict['data'][0]['y'][::-1], perm_importance_data['feature'])


@patch('evalml.model_understanding.graphs.calculate_permutation_importance')
def test_graph_permutation_importance_show_all_features(mock_perm_importance):
    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    mock_perm_importance.return_value = pd.DataFrame({"feature": ["f1", "f2"], "importance": [0.0, 0.6]})

    figure = graph_permutation_importance(test_pipeline, pd.DataFrame(), pd.Series(), "log_loss_binary")
    assert isinstance(figure, go.Figure)

    data = figure.data[0]
    assert (np.any(data['x'] == 0.0))


@patch('evalml.model_understanding.graphs.calculate_permutation_importance')
def test_graph_permutation_importance_threshold(mock_perm_importance):
    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    mock_perm_importance.return_value = pd.DataFrame({"feature": ["f1", "f2"], "importance": [0.0, 0.6]})

    with pytest.raises(ValueError, match="Provided importance threshold of -0.1 must be greater than or equal to 0"):
        fig = graph_permutation_importance(test_pipeline, pd.DataFrame(), pd.Series(), "log_loss_binary", importance_threshold=-0.1)
    fig = graph_permutation_importance(test_pipeline, pd.DataFrame(), pd.Series(), "log_loss_binary", importance_threshold=0.5)
    assert isinstance(fig, go.Figure)

    data = fig.data[0]
    assert (np.all(data['x'] >= 0.5))


def test_cost_benefit_matrix_vs_threshold(X_y_binary, logistic_regression_binary_pipeline_class):
    X, y = X_y_binary
    cbm = CostBenefitMatrix(true_positive=1, true_negative=-1,
                            false_positive=-7, false_negative=-2)
    pipeline = logistic_regression_binary_pipeline_class(parameters={})
    pipeline.fit(X, y)
    original_pipeline_threshold = pipeline.threshold
    cost_benefit_df = binary_objective_vs_threshold(pipeline, X, y, cbm)
    assert list(cost_benefit_df.columns) == ['threshold', 'score']
    assert cost_benefit_df.shape == (101, 2)
    assert not cost_benefit_df.isnull().all().all()
    assert pipeline.threshold == original_pipeline_threshold


def test_binary_objective_vs_threshold(X_y_binary, logistic_regression_binary_pipeline_class):
    X, y = X_y_binary
    pipeline = logistic_regression_binary_pipeline_class(parameters={})
    pipeline.fit(X, y)

    # test objective with score_needs_proba == True
    with pytest.raises(ValueError, match="Objective `score_needs_proba` must be False"):
        binary_objective_vs_threshold(pipeline, X, y, 'log_loss_binary')

    # test with non-binary objective
    with pytest.raises(ValueError, match="can only be calculated for binary classification objectives"):
        binary_objective_vs_threshold(pipeline, X, y, 'f1_micro')

    # test objective with score_needs_proba == False
    results_df = binary_objective_vs_threshold(pipeline, X, y, 'f1')
    assert list(results_df.columns) == ['threshold', 'score']
    assert results_df.shape == (101, 2)
    assert not results_df.isnull().all().all()


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
def test_binary_objective_vs_threshold_steps(mock_score,
                                             X_y_binary, logistic_regression_binary_pipeline_class):
    X, y = X_y_binary
    cbm = CostBenefitMatrix(true_positive=1, true_negative=-1,
                            false_positive=-7, false_negative=-2)
    pipeline = logistic_regression_binary_pipeline_class(parameters={})
    pipeline.fit(X, y)
    mock_score.return_value = {"Cost Benefit Matrix": 0.2}
    cost_benefit_df = binary_objective_vs_threshold(pipeline, X, y, cbm, steps=234)
    mock_score.assert_called()
    assert list(cost_benefit_df.columns) == ['threshold', 'score']
    assert cost_benefit_df.shape == (235, 2)


@patch('evalml.model_understanding.graphs.binary_objective_vs_threshold')
def test_graph_binary_objective_vs_threshold(mock_cb_thresholds, X_y_binary, logistic_regression_binary_pipeline_class):
    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    X, y = X_y_binary
    pipeline = logistic_regression_binary_pipeline_class(parameters={})
    cbm = CostBenefitMatrix(true_positive=1, true_negative=-1,
                            false_positive=-7, false_negative=-2)

    mock_cb_thresholds.return_value = pd.DataFrame({'threshold': [0, 0.5, 1.0],
                                                    'score': [100, -20, 5]})

    figure = graph_binary_objective_vs_threshold(pipeline, X, y, cbm)
    assert isinstance(figure, go.Figure)
    data = figure.data[0]
    assert not np.any(np.isnan(data['x']))
    assert not np.any(np.isnan(data['y']))
    assert np.array_equal(data['x'], mock_cb_thresholds.return_value['threshold'])
    assert np.array_equal(data['y'], mock_cb_thresholds.return_value['score'])


@pytest.mark.parametrize("problem_type", [ProblemTypes.BINARY, ProblemTypes.MULTICLASS, ProblemTypes.REGRESSION])
def test_partial_dependence_problem_types(problem_type, X_y_binary, X_y_multi, X_y_regression,
                                          logistic_regression_binary_pipeline_class,
                                          logistic_regression_multiclass_pipeline_class,
                                          linear_regression_pipeline_class):
    if problem_type == ProblemTypes.BINARY:
        X, y = X_y_binary
        pipeline = logistic_regression_binary_pipeline_class(parameters={})

    elif problem_type == ProblemTypes.MULTICLASS:
        X, y = X_y_multi
        pipeline = logistic_regression_multiclass_pipeline_class(parameters={})

    elif problem_type == ProblemTypes.REGRESSION:
        X, y = X_y_regression
        pipeline = linear_regression_pipeline_class(parameters={})

    pipeline.fit(X, y)
    part_dep = partial_dependence(pipeline, X, feature=0, grid_resolution=20)
    assert list(part_dep.columns) == ["feature_values", "partial_dependence"]
    assert len(part_dep["partial_dependence"]) == 20
    assert len(part_dep["feature_values"]) == 20
    assert not part_dep.isnull().all().all()


def test_partial_dependence_string_feature(logistic_regression_binary_pipeline_class):
    X, y = load_breast_cancer()
    pipeline = logistic_regression_binary_pipeline_class(parameters={})
    pipeline.fit(X, y)
    part_dep = partial_dependence(pipeline, X, feature="mean radius", grid_resolution=20)
    assert list(part_dep.columns) == ["feature_values", "partial_dependence"]
    assert len(part_dep["partial_dependence"]) == 20
    assert len(part_dep["feature_values"]) == 20
    assert not part_dep.isnull().all().all()


@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_partial_dependence_baseline(mock_fit, X_y_binary):
    X, y = X_y_binary

    class BaselineTestPipeline(BinaryClassificationPipeline):
        component_graph = ["Baseline Classifier"]
    pipeline = BaselineTestPipeline({})
    pipeline.fit(X, y)
    with pytest.raises(ValueError, match="Partial dependence plots are not supported for Baseline pipelines"):
        partial_dependence(pipeline, X, feature=0, grid_resolution=20)


def test_partial_dependence_catboost(X_y_binary, has_minimal_dependencies):
    if not has_minimal_dependencies:
        X, y = X_y_binary

        class CatBoostTestPipeline(BinaryClassificationPipeline):
            component_graph = ["CatBoost Classifier"]
        pipeline = CatBoostTestPipeline({})
        pipeline.fit(X, y)
        partial_dependence(pipeline, X, feature=0, grid_resolution=20)


def test_partial_dependence_not_fitted(X_y_binary, logistic_regression_binary_pipeline_class):
    X, y = X_y_binary
    pipeline = logistic_regression_binary_pipeline_class(parameters={})
    with pytest.raises(ValueError, match="Pipeline to calculate partial dependence for must be fitted"):
        partial_dependence(pipeline, X, feature=0, grid_resolution=20)


def test_graph_partial_dependence(test_pipeline):
    X, y = load_breast_cancer()

    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    clf = test_pipeline
    clf.fit(X, y)
    fig = graph_partial_dependence(clf, X, feature='mean radius', grid_resolution=20)
    assert isinstance(fig, go.Figure)
    fig_dict = fig.to_dict()
    assert fig_dict['layout']['title']['text'] == "Partial Dependence of 'mean radius'"
    assert len(fig_dict['data']) == 1

    part_dep_data = partial_dependence(clf, X, feature='mean radius', grid_resolution=20)
    assert np.array_equal(fig_dict['data'][0]['x'], part_dep_data['feature_values'])
    assert np.array_equal(fig_dict['data'][0]['y'], part_dep_data['partial_dependence'].values)


@patch('evalml.utils.gen_utils.import_or_warn')
@patch('evalml.utils.gen_utils.jupyter_check')
def test_jupyter_graph_check(import_check, jupyter_check, X_y_binary, test_pipeline):
    pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    X, y = X_y_binary
    clf = test_pipeline
    clf.fit(X, y)
    cbm = CostBenefitMatrix(true_positive=1, true_negative=-1, false_positive=-7, false_negative=-2)
    jupyter_check.return_value = False
    with pytest.warns(None) as graph_valid:
        graph_permutation_importance(test_pipeline, X, y, "log_loss_binary")
        assert len(graph_valid) == 0
    with pytest.warns(None) as graph_valid:
        graph_confusion_matrix(y, y)
        assert len(graph_valid) == 0
    with pytest.warns(None) as graph_valid:
        rs = np.random.RandomState(42)
        y_pred_proba = y * rs.random(y.shape)
        graph_roc_curve(y, y_pred_proba)
        assert len(graph_valid) == 0
    with pytest.warns(None) as graph_valid:
        graph_partial_dependence(clf, X, feature=0, grid_resolution=20)
        assert len(graph_valid) == 0
    with pytest.warns(None) as graph_valid:
        graph_binary_objective_vs_threshold(test_pipeline, X, y, cbm)
        assert len(graph_valid) == 0
    with pytest.warns(None) as graph_valid:
        rs = np.random.RandomState(42)
        y_pred_proba = y * rs.random(y.shape)
        graph_precision_recall_curve(y, y_pred_proba)
        assert len(graph_valid) == 0
