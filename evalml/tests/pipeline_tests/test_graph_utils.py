import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import label_binarize

from evalml.pipelines.graph_utils import (
    graph_confusion_matrix,
    graph_precision_recall_curve,
    graph_roc_curve,
    precision_recall_curve,
    roc_curve
)
from evalml.utils.graph_utils import confusion_matrix


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
