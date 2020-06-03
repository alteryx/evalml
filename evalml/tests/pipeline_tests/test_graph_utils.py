import numpy as np
import pandas as pd
import pytest

from evalml.pipelines.graph_utils import (
    confusion_matrix,
    graph_confusion_matrix,
    graph_precision_recall_curve,
    graph_roc_curve,
    normalize_confusion_matrix,
    precision_recall_curve,
    roc_curve
)


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


def test_graph_precision_recall_curve(X_y):
    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    X, y_true = X_y
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


def test_graph_precision_recall_curve_title_addition(X_y):
    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    X, y_true = X_y
    rs = np.random.RandomState(42)
    y_pred_proba = y_true * rs.random(y_true.shape)
    fig = graph_precision_recall_curve(y_true, y_pred_proba, title_addition='with added title text')
    assert isinstance(fig, type(go.Figure()))
    fig_dict = fig.to_dict()
    assert fig_dict['layout']['title']['text'] == 'Precision-Recall with added title text'


def test_roc_curve_binary():
    y_true = np.array([1, 1, 0, 0])
    y_predict_proba = np.array([0.1, 0.4, 0.35, 0.8])
    roc_curve_data = roc_curve(y_true, y_predict_proba)
    fpr_rates = roc_curve_data.get('fpr_rates')[0]
    tpr_rates = roc_curve_data.get('tpr_rates')[0]
    thresholds = roc_curve_data.get('thresholds')[0]
    auc_score = roc_curve_data.get('auc_score')[0]
    fpr_expected = np.array([0, 0.5, 0.5, 1, 1])
    tpr_expected = np.array([0, 0, 0.5, 0.5, 1])
    thresholds_expected = np.array([1.8, 0.8, 0.4, 0.35, 0.1])
    assert np.array_equal(fpr_expected, fpr_rates)
    assert np.array_equal(tpr_expected, tpr_rates)
    assert np.array_equal(thresholds_expected, thresholds)
    assert auc_score == pytest.approx(0.25, 1e-5)

def test_roc_curve_multiclass():
    y_true = np.array([[0, 0, 1],
                       [0, 1, 0],
                       [1, 0, 0],
                       [0, 0, 1],
                       [1, 0, 0],
                       [0, 0, 1],
                       [1, 0, 0]])
    y_predict_proba = np.array([[0, 0.1, 0.9],
                                [0.2,  0.7, 0.1],
                                [0.6, 0.2, 0.2],
                                [0.1, 0.1,  0.8],
                                [0.9, 0.05, 0.05],
                                [0.1, 0.05,  0.85],
                                [0.9, 0.05, 0.05]])
    roc_curve_data = roc_curve(y_true, y_predict_proba, n_classes=3)
    fpr_rates = roc_curve_data.get('fpr_rates')
    tpr_rates = roc_curve_data.get('tpr_rates')
    thresholds = roc_curve_data.get('thresholds')
    auc_score = roc_curve_data.get('auc_score')
    fpr_expected = {0: np.array([0., 0., 0., 0.25, 0.75, 1.]), 
                    1: np.array([0., 0., 0.16666667, 0.5, 1.]), 
                    2: np.array([0., 0., 0., 0.5, 1.])}
    tpr_expected = {0: np.array([0., 0.66666667, 1., 1., 1., 1.]), 
                    1: np.array([0., 1., 1., 1., 1.]),
                     2: np.array([0., 0.33333333, 1., 1., 1.])}
    thresholds_expected = {0: np.array([1.9, 0.9, 0.6, 0.2, 0.1, 0.]), 
                           1: np.array([1.7 , 0.7 , 0.2 , 0.1 , 0.05]), 
                           2: np.array([1.9 , 0.9 , 0.8 , 0.1 , 0.05])}
    for i in range(3):
        assert np.allclose(fpr_expected[i], fpr_rates[i])
        assert np.allclose(tpr_expected[i], tpr_rates[i])
        assert np.allclose(thresholds_expected[i], thresholds[i])
    assert auc_score == pytest.approx({0: 1., 1: 1., 2: 1.}, 1e-5)


def test_confusion_matrix():
    y_true = [2, 0, 2, 2, 0, 1]
    y_predicted = [0, 0, 2, 2, 0, 2]
    conf_mat = confusion_matrix(y_true, y_predicted, normalize_method=None)
    conf_mat_expected = np.array([[2, 0, 0], [0, 0, 1], [1, 0, 2]])
    assert np.array_equal(conf_mat_expected, conf_mat)
    conf_mat = confusion_matrix(y_true, y_predicted, normalize_method='true')
    conf_mat_expected = np.array([[1, 0, 0], [0, 0, 1], [1 / 3.0, 0, 2 / 3.0]])
    assert np.array_equal(conf_mat_expected, conf_mat)
    conf_mat = confusion_matrix(y_true, y_predicted, normalize_method='pred')
    conf_mat_expected = np.array([[2 / 3.0, np.nan, 0], [0, np.nan, 1 / 3.0], [1 / 3.0, np.nan, 2 / 3.0]])
    assert np.allclose(conf_mat_expected, conf_mat, equal_nan=True)
    conf_mat = confusion_matrix(y_true, y_predicted, normalize_method='all')
    conf_mat_expected = np.array([[1 / 3.0, 0, 0], [0, 0, 1 / 6.0], [1 / 6.0, 0, 1 / 3.0]])
    assert np.array_equal(conf_mat_expected, conf_mat)
    with pytest.raises(ValueError, match='Invalid value provided'):
        conf_mat = confusion_matrix(y_true, y_predicted, normalize_method='Invalid Option')


def test_normalize_confusion_matrix():
    conf_mat = np.array([[2, 3, 0], [0, 1, 1], [1, 0, 2]])
    conf_mat_normalized = normalize_confusion_matrix(conf_mat)
    assert all(conf_mat_normalized.sum(axis=1) == 1.0)

    conf_mat_normalized = normalize_confusion_matrix(conf_mat, 'pred')
    for col_sum in conf_mat_normalized.sum(axis=0):
        assert col_sum == 1.0 or col_sum == 0.0

    conf_mat_normalized = normalize_confusion_matrix(conf_mat, 'all')
    assert conf_mat_normalized.sum() == 1.0

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


def test_graph_roc_curve_binary(X_y):
    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    X, y_true = X_y
    rs = np.random.RandomState(42)
    y_pred_proba = y_true * rs.random(y_true.shape)
    fig = graph_roc_curve(y_true, y_pred_proba)
    assert isinstance(fig, type(go.Figure()))
    fig_dict = fig.to_dict()
    assert fig_dict['layout']['title']['text'] == 'Receiver Operating Characteristic'
    assert len(fig_dict['data']) == 2
    roc_curve_data = roc_curve(y_true, y_pred_proba)
    assert np.array_equal(fig_dict['data'][0]['x'], roc_curve_data['fpr_rates'][0])
    assert np.array_equal(fig_dict['data'][0]['y'], roc_curve_data['tpr_rates'][0])
    assert fig_dict['data'][0]['name'] == 'ROC (AUC {:06f}) of Class 1'.format(roc_curve_data['auc_score'][0])
    assert np.array_equal(fig_dict['data'][1]['x'], np.array([0, 1]))
    assert np.array_equal(fig_dict['data'][1]['y'], np.array([0, 1]))
    assert fig_dict['data'][1]['name'] == 'Trivial Model (AUC 0.5)'


def test_graph_roc_curve_title_addition(X_y):
    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    X, y_true = X_y
    rs = np.random.RandomState(42)
    y_pred_proba = y_true * rs.random(y_true.shape)
    fig = graph_roc_curve(y_true, y_pred_proba, title_addition='with added title text')
    assert isinstance(fig, type(go.Figure()))
    fig_dict = fig.to_dict()
    assert fig_dict['layout']['title']['text'] == 'Receiver Operating Characteristic with added title text'


def test_graph_confusion_matrix_default(X_y):
    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    X, y_true = X_y
    rs = np.random.RandomState(42)
    y_pred = np.round(y_true * rs.random(y_true.shape)).astype(int)
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


def test_graph_confusion_matrix_norm_disabled(X_y):
    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    X, y_true = X_y
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


def test_graph_confusion_matrix_title_addition(X_y):
    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    X, y_true = X_y
    rs = np.random.RandomState(42)
    y_pred = np.round(y_true * rs.random(y_true.shape)).astype(int)
    fig = graph_confusion_matrix(y_true, y_pred, title_addition='with added title text')
    assert isinstance(fig, type(go.Figure()))
    fig_dict = fig.to_dict()
    assert fig_dict['layout']['title']['text'] == 'Confusion matrix with added title text, normalized using method "true"'
