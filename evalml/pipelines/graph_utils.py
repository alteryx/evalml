import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import auc as sklearn_auc
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from sklearn.metrics import roc_curve as sklearn_roc_curve
from sklearn.utils.multiclass import unique_labels

from evalml.utils import import_or_raise


def roc_curve(y_true, y_pred_proba):
    """
    Given labels and binary classifier predicted probabilities, compute and return the data representing a Receiver Operating Characteristic (ROC) curve.

    Arguments:
        y_true (pd.Series or np.array): true binary labels.
        y_pred_proba (pd.Series or np.array): predictions from a binary classifier, before thresholding has been applied. Note this should be the predicted probability for the "true" label.

    Returns:
        dict: Dictionary containing metrics used to generate an ROC plot, with the following keys:
                  * `fpr_rates`: False positive rates.
                  * `tpr_rates`: True positive rates.
                  * `thresholds`: Threshold values used to produce each pair of true/false positive rates.
                  * `auc_score`: The area under the ROC curve.
    """
    fpr_rates, tpr_rates, thresholds = sklearn_roc_curve(y_true, y_pred_proba)
    auc_score = sklearn_auc(fpr_rates, tpr_rates)
    return {'fpr_rates': fpr_rates,
            'tpr_rates': tpr_rates,
            'thresholds': thresholds,
            'auc_score': auc_score}


def graph_roc_curve(y_true, y_pred_proba, title_addition=None):
    """Generate and display a Receiver Operating Characteristic (ROC) plot.

    Arguments:
        y_true (pd.Series or np.array): true binary labels.
        y_pred_proba (pd.Series or np.array): predictions from a binary classifier, before thresholding has been applied. Note this should be the predicted probability for the "true" label.
        title_addition (str or None): if not None, append to plot title. Default None.

    Returns:
        plotly.Figure representing the ROC plot generated
    """
    _go = import_or_raise("plotly.graph_objects", error_msg="Cannot find dependency plotly.graph_objects")
    roc_curve_data = roc_curve(y_true, y_pred_proba)
    title = 'Receiver Operating Characteristic{}'.format('' if title_addition is None else (' ' + title_addition))
    layout = _go.Layout(title={'text': title},
                        xaxis={'title': 'False Positive Rate', 'range': [-0.05, 1.05]},
                        yaxis={'title': 'True Positive Rate', 'range': [-0.05, 1.05]})
    data = []
    data.append(_go.Scatter(x=roc_curve_data['fpr_rates'], y=roc_curve_data['tpr_rates'],
                            name='ROC (AUC {:06f})'.format(roc_curve_data['auc_score']),
                            line=dict(width=3)))
    data.append(_go.Scatter(x=[0, 1], y=[0, 1],
                            name='Trivial Model (AUC 0.5)',
                            line=dict(dash='dash')))
    return _go.Figure(layout=layout, data=data)


def confusion_matrix(y_true, y_predicted, normalize_method='true'):
    """Confusion matrix for binary and multiclass classification.

    Arguments:
        y_true (pd.Series or np.array): true binary labels.
        y_pred (pd.Series or np.array): predictions from a binary classifier.
        normalize_method ({'true', 'pred', 'all'}): Normalization method. Supported options are: 'true' to normalize by row, 'pred' to normalize by column, or 'all' to normalize by all values. Defaults to 'true'.

    Returns:
        np.array: confusion matrix
    """
    labels = unique_labels(y_true, y_predicted)
    conf_mat = sklearn_confusion_matrix(y_true, y_predicted)
    conf_mat = pd.DataFrame(conf_mat, columns=labels)
    if normalize_method is not None:
        return normalize_confusion_matrix(conf_mat, normalize_method=normalize_method)
    return conf_mat


def normalize_confusion_matrix(conf_mat, normalize_method='true'):
    """Normalizes a confusion matrix.

    Arguments:
        conf_mat (pd.DataFrame or np.array): confusion matrix to normalize.
        normalize_method ({'true', 'pred', 'all'}): Normalization method. Supported options are: 'true' to normalize by row, 'pred' to normalize by column, or 'all' to normalize by all values. Defaults to 'true'.

    Returns:
        A normalized version of the input confusion matrix.
    """
    with warnings.catch_warnings(record=True) as w:
        if normalize_method == 'true':
            conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
        elif normalize_method == 'pred':
            conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=0)
        elif normalize_method == 'all':
            conf_mat = conf_mat.astype('float') / conf_mat.sum().sum()
        else:
            raise ValueError('Invalid value provided for "normalize_method": %s'.format(normalize_method))
        if w and "invalid value encountered in" in str(w[0].message):
            raise ValueError("Sum of given axis is 0 and normalization is not possible. Please select another option.")
    return conf_mat


def graph_confusion_matrix(y_true, y_pred, normalize_method='true', title_addition=None):
    """Generate and display a confusion matrix plot.

    If `normalize_method` is set, hover text will show raw count, otherwise hover text will show count normalized with method 'true'.

    Arguments:
        y_true (pd.Series or np.array): true binary labels.
        y_pred (pd.Series or np.array): predictions from a binary classifier.
        normalize_method ({'true', 'pred', 'all'}): Normalization method. Supported options are: 'true' to normalize by row, 'pred' to normalize by column, or 'all' to normalize by all values. Defaults to 'true'.
        title_addition (str or None): if not None, append to plot title. Default None.

    Returns:
        plotly.Figure representing the confusion matrix plot generated
    """
    _go = import_or_raise("plotly.graph_objects", error_msg="Cannot find dependency plotly.graph_objects")
    conf_mat = confusion_matrix(y_true, y_pred, normalize_method=None)
    conf_mat_normalized = confusion_matrix(y_true, y_pred, normalize_method=normalize_method or 'true')
    labels = conf_mat.columns
    reversed_labels = labels[::-1]

    title = 'Confusion matrix{}{}'.format(
        '' if title_addition is None else (' ' + title_addition),
        '' if normalize_method is None else (', normalized using method "' + normalize_method + '"'))
    z_data, custom_data = (conf_mat, conf_mat_normalized) if normalize_method is None else (conf_mat_normalized, conf_mat)
    primary_heading, secondary_heading = ('Raw', 'Normalized') if normalize_method is None else ('Normalized', 'Raw')
    hover_text = '<br><b>' + primary_heading + ' Count</b>: %{z}<br><b>' + secondary_heading + ' Count</b>: %{customdata} <br>'
    # the "<extra> tags at the end are necessary to remove unwanted trace info
    hover_template = '<b>True</b>: %{y}<br><b>Predicted</b>: %{x}' + hover_text + '<extra></extra>'
    layout = _go.Layout(title={'text': title},
                        xaxis={'title': 'Predicted Label', 'type': 'category', 'tickvals': labels},
                        yaxis={'title': 'True Label', 'type': 'category', 'tickvals': reversed_labels})
    return _go.Figure(data=_go.Heatmap(x=labels, y=reversed_labels, z=z_data,
                                       customdata=custom_data,
                                       hovertemplate=hover_template,
                                       colorscale='Blues'),
                      layout=layout)
