import warnings

import numpy as np
import pandas as pd
from sklearn.inspection import \
    permutation_importance as sk_permutation_importance
from sklearn.metrics import auc as sklearn_auc
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from sklearn.metrics import \
    precision_recall_curve as sklearn_precision_recall_curve
from sklearn.metrics import roc_curve as sklearn_roc_curve
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.multiclass import unique_labels

from evalml.objectives import get_objective
from evalml.utils import import_or_raise


def precision_recall_curve(y_true, y_pred_proba):
    """
    Given labels and binary classifier predicted probabilities, compute and return the data representing a precision-recall curve.

    Arguments:
        y_true (pd.Series or np.array): true binary labels.
        y_pred_proba (pd.Series or np.array): predictions from a binary classifier, before thresholding has been applied. Note this should be the predicted probability for the "true" label.

    Returns:
        list: Dictionary containing metrics used to generate a precision-recall plot, with the following keys:

                  * `precision`: Precision values.
                  * `recall`: Recall values.
                  * `thresholds`: Threshold values used to produce the precision and recall.
                  * `auc_score`: The area under the ROC curve.
    """
    precision, recall, thresholds = sklearn_precision_recall_curve(y_true, y_pred_proba)
    auc_score = sklearn_auc(recall, precision)
    return {'precision': precision,
            'recall': recall,
            'thresholds': thresholds,
            'auc_score': auc_score}


def graph_precision_recall_curve(y_true, y_pred_proba, title_addition=None):
    """Generate and display a precision-recall plot.

    Arguments:
        y_true (pd.Series or np.array): true binary labels.
        y_pred_proba (pd.Series or np.array): predictions from a binary classifier, before thresholding has been applied. Note this should be the predicted probability for the "true" label.
        title_addition (str or None): if not None, append to plot title. Default None.

    Returns:
        plotly.Figure representing the precision-recall plot generated
    """
    _go = import_or_raise("plotly.graph_objects", error_msg="Cannot find dependency plotly.graph_objects")

    if isinstance(y_true, pd.Series):
        y_true = y_true.to_numpy()
    if isinstance(y_pred_proba, (pd.Series, pd.DataFrame)):
        y_pred_proba = y_pred_proba.to_numpy()

    precision_recall_curve_data = precision_recall_curve(y_true, y_pred_proba)
    title = 'Precision-Recall{}'.format('' if title_addition is None else (' ' + title_addition))
    layout = _go.Layout(title={'text': title},
                        xaxis={'title': 'Recall', 'range': [-0.05, 1.05]},
                        yaxis={'title': 'Precision', 'range': [-0.05, 1.05]})
    data = []
    data.append(_go.Scatter(x=precision_recall_curve_data['recall'], y=precision_recall_curve_data['precision'],
                            name='Precision-Recall (AUC {:06f})'.format(precision_recall_curve_data['auc_score']),
                            line=dict(width=3)))
    return _go.Figure(layout=layout, data=data)


def roc_curve(y_true, y_pred_proba):
    """
    Given labels and classifier predicted probabilities, compute and return the data representing a Receiver Operating Characteristic (ROC) curve.

    Arguments:
        y_true (pd.Series or np.array): true labels.
        y_pred_proba (pd.Series or np.array): predictions from a classifier, before thresholding has been applied. Note that 1 dimensional input is expected.


    Returns:
        dict: Dictionary containing metrics used to generate an ROC plot, with the following keys:
                  * `fpr_rate`: False positive rate.
                  * `tpr_rate`: True positive rate.
                  * `threshold`: Threshold values used to produce each pair of true/false positive rates.
                  * `auc_score`: The area under the ROC curve.
    """
    fpr_rates, tpr_rates, thresholds = sklearn_roc_curve(y_true, y_pred_proba)
    auc_score = sklearn_auc(fpr_rates, tpr_rates)
    return {'fpr_rates': fpr_rates,
            'tpr_rates': tpr_rates,
            'thresholds': thresholds,
            'auc_score': auc_score}


def graph_roc_curve(y_true, y_pred_proba, custom_class_names=None, title_addition=None):
    """Generate and display a Receiver Operating Characteristic (ROC) plot.

    Arguments:
        y_true (pd.Series or np.array): true labels.
        y_pred_proba (pd.Series or np.array): predictions from a classifier, before thresholding has been applied. Note this should a one dimensional array with the predicted probability for the "true" label in the binary case.
        custom_class_labels (list or None): if not None, custom labels for classes. Default None.
        title_addition (str or None): if not None, append to plot title. Default None.

    Returns:
        plotly.Figure representing the ROC plot generated
    """
    _go = import_or_raise("plotly.graph_objects", error_msg="Cannot find dependency plotly.graph_objects")

    if isinstance(y_true, pd.Series):
        y_true = y_true.to_numpy()
    if isinstance(y_pred_proba, (pd.Series, pd.DataFrame)):
        y_pred_proba = y_pred_proba.to_numpy()

    if y_pred_proba.ndim == 1:
        y_pred_proba = y_pred_proba.reshape(-1, 1)

    nan_indices = np.logical_or(np.isnan(y_true), np.isnan(y_pred_proba).any(axis=1))
    y_true = y_true[~nan_indices]
    y_pred_proba = y_pred_proba[~nan_indices]

    lb = LabelBinarizer()
    lb.fit(np.unique(y_true))
    y_one_hot_true = lb.transform(y_true)
    n_classes = y_one_hot_true.shape[1]

    if custom_class_names and len(custom_class_names) != n_classes:
        raise ValueError('Number of custom class names does not match number of classes')

    title = 'Receiver Operating Characteristic{}'.format('' if title_addition is None else (' ' + title_addition))
    layout = _go.Layout(title={'text': title},
                        xaxis={'title': 'False Positive Rate', 'range': [-0.05, 1.05]},
                        yaxis={'title': 'True Positive Rate', 'range': [-0.05, 1.05]})

    data = []
    for i in range(n_classes):
        roc_curve_data = roc_curve(y_one_hot_true[:, i], y_pred_proba[:, i])
        data.append(_go.Scatter(x=roc_curve_data['fpr_rates'], y=roc_curve_data['tpr_rates'],
                                name='Class {name} (AUC {:06f})'
                                .format(roc_curve_data['auc_score'],
                                        name=i + 1 if custom_class_names is None else custom_class_names[i]),
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
    if isinstance(y_true, pd.Series):
        y_true = y_true.to_numpy()
    if isinstance(y_predicted, pd.Series):
        y_predicted = y_predicted.to_numpy()

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

    if isinstance(y_true, pd.Series):
        y_true = y_true.to_numpy()
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.to_numpy()

    conf_mat = confusion_matrix(y_true, y_pred, normalize_method=None)
    conf_mat_normalized = confusion_matrix(y_true, y_pred, normalize_method=normalize_method or 'true')
    labels = conf_mat.columns

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
                        yaxis={'title': 'True Label', 'type': 'category', 'tickvals': labels})
    fig = _go.Figure(data=_go.Heatmap(x=labels, y=labels, z=z_data,
                                      customdata=custom_data,
                                      hovertemplate=hover_template,
                                      colorscale='Blues'),
                     layout=layout)
    # plotly Heatmap y axis defaults to the reverse of what we want: https://community.plotly.com/t/heatmap-y-axis-is-reversed-by-default-going-against-standard-convention-for-matrices/32180
    fig.update_yaxes(autorange="reversed")
    return fig


def calculate_permutation_importance(pipeline, X, y, objective, n_repeats=5, n_jobs=None, random_state=0):
    """Calculates permutation importance for features.

    Arguments:
        pipeline (PipelineBase or subclass): fitted pipeline
        X (pd.DataFrame): the input data used to score and compute permutation importance
        y (pd.Series): the target labels
        objective (str, ObjectiveBase): objective to score on
        n_repeats (int): Number of times to permute a feature. Defaults to 5.
        n_jobs (int or None): Non-negative integer describing level of parallelism used for pipelines.
            None and 1 are equivalent. If set to -1, all CPUs are used. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.
        random_state (int, np.random.RandomState): The random seed/state. Defaults to 0.

    Returns:
        Mean feature importance scores over 5 shuffles.
    """
    objective = get_objective(objective)
    if objective.problem_type != pipeline.problem_type:
        raise ValueError(f"Given objective '{objective.name}' cannot be used with '{pipeline.name}'")

    def scorer(pipeline, X, y):
        scores = pipeline.score(X, y, objectives=[objective])
        return scores[objective.name] if objective.greater_is_better else -scores[objective.name]
    perm_importance = sk_permutation_importance(pipeline, X, y, n_repeats=n_repeats, scoring=scorer, n_jobs=n_jobs, random_state=random_state)
    mean_perm_importance = perm_importance["importances_mean"]
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    feature_names = list(X.columns)
    mean_perm_importance = list(zip(feature_names, mean_perm_importance))
    mean_perm_importance.sort(key=lambda x: x[1], reverse=True)
    return pd.DataFrame(mean_perm_importance, columns=["feature", "importance"])


def graph_permutation_importance(pipeline, X, y, objective, show_all_features=False):
    """Generate a bar graph of the pipeline's permutation importance.

    Arguments:
        pipeline (PipelineBase or subclass): Fitted pipeline
        X (pd.DataFrame): The input data used to score and compute permutation importance
        y (pd.Series): The target labels
        objective (str, ObjectiveBase): Objective to score on
        show_all_features (bool, optional) : If True, graph features with a permutation importance value of zero. Defaults to False.

    Returns:
        plotly.Figure, a bar graph showing features and their respective permutation importance.
    """
    go = import_or_raise("plotly.graph_objects", error_msg="Cannot find dependency plotly.graph_objects")
    perm_importance = calculate_permutation_importance(pipeline, X, y, objective)
    perm_importance['importance'] = perm_importance['importance']

    if not show_all_features:
        # Remove features with close to zero importance
        perm_importance = perm_importance[abs(perm_importance['importance']) >= 1e-3]
    # List is reversed to go from ascending order to descending order
    perm_importance = perm_importance.iloc[::-1]

    title = "Permutation Importance"
    subtitle = "The relative importance of each input feature's "\
               "overall influence on the pipelines' predictions, computed using "\
               "the permutation importance algorithm."
    data = [go.Bar(x=perm_importance['importance'],
                   y=perm_importance['feature'],
                   orientation='h'
                   )]

    layout = {
        'title': '{0}<br><sub>{1}</sub>'.format(title, subtitle),
        'height': 800,
        'xaxis_title': 'Permutation Importance',
        'yaxis_title': 'Feature',
        'yaxis': {
            'type': 'category'
        }
    }

    fig = go.Figure(data=data, layout=layout)
    return fig
