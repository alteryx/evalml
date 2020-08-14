
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from sklearn.utils.multiclass import unique_labels
from evalml.utils import import_or_raise


def confusion_matrix(y_true, y_predicted, normalize_method='true'):
    """Confusion matrix for binary and multiclass classification.

    Arguments:
        y_true (pd.Series or np.array): true binary labels.
        y_pred (pd.Series or np.array): predictions from a binary classifier.
        normalize_method ({'true', 'pred', 'all'}): Normalization method. Supported options are: 'true' to normalize by row, 'pred' to normalize by column, or 'all' to normalize by all values. Defaults to 'true'.

    Returns:
        pd.DataFrame: confusion matrix
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
        pd.DataFrame: normalized version of the input confusion matrix.
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


def graph_cost_benefit_thresholds(pipeline, X, y, cbm_objective, steps=100):
    """Generates cost-benefit matrix score vs. threshold graph for a fitted pipeline.
    """
    _go = import_or_raise("plotly.graph_objects", error_msg="Cannot find dependency plotly.graph_objects")

    ypred_proba = pipeline.predict_proba(X).iloc[:, 1]
    thresholds = np.linspace(0, 1, steps + 1)
    costs = []
    for threshold in thresholds:
        y_predicted = cbm_objective.decision_function(ypred_proba=ypred_proba, threshold=threshold, X=X)
        cost = cbm_objective.objective_function(y, y_predicted, X=X)
        costs.append(cost)
    df = pd.DataFrame({"thresholds": thresholds, "costs": costs})

    max_costs = df["costs"].max()
    min_costs = df["costs"].min()
    margins = abs(max_costs - min_costs) * 0.05
    title = 'Cost-Benefit Matrix Scores vs. Thresholds'
    layout = _go.Layout(title={'text': title},
                        xaxis={'title': 'Thresholds', 'range': [-0.05, 1.05]},
                        yaxis={'title': 'Cost-Benefit Score', 'range': [min_costs - margins, max_costs + margins]})
    data = []
    data.append(_go.Scatter(x=df['thresholds'], y=df['costs'],
                            name='Scores vs. Thresholds',
                            line=dict(width=3)))
    return _go.Figure(layout=layout, data=data)
