"""Standard metrics used for model understanding."""
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import auc as sklearn_auc
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from sklearn.metrics import precision_recall_curve as sklearn_precision_recall_curve
from sklearn.metrics import roc_curve as sklearn_roc_curve
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.multiclass import unique_labels
from woodwork.logical_types import BooleanNullable, IntegerNullable

from evalml.exceptions import NoPositiveLabelException
from evalml.utils import import_or_raise, infer_feature_types, jupyter_check


def _convert_ww_series_to_np_array(ww_series):
    """Helper function to properly convert IntegerNullable/BooleanNullable Woodwork series to numpy arrays.

    Args:
        ww_series: Woodwork init-ed series possibly containing IntegerNullable or BooleanNullable datatype

    Returns:
        numpy.ndarray: The values of ww_series but in an array.
    """
    np_series = ww_series.to_numpy()
    if isinstance(ww_series.ww.logical_type, BooleanNullable):
        np_series = np_series.astype("bool")
    if isinstance(ww_series.ww.logical_type, IntegerNullable):
        try:
            np_series = np_series.astype("int64")
        except TypeError:
            np_series = ww_series.astype(float).to_numpy()

    return np_series


def confusion_matrix(y_true, y_predicted, normalize_method="true"):
    """Confusion matrix for binary and multiclass classification.

    Args:
        y_true (pd.Series or np.ndarray): True binary labels.
        y_predicted (pd.Series or np.ndarray): Predictions from a binary classifier.
        normalize_method ({'true', 'pred', 'all', None}): Normalization method to use, if not None. Supported options are: 'true' to normalize by row, 'pred' to normalize by column, or 'all' to normalize by all values. Defaults to 'true'.

    Returns:
        pd.DataFrame: Confusion matrix. The column header represents the predicted labels while row header represents the actual labels.
    """
    y_true_ww = infer_feature_types(y_true)
    y_true_np = _convert_ww_series_to_np_array(y_true_ww)
    y_predicted = infer_feature_types(y_predicted)
    y_predicted = y_predicted.to_numpy()
    labels = unique_labels(y_true_np, y_predicted)
    conf_mat = sklearn_confusion_matrix(y_true_np, y_predicted)
    conf_mat = pd.DataFrame(conf_mat, index=labels, columns=labels)
    if normalize_method is not None:
        return normalize_confusion_matrix(conf_mat, normalize_method=normalize_method)
    return conf_mat


def normalize_confusion_matrix(conf_mat, normalize_method="true"):
    """Normalizes a confusion matrix.

    Args:
        conf_mat (pd.DataFrame or np.ndarray): Confusion matrix to normalize.
        normalize_method ({'true', 'pred', 'all'}): Normalization method. Supported options are: 'true' to normalize by row, 'pred' to normalize by column, or 'all' to normalize by all values. Defaults to 'true'.

    Returns:
        pd.DataFrame: normalized version of the input confusion matrix. The column header represents the predicted labels while row header represents the actual labels.

    Raises:
        ValueError: If configuration is invalid, or if the sum of a given axis is zero and normalization by axis is specified.
    """
    conf_mat = infer_feature_types(conf_mat)
    col_names = conf_mat.columns

    conf_mat = conf_mat.to_numpy()
    with warnings.catch_warnings(record=True) as w:
        if normalize_method == "true":
            conf_mat = conf_mat.astype("float") / conf_mat.sum(axis=1)[:, np.newaxis]
        elif normalize_method == "pred":
            conf_mat = conf_mat.astype("float") / conf_mat.sum(axis=0)
        elif normalize_method == "all":
            conf_mat = conf_mat.astype("float") / conf_mat.sum().sum()
        else:
            raise ValueError(
                'Invalid value provided for "normalize_method": {}'.format(
                    normalize_method,
                ),
            )
        if w and "invalid value encountered in" in str(w[0].message):
            raise ValueError(
                "Sum of given axis is 0 and normalization is not possible. Please select another option.",
            )
    conf_mat = pd.DataFrame(conf_mat, index=col_names, columns=col_names)
    return conf_mat


def graph_confusion_matrix(
    y_true,
    y_pred,
    normalize_method="true",
    title_addition=None,
):
    """Generate and display a confusion matrix plot.

    If `normalize_method` is set, hover text will show raw count, otherwise hover text will show count normalized with method 'true'.

    Args:
        y_true (pd.Series or np.ndarray): True binary labels.
        y_pred (pd.Series or np.ndarray): Predictions from a binary classifier.
        normalize_method ({'true', 'pred', 'all', None}): Normalization method to use, if not None. Supported options are: 'true' to normalize by row, 'pred' to normalize by column, or 'all' to normalize by all values. Defaults to 'true'.
        title_addition (str): If not None, append to plot title. Defaults to None.

    Returns:
        plotly.Figure representing the confusion matrix plot generated.
    """
    _go = import_or_raise(
        "plotly.graph_objects",
        error_msg="Cannot find dependency plotly.graph_objects",
    )
    _ff = import_or_raise(
        "plotly.figure_factory",
        error_msg="Cannot find dependency plotly.figure_factory",
    )
    if jupyter_check():
        import_or_raise("ipywidgets", warning=True)

    conf_mat = confusion_matrix(y_true, y_pred, normalize_method=None)
    conf_mat_normalized = confusion_matrix(
        y_true,
        y_pred,
        normalize_method=normalize_method or "true",
    )
    labels = conf_mat.columns.tolist()

    title = "Confusion matrix{}{}".format(
        "" if title_addition is None else (" " + title_addition),
        ""
        if normalize_method is None
        else (', normalized using method "' + normalize_method + '"'),
    )
    z_data, custom_data = (
        (conf_mat, conf_mat_normalized)
        if normalize_method is None
        else (conf_mat_normalized, conf_mat)
    )
    z_data = z_data.to_numpy()
    z_text = [["{:.3f}".format(y) for y in x] for x in z_data]
    primary_heading, secondary_heading = (
        ("Raw", "Normalized") if normalize_method is None else ("Normalized", "Raw")
    )
    hover_text = (
        "<br><b>"
        + primary_heading
        + " Count</b>: %{z}<br><b>"
        + secondary_heading
        + " Count</b>: %{customdata} <br>"
    )
    # the "<extra> tags at the end are necessary to remove unwanted trace info
    hover_template = (
        "<b>True</b>: %{y}<br><b>Predicted</b>: %{x}" + hover_text + "<extra></extra>"
    )
    layout = _go.Layout(
        title={"text": title},
        xaxis={"title": "Predicted Label", "type": "category", "tickvals": labels},
        yaxis={"title": "True Label", "type": "category", "tickvals": labels},
    )
    fig = _ff.create_annotated_heatmap(
        z_data,
        x=labels,
        y=labels,
        annotation_text=z_text,
        customdata=custom_data,
        hovertemplate=hover_template,
        colorscale="Blues",
        showscale=True,
    )
    fig.update_layout(layout)
    # put xaxis text on bottom to not overlap with title
    fig["layout"]["xaxis"].update(side="bottom")
    # plotly Heatmap y axis defaults to the reverse of what we want: https://community.plotly.com/t/heatmap-y-axis-is-reversed-by-default-going-against-standard-convention-for-matrices/32180
    fig.update_yaxes(autorange="reversed")
    return fig


def precision_recall_curve(y_true, y_pred_proba, pos_label_idx=-1):
    """Given labels and binary classifier predicted probabilities, compute and return the data representing a precision-recall curve.

    Args:
        y_true (pd.Series or np.ndarray): True binary labels.
        y_pred_proba (pd.Series or np.ndarray): Predictions from a binary classifier, before thresholding has been applied. Note this should be the predicted probability for the "true" label.
        pos_label_idx (int): the column index corresponding to the positive class. If predicted probabilities are two-dimensional, this will be used to access the probabilities for the positive class.

    Returns:
        list: Dictionary containing metrics used to generate a precision-recall plot, with the following keys:

                  * `precision`: Precision values.
                  * `recall`: Recall values.
                  * `thresholds`: Threshold values used to produce the precision and recall.
                  * `auc_score`: The area under the ROC curve.

    Raises:
        NoPositiveLabelException: If predicted probabilities do not contain a column at the specified label.
    """
    y_true = infer_feature_types(y_true)
    y_pred_proba = infer_feature_types(y_pred_proba)

    if isinstance(y_pred_proba, pd.DataFrame):
        y_pred_proba_shape = y_pred_proba.shape
        try:
            y_pred_proba = y_pred_proba.iloc[:, pos_label_idx]
        except IndexError:
            raise NoPositiveLabelException(
                f"Predicted probabilities of shape {y_pred_proba_shape} don't contain a column at index {pos_label_idx}",
            )

    precision, recall, thresholds = sklearn_precision_recall_curve(y_true, y_pred_proba)
    auc_score = sklearn_auc(recall, precision)
    return {
        "precision": precision,
        "recall": recall,
        "thresholds": thresholds,
        "auc_score": auc_score,
    }


def graph_precision_recall_curve(y_true, y_pred_proba, title_addition=None):
    """Generate and display a precision-recall plot.

    Args:
        y_true (pd.Series or np.ndarray): True binary labels.
        y_pred_proba (pd.Series or np.ndarray): Predictions from a binary classifier, before thresholding has been applied. Note this should be the predicted probability for the "true" label.
        title_addition (str or None): If not None, append to plot title. Defaults to None.

    Returns:
        plotly.Figure representing the precision-recall plot generated
    """
    _go = import_or_raise(
        "plotly.graph_objects",
        error_msg="Cannot find dependency plotly.graph_objects",
    )
    if jupyter_check():
        import_or_raise("ipywidgets", warning=True)
    precision_recall_curve_data = precision_recall_curve(y_true, y_pred_proba)
    title = "Precision-Recall{}".format(
        "" if title_addition is None else (" " + title_addition),
    )
    layout = _go.Layout(
        title={"text": title},
        xaxis={"title": "Recall", "range": [-0.05, 1.05]},
        yaxis={"title": "Precision", "range": [-0.05, 1.05]},
    )
    data = []
    data.append(
        _go.Scatter(
            x=precision_recall_curve_data["recall"],
            y=precision_recall_curve_data["precision"],
            name="Precision-Recall (AUC {:06f})".format(
                precision_recall_curve_data["auc_score"],
            ),
            line=dict(width=3),
        ),
    )
    return _go.Figure(layout=layout, data=data)


def roc_curve(y_true, y_pred_proba):
    """Given labels and classifier predicted probabilities, compute and return the data representing a Receiver Operating Characteristic (ROC) curve. Works with binary or multiclass problems.

    Args:
        y_true (pd.Series or np.ndarray): True labels.
        y_pred_proba (pd.Series or np.ndarray): Predictions from a classifier, before thresholding has been applied.

    Returns:
        list(dict): A list of dictionaries (with one for each class) is returned. Binary classification problems return a list with one dictionary.
            Each dictionary contains metrics used to generate an ROC plot with the following keys:
                  * `fpr_rate`: False positive rate.
                  * `tpr_rate`: True positive rate.
                  * `threshold`: Threshold values used to produce each pair of true/false positive rates.
                  * `auc_score`: The area under the ROC curve.
    """
    y_true_ww = infer_feature_types(y_true)
    y_true_np = _convert_ww_series_to_np_array(y_true_ww)
    y_pred_proba = infer_feature_types(y_pred_proba).to_numpy()

    if len(y_pred_proba.shape) == 1:
        y_pred_proba = y_pred_proba.reshape(-1, 1)
    if y_pred_proba.shape[1] == 2:
        y_pred_proba = y_pred_proba[:, 1].reshape(-1, 1)
    nan_indices = np.logical_or(pd.isna(y_true_np), np.isnan(y_pred_proba).any(axis=1))
    y_true_np = y_true_np[~nan_indices]
    y_pred_proba = y_pred_proba[~nan_indices]

    lb = LabelBinarizer()
    lb.fit(np.unique(y_true_np))
    y_one_hot_true = lb.transform(y_true_np)
    n_classes = y_one_hot_true.shape[1]

    curve_data = []
    for i in range(n_classes):
        fpr_rates, tpr_rates, thresholds = sklearn_roc_curve(
            y_one_hot_true[:, i],
            y_pred_proba[:, i],
        )
        auc_score = sklearn_auc(fpr_rates, tpr_rates)
        curve_data.append(
            {
                "fpr_rates": fpr_rates,
                "tpr_rates": tpr_rates,
                "thresholds": thresholds,
                "auc_score": auc_score,
            },
        )

    return curve_data


def graph_roc_curve(y_true, y_pred_proba, custom_class_names=None, title_addition=None):
    """Generate and display a Receiver Operating Characteristic (ROC) plot for binary and multiclass classification problems.

    Args:
        y_true (pd.Series or np.ndarray): True labels.
        y_pred_proba (pd.Series or np.ndarray): Predictions from a classifier, before thresholding has been applied. Note this should a one dimensional array with the predicted probability for the "true" label in the binary case.
        custom_class_names (list or None): If not None, custom labels for classes. Defaults to None.
        title_addition (str or None): if not None, append to plot title. Defaults to None.

    Returns:
        plotly.Figure representing the ROC plot generated

    Raises:
        ValueError: If the number of custom class names does not match number of classes in the input data.
    """
    _go = import_or_raise(
        "plotly.graph_objects",
        error_msg="Cannot find dependency plotly.graph_objects",
    )
    if jupyter_check():
        import_or_raise("ipywidgets", warning=True)

    title = "Receiver Operating Characteristic{}".format(
        "" if title_addition is None else (" " + title_addition),
    )
    layout = _go.Layout(
        title={"text": title},
        xaxis={"title": "False Positive Rate", "range": [-0.05, 1.05]},
        yaxis={"title": "True Positive Rate", "range": [-0.05, 1.05]},
    )

    all_curve_data = roc_curve(y_true, y_pred_proba)
    graph_data = []

    n_classes = len(all_curve_data)

    if custom_class_names and len(custom_class_names) != n_classes:
        raise ValueError(
            "Number of custom class names does not match number of classes",
        )

    for i in range(n_classes):
        roc_curve_data = all_curve_data[i]
        name = i + 1 if custom_class_names is None else custom_class_names[i]
        graph_data.append(
            _go.Scatter(
                x=roc_curve_data["fpr_rates"],
                y=roc_curve_data["tpr_rates"],
                hovertemplate="(False Postive Rate: %{x}, True Positive Rate: %{y})<br>"
                + "Threshold: %{text}",
                name=f"Class {name} (AUC {roc_curve_data['auc_score']:.06f})",
                text=roc_curve_data["thresholds"],
                line=dict(width=3),
            ),
        )
    graph_data.append(
        _go.Scatter(
            x=[0, 1],
            y=[0, 1],
            name="Trivial Model (AUC 0.5)",
            line=dict(dash="dash"),
        ),
    )
    return _go.Figure(layout=layout, data=graph_data)
