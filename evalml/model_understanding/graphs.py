import copy
import os
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import woodwork as ww
from sklearn.exceptions import NotFittedError
from sklearn.inspection import partial_dependence as sk_partial_dependence
from sklearn.inspection import \
    permutation_importance as sk_permutation_importance
from sklearn.metrics import auc as sklearn_auc
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from sklearn.metrics import \
    precision_recall_curve as sklearn_precision_recall_curve
from sklearn.metrics import roc_curve as sklearn_roc_curve
from sklearn.preprocessing import LabelBinarizer
from sklearn.tree import export_graphviz
from sklearn.utils.multiclass import unique_labels

import evalml
from evalml.exceptions import NullsInColumnWarning
from evalml.model_family import ModelFamily
from evalml.objectives.utils import get_objective
from evalml.problem_types import ProblemTypes
from evalml.utils import (
    _convert_to_woodwork_structure,
    _convert_woodwork_types_wrapper,
    import_or_raise,
    jupyter_check
)


def confusion_matrix(y_true, y_predicted, normalize_method='true'):
    """Confusion matrix for binary and multiclass classification.

    Arguments:
        y_true (ww.DataColumn, pd.Series or np.ndarray): True binary labels.
        y_pred (ww.DataColumn, pd.Series or np.ndarray): Predictions from a binary classifier.
        normalize_method ({'true', 'pred', 'all', None}): Normalization method to use, if not None. Supported options are: 'true' to normalize by row, 'pred' to normalize by column, or 'all' to normalize by all values. Defaults to 'true'.

    Returns:
        pd.DataFrame: Confusion matrix. The column header represents the predicted labels while row header represents the actual labels.
    """
    y_true = _convert_to_woodwork_structure(y_true)
    y_predicted = _convert_to_woodwork_structure(y_predicted)
    y_true = _convert_woodwork_types_wrapper(y_true.to_series()).to_numpy()
    y_predicted = _convert_woodwork_types_wrapper(y_predicted.to_series()).to_numpy()
    labels = unique_labels(y_true, y_predicted)
    conf_mat = sklearn_confusion_matrix(y_true, y_predicted)
    conf_mat = pd.DataFrame(conf_mat, index=labels, columns=labels)
    if normalize_method is not None:
        return normalize_confusion_matrix(conf_mat, normalize_method=normalize_method)
    return conf_mat


def normalize_confusion_matrix(conf_mat, normalize_method='true'):
    """Normalizes a confusion matrix.

    Arguments:
        conf_mat (ww.DataTable, pd.DataFrame or np.ndarray): Confusion matrix to normalize.
        normalize_method ({'true', 'pred', 'all'}): Normalization method. Supported options are: 'true' to normalize by row, 'pred' to normalize by column, or 'all' to normalize by all values. Defaults to 'true'.

    Returns:
        pd.DataFrame: normalized version of the input confusion matrix. The column header represents the predicted labels while row header represents the actual labels.
    """
    conf_mat = _convert_to_woodwork_structure(conf_mat)
    conf_mat = _convert_woodwork_types_wrapper(conf_mat.to_dataframe())
    col_names = conf_mat.columns

    conf_mat = conf_mat.to_numpy()
    with warnings.catch_warnings(record=True) as w:
        if normalize_method == 'true':
            conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
        elif normalize_method == 'pred':
            conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=0)
        elif normalize_method == 'all':
            conf_mat = conf_mat.astype('float') / conf_mat.sum().sum()
        else:
            raise ValueError('Invalid value provided for "normalize_method": {}'.format(normalize_method))
        if w and "invalid value encountered in" in str(w[0].message):
            raise ValueError("Sum of given axis is 0 and normalization is not possible. Please select another option.")
    conf_mat = pd.DataFrame(conf_mat, index=col_names, columns=col_names)
    return conf_mat


def graph_confusion_matrix(y_true, y_pred, normalize_method='true', title_addition=None):
    """Generate and display a confusion matrix plot.

    If `normalize_method` is set, hover text will show raw count, otherwise hover text will show count normalized with method 'true'.

    Arguments:
        y_true (ww.DataColumn, pd.Series or np.ndarray): True binary labels.
        y_pred (ww.DataColumn, pd.Series or np.ndarray): Predictions from a binary classifier.
        normalize_method ({'true', 'pred', 'all', None}): Normalization method to use, if not None. Supported options are: 'true' to normalize by row, 'pred' to normalize by column, or 'all' to normalize by all values. Defaults to 'true'.
        title_addition (str or None): if not None, append to plot title. Defaults to None.

    Returns:
        plotly.Figure representing the confusion matrix plot generated
    """
    _go = import_or_raise("plotly.graph_objects", error_msg="Cannot find dependency plotly.graph_objects")
    if jupyter_check():
        import_or_raise("ipywidgets", warning=True)

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


def precision_recall_curve(y_true, y_pred_proba):
    """
    Given labels and binary classifier predicted probabilities, compute and return the data representing a precision-recall curve.

    Arguments:
        y_true (ww.DataColumn, pd.Series or np.ndarray): True binary labels.
        y_pred_proba (ww.DataColumn, pd.Series or np.ndarray): Predictions from a binary classifier, before thresholding has been applied. Note this should be the predicted probability for the "true" label.

    Returns:
        list: Dictionary containing metrics used to generate a precision-recall plot, with the following keys:

                  * `precision`: Precision values.
                  * `recall`: Recall values.
                  * `thresholds`: Threshold values used to produce the precision and recall.
                  * `auc_score`: The area under the ROC curve.
    """
    y_true = _convert_to_woodwork_structure(y_true)
    y_pred_proba = _convert_to_woodwork_structure(y_pred_proba)
    y_true = _convert_woodwork_types_wrapper(y_true.to_series())
    y_pred_proba = _convert_woodwork_types_wrapper(y_pred_proba.to_series())

    precision, recall, thresholds = sklearn_precision_recall_curve(y_true, y_pred_proba)
    auc_score = sklearn_auc(recall, precision)
    return {'precision': precision,
            'recall': recall,
            'thresholds': thresholds,
            'auc_score': auc_score}


def graph_precision_recall_curve(y_true, y_pred_proba, title_addition=None):
    """Generate and display a precision-recall plot.

    Arguments:
        y_true (ww.DataColumn, pd.Series or np.ndarray): True binary labels.
        y_pred_proba (ww.DataColumn, pd.Series or np.ndarray): Predictions from a binary classifier, before thresholding has been applied. Note this should be the predicted probability for the "true" label.
        title_addition (str or None): If not None, append to plot title. Default None.

    Returns:
        plotly.Figure representing the precision-recall plot generated
    """
    _go = import_or_raise("plotly.graph_objects", error_msg="Cannot find dependency plotly.graph_objects")
    if jupyter_check():
        import_or_raise("ipywidgets", warning=True)
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
    Given labels and classifier predicted probabilities, compute and return the data representing a Receiver Operating Characteristic (ROC) curve. Works with binary or multiclass problems.

    Arguments:
        y_true (ww.DataColumn, pd.Series or np.ndarray): True labels.
        y_pred_proba (ww.DataColumn, pd.Series or np.ndarray): Predictions from a classifier, before thresholding has been applied.

    Returns:
        list(dict): A list of dictionaries (with one for each class) is returned. Binary classification problems return a list with one dictionary.
            Each dictionary contains metrics used to generate an ROC plot with the following keys:
                  * `fpr_rate`: False positive rate.
                  * `tpr_rate`: True positive rate.
                  * `threshold`: Threshold values used to produce each pair of true/false positive rates.
                  * `auc_score`: The area under the ROC curve.
    """
    y_true = _convert_to_woodwork_structure(y_true)
    y_pred_proba = _convert_to_woodwork_structure(y_pred_proba)
    if isinstance(y_pred_proba, ww.DataTable):
        y_pred_proba = _convert_woodwork_types_wrapper(y_pred_proba.to_dataframe()).to_numpy()
    else:
        y_pred_proba = _convert_woodwork_types_wrapper(y_pred_proba.to_series()).to_numpy()
    y_true = _convert_woodwork_types_wrapper(y_true.to_series()).to_numpy()

    if len(y_pred_proba.shape) == 1:
        y_pred_proba = y_pred_proba.reshape(-1, 1)
    if y_pred_proba.shape[1] == 2:
        y_pred_proba = y_pred_proba[:, 1].reshape(-1, 1)
    nan_indices = np.logical_or(pd.isna(y_true), np.isnan(y_pred_proba).any(axis=1))
    y_true = y_true[~nan_indices]
    y_pred_proba = y_pred_proba[~nan_indices]

    lb = LabelBinarizer()
    lb.fit(np.unique(y_true))
    y_one_hot_true = lb.transform(y_true)
    n_classes = y_one_hot_true.shape[1]

    curve_data = []
    for i in range(n_classes):
        fpr_rates, tpr_rates, thresholds = sklearn_roc_curve(y_one_hot_true[:, i], y_pred_proba[:, i])
        auc_score = sklearn_auc(fpr_rates, tpr_rates)
        curve_data.append({'fpr_rates': fpr_rates,
                           'tpr_rates': tpr_rates,
                           'thresholds': thresholds,
                           'auc_score': auc_score})

    return curve_data


def graph_roc_curve(y_true, y_pred_proba, custom_class_names=None, title_addition=None):
    """Generate and display a Receiver Operating Characteristic (ROC) plot for binary and multiclass classification problems.

    Arguments:
        y_true (ww.DataColumn, pd.Series or np.ndarray): True labels.
        y_pred_proba (ww.DataColumn, pd.Series or np.ndarray): Predictions from a classifier, before thresholding has been applied. Note this should a one dimensional array with the predicted probability for the "true" label in the binary case.
        custom_class_labels (list or None): If not None, custom labels for classes. Default None.
        title_addition (str or None): if not None, append to plot title. Default None.

    Returns:
        plotly.Figure representing the ROC plot generated
    """
    _go = import_or_raise("plotly.graph_objects", error_msg="Cannot find dependency plotly.graph_objects")
    if jupyter_check():
        import_or_raise("ipywidgets", warning=True)

    title = 'Receiver Operating Characteristic{}'.format('' if title_addition is None else (' ' + title_addition))
    layout = _go.Layout(title={'text': title},
                        xaxis={'title': 'False Positive Rate', 'range': [-0.05, 1.05]},
                        yaxis={'title': 'True Positive Rate', 'range': [-0.05, 1.05]})

    all_curve_data = roc_curve(y_true, y_pred_proba)
    graph_data = []

    n_classes = len(all_curve_data)

    if custom_class_names and len(custom_class_names) != n_classes:
        raise ValueError('Number of custom class names does not match number of classes')

    for i in range(n_classes):
        roc_curve_data = all_curve_data[i]
        name = i + 1 if custom_class_names is None else custom_class_names[i]
        graph_data.append(_go.Scatter(x=roc_curve_data['fpr_rates'], y=roc_curve_data['tpr_rates'],
                                      hovertemplate="(False Postive Rate: %{x}, True Positive Rate: %{y})<br>" + "Threshold: %{text}",
                                      name=f"Class {name} (AUC {roc_curve_data['auc_score']:.06f})",
                                      text=roc_curve_data["thresholds"],
                                      line=dict(width=3)))
    graph_data.append(_go.Scatter(x=[0, 1], y=[0, 1],
                                  name='Trivial Model (AUC 0.5)',
                                  line=dict(dash='dash')))
    return _go.Figure(layout=layout, data=graph_data)


def calculate_permutation_importance(pipeline, X, y, objective, n_repeats=5, n_jobs=None, random_state=0):
    """Calculates permutation importance for features.

    Arguments:
        pipeline (PipelineBase or subclass): Fitted pipeline
        X (ww.DataTable, pd.DataFrame): The input data used to score and compute permutation importance
        y (ww.DataColumn, pd.Series): The target data
        objective (str, ObjectiveBase): Objective to score on
        n_repeats (int): Number of times to permute a feature. Defaults to 5.
        n_jobs (int or None): Non-negative integer describing level of parallelism used for pipelines.
            None and 1 are equivalent. If set to -1, all CPUs are used. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.
        random_state (int, np.random.RandomState): The random seed/state. Defaults to 0.

    Returns:
        Mean feature importance scores over 5 shuffles.
    """
    X = _convert_to_woodwork_structure(X)
    y = _convert_to_woodwork_structure(y)
    X = _convert_woodwork_types_wrapper(X.to_dataframe())
    y = _convert_woodwork_types_wrapper(y.to_series())

    objective = get_objective(objective, return_instance=True)
    if not objective.is_defined_for_problem_type(pipeline.problem_type):
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


def graph_permutation_importance(pipeline, X, y, objective, importance_threshold=0):
    """Generate a bar graph of the pipeline's permutation importance.

    Arguments:
        pipeline (PipelineBase or subclass): Fitted pipeline
        X (ww.DataTable, pd.DataFrame): The input data used to score and compute permutation importance
        y (ww.DataColumn, pd.Series): The target data
        objective (str, ObjectiveBase): Objective to score on
        importance_threshold (float, optional): If provided, graph features with a permutation importance whose absolute value is larger than importance_threshold. Defaults to zero.

    Returns:
        plotly.Figure, a bar graph showing features and their respective permutation importance.
    """
    go = import_or_raise("plotly.graph_objects", error_msg="Cannot find dependency plotly.graph_objects")
    if jupyter_check():
        import_or_raise("ipywidgets", warning=True)

    perm_importance = calculate_permutation_importance(pipeline, X, y, objective)
    perm_importance['importance'] = perm_importance['importance']

    if importance_threshold < 0:
        raise ValueError(f'Provided importance threshold of {importance_threshold} must be greater than or equal to 0')
    # Remove features with close to zero importance
    perm_importance = perm_importance[abs(perm_importance['importance']) >= importance_threshold]
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


def binary_objective_vs_threshold(pipeline, X, y, objective, steps=100):
    """Computes objective score as a function of potential binary classification
        decision thresholds for a fitted binary classification pipeline.

    Arguments:
        pipeline (BinaryClassificationPipeline obj): Fitted binary classification pipeline
        X (ww.DataTable, pd.DataFrame): The input data used to compute objective score
        y (ww.DataColumn, pd.Series): The target labels
        objective (ObjectiveBase obj, str): Objective used to score
        steps (int): Number of intervals to divide and calculate objective score at

    Returns:
        pd.DataFrame: DataFrame with thresholds and the corresponding objective score calculated at each threshold

    """
    objective = get_objective(objective, return_instance=True)
    if not objective.is_defined_for_problem_type(ProblemTypes.BINARY):
        raise ValueError("`binary_objective_vs_threshold` can only be calculated for binary classification objectives")
    if objective.score_needs_proba:
        raise ValueError("Objective `score_needs_proba` must be False")

    pipeline_tmp = copy.copy(pipeline)
    thresholds = np.linspace(0, 1, steps + 1)
    costs = []
    for threshold in thresholds:
        pipeline_tmp.threshold = threshold
        scores = pipeline_tmp.score(X, y, [objective])
        costs.append(scores[objective.name])
    df = pd.DataFrame({"threshold": thresholds, "score": costs})
    return df


def graph_binary_objective_vs_threshold(pipeline, X, y, objective, steps=100):
    """Generates a plot graphing objective score vs. decision thresholds for a fitted binary classification pipeline.

    Arguments:
        pipeline (PipelineBase or subclass): Fitted pipeline
        X (ww.DataTable, pd.DataFrame): The input data used to score and compute scores
        y (ww.DataColumn, pd.Series): The target labels
        objective (ObjectiveBase obj, str): Objective used to score, shown on the y-axis of the graph
        steps (int): Number of intervals to divide and calculate objective score at

    Returns:
        plotly.Figure representing the objective score vs. threshold graph generated

    """
    _go = import_or_raise("plotly.graph_objects", error_msg="Cannot find dependency plotly.graph_objects")
    if jupyter_check():
        import_or_raise("ipywidgets", warning=True)

    objective = get_objective(objective, return_instance=True)
    df = binary_objective_vs_threshold(pipeline, X, y, objective, steps)
    title = f'{objective.name} Scores vs. Thresholds'
    layout = _go.Layout(title={'text': title},
                        xaxis={'title': 'Threshold', 'range': _calculate_axis_range(df['threshold'])},
                        yaxis={'title': f"{objective.name} Scores vs. Binary Classification Decision Threshold", 'range': _calculate_axis_range(df['score'])})
    data = []
    data.append(_go.Scatter(x=df['threshold'],
                            y=df['score'],
                            line=dict(width=3)))
    return _go.Figure(layout=layout, data=data)


def partial_dependence(pipeline, X, feature, grid_resolution=100):
    """Calculates partial dependence.

    Arguments:
        pipeline (PipelineBase or subclass): Fitted pipeline
        X (ww.DataTable, pd.DataFrame, np.ndarray): The input data used to generate a grid of values
            for feature where partial dependence will be calculated at
        feature (int, string): The target features for which to create the partial dependence plot for.
            If feature is an int, it must be the index of the feature to use.
            If feature is a string, it must be a valid column name in X.

    Returns:
        pd.DataFrame: DataFrame with averaged predictions for all points in the grid averaged
            over all samples of X and the values used to calculate those predictions. The dataframe will
            contain two columns: "feature_values" (grid points at which the partial dependence was calculated) and
            "partial_dependence" (the partial dependence at that feature value). For classification problems, there
            will be a third column called "class_label" (the class label for which the partial
            dependence was calculated). For binary classification, the partial dependence is only calculated for the
            "positive" class.

    """
    X = _convert_to_woodwork_structure(X)
    X = _convert_woodwork_types_wrapper(X.to_dataframe())

    if not pipeline._is_fitted:
        raise ValueError("Pipeline to calculate partial dependence for must be fitted")
    if pipeline.model_family == ModelFamily.BASELINE:
        raise ValueError("Partial dependence plots are not supported for Baseline pipelines")
    if isinstance(pipeline, evalml.pipelines.ClassificationPipeline):
        pipeline._estimator_type = "classifier"
    elif isinstance(pipeline, evalml.pipelines.RegressionPipeline):
        pipeline._estimator_type = "regressor"
    pipeline.feature_importances_ = pipeline.feature_importance
    if ((isinstance(feature, int) and X.iloc[:, feature].isnull().sum()) or (isinstance(feature, str) and X[feature].isnull().sum())):
        warnings.warn("There are null values in the features, which will cause NaN values in the partial dependence output. Fill in these values to remove the NaN values.", NullsInColumnWarning)
    try:
        avg_pred, values = sk_partial_dependence(pipeline, X=X, features=[feature], grid_resolution=grid_resolution)
    finally:
        # Delete scikit-learn attributes that were temporarily set
        del pipeline._estimator_type
        del pipeline.feature_importances_
    classes = None
    if isinstance(pipeline, evalml.pipelines.BinaryClassificationPipeline):
        classes = [pipeline.classes_[1]]
    elif isinstance(pipeline, evalml.pipelines.MulticlassClassificationPipeline):
        classes = pipeline.classes_

    data = pd.DataFrame({"feature_values": np.tile(values[0], avg_pred.shape[0]),
                         "partial_dependence": np.concatenate([pred for pred in avg_pred])})
    if classes is not None:
        data['class_label'] = np.repeat(classes, len(values[0]))

    return data


def graph_partial_dependence(pipeline, X, feature, class_label=None, grid_resolution=100):
    """Create an one-way partial dependence plot.

    Arguments:
        pipeline (PipelineBase or subclass): Fitted pipeline
        X (ww.DataTable, pd.DataFrame, np.ndarray): The input data used to generate a grid of values
            for feature where partial dependence will be calculated at
        feature (int, string): The target feature for which to create the partial dependence plot for.
            If feature is an int, it must be the index of the feature to use.
            If feature is a string, it must be a valid column name in X.
        class_label (string, optional): Name of class to plot for multiclass problems. If None, will plot
            the partial dependence for each class. This argument does not change behavior for regression or binary
            classification pipelines. For binary classification, the partial dependence for the positive label will
            always be displayed. Defaults to None.

    Returns:
        pd.DataFrame: pd.DataFrame with averaged predictions for all points in the grid averaged
            over all samples of X and the values used to calculate those predictions.

    """
    _go = import_or_raise("plotly.graph_objects", error_msg="Cannot find dependency plotly.graph_objects")
    if jupyter_check():
        import_or_raise("ipywidgets", warning=True)
    if isinstance(pipeline, evalml.pipelines.MulticlassClassificationPipeline) and class_label is not None:
        if class_label not in pipeline.classes_:
            msg = f"Class {class_label} is not one of the classes the pipeline was fit on: {', '.join(list(pipeline.classes_))}"
            raise ValueError(msg)

    part_dep = partial_dependence(pipeline, X, feature=feature, grid_resolution=grid_resolution)
    feature_name = str(feature)
    title = f"Partial Dependence of '{feature_name}'"
    layout = _go.Layout(title={'text': title},
                        xaxis={'title': f'{feature_name}'},
                        yaxis={'title': 'Partial Dependence'},
                        showlegend=False)
    if isinstance(pipeline, evalml.pipelines.MulticlassClassificationPipeline):
        class_labels = [class_label] if class_label is not None else pipeline.classes_
        _subplots = import_or_raise("plotly.subplots", error_msg="Cannot find dependency plotly.graph_objects")

        # If the user passes in a value for class_label, we want to create a 1 x 1 subplot or else there would
        # be an empty column in the plot and it would look awkward
        rows, cols = ((len(class_labels) + 1) // 2, 2) if len(class_labels) > 1 else (1, len(class_labels))

        # Don't specify share_xaxis and share_yaxis so that we get tickmarks in each subplot
        fig = _subplots.make_subplots(rows=rows, cols=cols, subplot_titles=class_labels)
        for i, label in enumerate(class_labels):

            # Plotly trace indexing begins at 1 so we add 1 to i
            fig.add_trace(_go.Scatter(x=part_dep.loc[part_dep.class_label == label, 'feature_values'],
                                      y=part_dep.loc[part_dep.class_label == label, 'partial_dependence'],
                                      line=dict(width=3),
                                      name=label),
                          row=(i + 2) // 2, col=(i % 2) + 1)
        fig.update_layout(layout)
        fig.update_xaxes(title=f'{feature_name}', range=_calculate_axis_range(part_dep['feature_values']))
        fig.update_yaxes(range=_calculate_axis_range(part_dep['partial_dependence']))
    else:
        trace = _go.Scatter(x=part_dep['feature_values'],
                            y=part_dep['partial_dependence'],
                            name='Partial Dependence',
                            line=dict(width=3))
        fig = _go.Figure(layout=layout, data=[trace])

    return fig


def _calculate_axis_range(arr):
    """Helper method to help calculate the appropriate range for an axis based on the data to graph."""
    max_value = arr.max()
    min_value = arr.min()
    margins = abs(max_value - min_value) * 0.05
    return [min_value - margins, max_value + margins]


def get_prediction_vs_actual_data(y_true, y_pred, outlier_threshold=None):
    """Combines y_true and y_pred into a single dataframe and adds a column for outliers. Used in `graph_prediction_vs_actual()`.

    Arguments:
        y_true (pd.Series, ww.DataColumn, or np.ndarray): The real target values of the data
        y_pred (pd.Series, ww.DataColumn, or np.ndarray): The predicted values outputted by the regression model.
        outlier_threshold (int, float): A positive threshold for what is considered an outlier value. This value is compared to the absolute difference
                                 between each value of y_true and y_pred. Values within this threshold will be blue, otherwise they will be yellow.
                                 Defaults to None

    Returns:
        pd.DataFrame with the following columns:
                * `prediction`: Predicted values from regression model.
                * `actual`: Real target values.
                * `outlier`: Colors indicating which values are in the threshold for what is considered an outlier value.

    """
    if outlier_threshold and outlier_threshold <= 0:
        raise ValueError(f"Threshold must be positive! Provided threshold is {outlier_threshold}")

    y_true = _convert_to_woodwork_structure(y_true)
    y_true = _convert_woodwork_types_wrapper(y_true.to_series())
    y_pred = _convert_to_woodwork_structure(y_pred)
    y_pred = _convert_woodwork_types_wrapper(y_pred.to_series())

    predictions = y_pred.reset_index(drop=True)
    actual = y_true.reset_index(drop=True)
    data = pd.concat([pd.Series(predictions),
                      pd.Series(actual)], axis=1)
    data.columns = ['prediction', 'actual']
    if outlier_threshold:
        data['outlier'] = np.where((abs(data['prediction'] - data['actual']) >= outlier_threshold), "#ffff00", "#0000ff")
    else:
        data['outlier'] = '#0000ff'
    return data


def graph_prediction_vs_actual(y_true, y_pred, outlier_threshold=None):
    """Generate a scatter plot comparing the true and predicted values. Used for regression plotting

    Arguments:
        y_true (ww.DataColumn, pd.Series): The real target values of the data
        y_pred (ww.DataColumn, pd.Series): The predicted values outputted by the regression model.
        outlier_threshold (int, float): A positive threshold for what is considered an outlier value. This value is compared to the absolute difference
                                 between each value of y_true and y_pred. Values within this threshold will be blue, otherwise they will be yellow.
                                 Defaults to None

    Returns:
        plotly.Figure representing the predicted vs. actual values graph

    """
    _go = import_or_raise("plotly.graph_objects", error_msg="Cannot find dependency plotly.graph_objects")
    if jupyter_check():
        import_or_raise("ipywidgets", warning=True)

    if outlier_threshold and outlier_threshold <= 0:
        raise ValueError(f"Threshold must be positive! Provided threshold is {outlier_threshold}")

    df = get_prediction_vs_actual_data(y_true, y_pred, outlier_threshold)
    data = []

    x_axis = _calculate_axis_range(df['prediction'])
    y_axis = _calculate_axis_range(df['actual'])
    x_y_line = [min(x_axis[0], y_axis[0]), max(x_axis[1], y_axis[1])]
    data.append(_go.Scatter(x=x_y_line, y=x_y_line, name="y = x line", line_color='grey'))

    title = 'Predicted vs Actual Values Scatter Plot'
    layout = _go.Layout(title={'text': title},
                        xaxis={'title': 'Prediction', 'range': x_y_line},
                        yaxis={'title': 'Actual', 'range': x_y_line})

    for color, outlier_group in df.groupby('outlier'):
        if outlier_threshold:
            name = "< outlier_threshold" if color == "#0000ff" else ">= outlier_threshold"
        else:
            name = "Values"
        data.append(_go.Scatter(x=outlier_group['prediction'],
                                y=outlier_group['actual'],
                                mode='markers',
                                marker=_go.scatter.Marker(color=color),
                                name=name))
    return _go.Figure(layout=layout, data=data)


def _tree_parse(est, feature_names):
    children_left = est.tree_.children_left
    children_right = est.tree_.children_right
    features = est.tree_.feature
    thresholds = est.tree_.threshold
    values = est.tree_.value

    def recurse(i):
        if children_left[i] == children_right[i]:
            return {'Value': values[i]}
        return OrderedDict({
            'Feature': feature_names[features[i]],
            'Threshold': thresholds[i],
            'Value': values[i],
            'Left_Child': recurse(children_left[i]),
            'Right_Child': recurse(children_right[i])
        })

    return recurse(0)


def decision_tree_data_from_estimator(estimator, feature_names=None):
    """Return data for a fitted tree in a restructured format

    Arguments:
        estimator (ComponentBase): A fitted DecisionTree-based estimator.
        feature_names (List): A list of feature names to replace column index values.

    Returns:
        OrderedDict: An OrderedDict of OrderedDicts describing a tree structure
    """
    if not estimator.model_family == ModelFamily.DECISION_TREE:
        raise ValueError("Tree structure reformatting is only supported for decision tree estimators")
    if not estimator._is_fitted:
        raise NotFittedError("This DecisionTree estimator is not fitted yet. Call 'fit' with appropriate arguments "
                             "before using this estimator.")
    est = estimator._component_obj

    if feature_names:
        if not isinstance(feature_names, list):
            feature_names = list(feature_names)
        if len(feature_names) != est.n_features_:
            raise ValueError("Length mismatch: Expected features has length {} but got list with length {}"
                             .format(est.n_features_, len(feature_names)))

    return _tree_parse(est, feature_names)


def decision_tree_data_from_pipeline(pipeline_):
    """Return data for a fitted pipeline with  in a restructured format

    Arguments:
        pipeline_ (PipelineBase): A pipeline with a DecisionTree-based estimator.

    Returns:
        OrderedDict: An OrderedDict of OrderedDicts describing a tree structure
    """
    if not pipeline_.model_family == ModelFamily.DECISION_TREE:
        raise ValueError("Tree structure reformatting is only supported for decision tree estimators")
    if not pipeline_._is_fitted:
        raise NotFittedError("The DecisionTree estimator associated with this pipeline is not fitted yet. Call 'fit' "
                             "with appropriate arguments before using this estimator.")
    est = pipeline_.estimator._component_obj
    feature_names = pipeline_.input_feature_names[pipeline_.estimator.name]

    return _tree_parse(est, feature_names)


def visualize_decision_tree(estimator, max_depth=None, rotate=False, filled=False, filepath=None):
    """Generate an image visualizing the decision tree

    Arguments:
        estimator (ComponentBase): A fitted DecisionTree-based estimator.
        max_depth (int, optional): The depth to which the tree should be displayed. If set to None (as by default),
        tree is fully generated.
        rotate (bool, optional): Orient tree left to right rather than top-down.
        filled (bool, optional): Paint nodes to indicate majority class for classification, extremity of values for
        regression, or purity of node for multi-output.
        filepath (str, optional): Path to where the graph should be saved. If set to None (as by default), the graph
        will not be saved.

    Returns:
        graphviz.Source: DOT object that can be directly displayed in Jupyter notebooks.
    """
    if not estimator.model_family == ModelFamily.DECISION_TREE:
        raise ValueError("Tree visualizations are only supported for decision tree estimators")
    if max_depth and (not isinstance(max_depth, int) or not max_depth >= 0):
        raise ValueError("Unknown value: '{}'. The parameter max_depth has to be a non-negative integer"
                         .format(max_depth))
    if not estimator._is_fitted:
        raise NotFittedError("This DecisionTree estimator is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

    est = estimator._component_obj

    graphviz = import_or_raise('graphviz', error_msg='Please install graphviz to visualize trees.')

    graph_format = None
    if filepath:
        # Cast to str in case a Path object was passed in
        filepath = str(filepath)
        try:
            f = open(filepath, 'w')
            f.close()
        except (IOError, FileNotFoundError):
            raise ValueError(('Specified filepath is not writeable: {}'.format(filepath)))
        path_and_name, graph_format = os.path.splitext(filepath)
        if graph_format:
            graph_format = graph_format[1:].lower()  # ignore the dot
            supported_filetypes = graphviz.backend.FORMATS
            if graph_format not in supported_filetypes:
                raise ValueError(("Unknown format '{}'. Make sure your format is one of the " +
                                  "following: {}").format(graph_format, supported_filetypes))
        else:
            graph_format = 'pdf'  # If the filepath has no extension default to pdf

    dot_data = export_graphviz(decision_tree=est, max_depth=max_depth, rotate=rotate, filled=filled)
    source_obj = graphviz.Source(source=dot_data, format=graph_format)
    if filepath:
        source_obj.render(filename=path_and_name, cleanup=True)

    return source_obj


def get_prediction_vs_actual_over_time_data(pipeline, X, y, dates):
    """Get the data needed for the prediction_vs_actual_over_time plot.

    Arguments:
        pipeline (TimeSeriesRegressionPipeline): Fitted time series regression pipeline.
        X (ww.DataTable, pd.DataFrame): Features used to generate new predictions.
        y (ww.DataColumn, pd.Series): Target values to compare predictions against.
        dates (ww.DataColumn, pd.Series): Dates corresponding to target values and predictions.

    Returns:
       pd.DataFrame
    """

    dates = _convert_to_woodwork_structure(dates)
    y = _convert_to_woodwork_structure(y)
    prediction = pipeline.predict(X, y)

    dates = _convert_woodwork_types_wrapper(dates.to_series())
    y = _convert_woodwork_types_wrapper(y.to_series())
    return pd.DataFrame({"dates": dates.reset_index(drop=True),
                         "target": y.reset_index(drop=True),
                         "prediction": prediction.reset_index(drop=True)})


def graph_prediction_vs_actual_over_time(pipeline, X, y, dates):
    """Plot the target values and predictions against time on the x-axis.

    Arguments:
        pipeline (TimeSeriesRegressionPipeline): Fitted time series regression pipeline.
        X (ww.DataTable, pd.DataFrame): Features used to generate new predictions.
        y (ww.DataColumn, pd.Series): Target values to compare predictions against.
        dates (ww.DataColumn, pd.Series): Dates corresponding to target values and predictions.

    Returns:
        plotly.Figure showing the prediction vs actual over time.
    """
    _go = import_or_raise("plotly.graph_objects", error_msg="Cannot find dependency plotly.graph_objects")

    if pipeline.problem_type != ProblemTypes.TIME_SERIES_REGRESSION:
        raise ValueError("graph_prediction_vs_actual_over_time only supports time series regression pipelines! "
                         f"Received {str(pipeline.problem_type)}.")

    data = get_prediction_vs_actual_over_time_data(pipeline, X, y, dates)

    data = [_go.Scatter(x=data["dates"], y=data["target"], mode='lines+markers', name="Target",
                        line=dict(color='#1f77b4')),
            _go.Scatter(x=data["dates"], y=data["prediction"], mode='lines+markers', name='Prediction',
                        line=dict(color='#d62728'))]
    # Let plotly pick the best date format.
    layout = _go.Layout(title={'text': "Prediction vs Target over time"},
                        xaxis={'title': 'Time'},
                        yaxis={'title': 'Target Values and Predictions'})

    return _go.Figure(data=data, layout=layout)
