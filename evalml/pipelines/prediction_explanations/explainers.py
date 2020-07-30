import sys
import traceback

import numpy as np
import pandas as pd

from evalml.exceptions import PipelineScoreError
from evalml.pipelines.prediction_explanations._algorithms import (
    _compute_shap_values,
    _normalize_shap_values
)
from evalml.pipelines.prediction_explanations._user_interface import (
    _make_single_prediction_table
)
from evalml.problem_types import ProblemTypes


def explain_prediction(pipeline, input_features, top_k=3, training_data=None, include_shap_values=False):
    """Creates table summarizing the top_k positive and top_k negative contributing features to the prediction of a single datapoint.

    XGBoost models and CatBoost multiclass classifiers are not currently supported.

    Arguments:
        pipeline (PipelineBase): Fitted pipeline whose predictions we want to explain with SHAP.
        input_features (pd.DataFrame): Dataframe of features - needs to correspond to data the pipeline was fit on.
        top_k (int): How many of the highest/lowest features to include in the table.
        training_data (pd.DataFrame): Training data the pipeline was fit on.
            This is required for non-tree estimators because we need a sample of training data for the KernelSHAP algorithm.
        include_shap_values (bool): Whether the SHAP values should be included in an extra column in the output.

    Returns:
        str: Table
    """
    if not (isinstance(input_features, pd.DataFrame) and input_features.shape[0] == 1):
        raise ValueError("features must be stored in a dataframe of one row.")
    shap_values = _compute_shap_values(pipeline, input_features, training_data)
    normalized_shap_values = _normalize_shap_values(shap_values)
    class_names = None
    if hasattr(pipeline, "_classes"):
        class_names = pipeline._classes
    return _make_single_prediction_table(shap_values, normalized_shap_values, top_k, include_shap_values, class_names)


def _abs_error(y_true, y_pred):
    """Computes the absolute error per data point for regression problems.

    Arguments:
        y_true (pd.Series): Ground truth
        y_pred (pd.Series): Predicted values.

    Returns:
        pd.Series
    """
    return pd.abs(y_true - y_pred)


def _cross_entropy(y_true, y_pred_proba):
    """Computes Cross Entropy Loss per data point for classification problems.

    Arguments:
        y_true (pd.Series): Ground truth labels. Not one-hot-encoded.
        y_pred_proba (pd.DataFrame): Predicted probabilities. One column per class.

    Returns:
        pd.Series
    """
    n_data_points = y_pred_proba.shape[0]
    log_likelihood = -np.log(y_pred_proba.values[range(n_data_points), y_true.values.astype("int")])
    return pd.Series(log_likelihood)


_DEFAULT_METRICS = {ProblemTypes.BINARY: _cross_entropy,
                    ProblemTypes.MULTICLASS: _cross_entropy,
                    ProblemTypes.REGRESSION: _abs_error}


def _explain_indices(pipeline, input_features, y_true, y_pred, scores, indices, prefix,
                     error_name,
                     top_k_features=3, include_shap_values=False):
    strings = []
    for rank, index in enumerate(indices):
        strings.append(f"\t{prefix} {rank + 1} of {len(indices)}\n\n")

        if y_pred is not None:
            if pipeline.problem_type != ProblemTypes.REGRESSION:
                pred_value = [f"{col_name}: {pred}" for col_name, pred in zip(y_pred.columns, round(y_pred.iloc[index], 3).tolist())]
                pred_value = "[" + ", ".join(pred_value) + "]"
                true_value = y_true[index]
                prediction_name = "Predicted Probabilities"
            else:
                pred_value = round(y_pred.iloc[index], 3)
                true_value = round(y_true[index], 3)
                prediction_name = "Predicted Value"

            strings.append(f"\t\t{prediction_name}: {pred_value}\n")
            strings.append(f"\t\tTarget Value: {true_value}\n")
            strings.append(f"\t\t{error_name}: {round(scores[index], 3)}\n\n")

        table = explain_prediction(pipeline, input_features.iloc[index:(index + 1)],
                                   training_data=input_features, top_k=top_k_features,
                                   include_shap_values=include_shap_values)
        table = table.splitlines()
        # Indent the rows of the table to match the indentation of the entire report.
        table = ["\t\t" + line + "\n" for line in table] + ["\n\n"]
        strings.extend(table)
    return strings


def explain_predictions_best_worst(pipeline, input_features, y_true, num_to_explain=5, top_k_features=3,
                                   include_shap_values=False, metric=None):
    """Creates a report summarizing the top contributing features for the best and worst points in the dataset as measured by error to ground truth.

    XGBoost models and CatBoost multiclass classifiers are not currently supported.

    Arguments:
        pipeline (PipelineBase): Fitted pipeline whose predictions we want to explain with SHAP.
        input_features (pd.DataFrame): Dataframe of input data to evaluate the pipeline on.
        y_true (pd.Series): Ground truth for the input data.
        num_to_explain (int): How many of the best, worst, random data points to explain.
        top_k_features (int): How many of the highest/lowest contributing feature to include in the table for each
            data point.
        include_shap_values (bool): Whether SHAP values should be included in the table.
        metric (callable): The metric used to identify the best and worst points in the dataset. Function must accept
            the ground_truth and predicted value or probabilities as the only arguments and lower values
             must be better. By default, this will be the absolute error for regression problems and cross entropy loss
             for classification problems.

    Returns:
        str
    """
    if not (isinstance(input_features, pd.DataFrame) and input_features.shape[0] >= num_to_explain * 2):
        raise ValueError(f"Input features must be a dataframe with more than {num_to_explain * 2} rows! "
                         "Convert to a dataframe and select a smaller value for num_to_explain if you do not have "
                         "enough data.")
    if not isinstance(y_true, pd.Series):
        raise ValueError("Parameter y_true must be a pd.Series.")
    if y_true.shape[0] != input_features.shape[0]:
        raise ValueError("Parameters y_true and input_features must have the same number of data points. Received: "
                         f"ground_truth: {y_true.shape[0]} and {input_features.shape[0]}")
    if not metric:
        metric = _DEFAULT_METRICS[pipeline.problem_type]

    try:
        if pipeline.problem_type == ProblemTypes.REGRESSION:
            y_pred = pipeline.predict(input_features)
            score = metric(y_true, y_pred)
        else:
            y_pred = pipeline.predict_proba(input_features)
            score = metric(pipeline._encode_targets(y_true), y_pred)
    except Exception as e:
        tb = traceback.format_tb(sys.exc_info()[2])
        raise PipelineScoreError(exceptions={metric.__name__: (e, tb)}, scored_successfully={})

    sorted_scores = score.sort_values()
    best = sorted_scores.index[:num_to_explain]
    worst = sorted_scores.index[-num_to_explain:]
    report = [pipeline.name + "\n\n", str(pipeline.parameters) + "\n\n"]

    for index_list, prefix in zip([best, worst], ["Best", "Worst"]):
        section = _explain_indices(pipeline, input_features, y_true, y_pred, score, index_list, prefix,
                                   metric.__name__, top_k_features, include_shap_values)
        report.extend(section)
    return "".join(report)


def explain_predictions(pipeline, input_features, top_k_features=3, include_shap_values=False):
    """Creates a report summarizing the top contributing features for each data point in the input features.

    XGBoost models and CatBoost multiclass classifiers are not currently supported.

    Arguments:
        pipeline (PipelineBase): Fitted pipeline whose predictions we want to explain with SHAP.
        input_features (pd.DataFrame): Dataframe of input data to evaluate the pipeline on.
        top_k_features (int): How many of the highest/lowest contributing feature to include in the table for each
            data point.
        include_shap_values (bool): Whether SHAP values should be included in the table.

    Returns:
        str
    """
    if not (isinstance(input_features, pd.DataFrame) and not input_features.empty):
        raise ValueError("Parameter input_features must be a non-empty dataframe.")
    header = [pipeline.name + "\n\n", str(pipeline.parameters) + "\n\n"]
    body = _explain_indices(pipeline, input_features, y_true=None, y_pred=None, scores=None,
                            indices=range(input_features.shape[0]), prefix="",
                            top_k_features=top_k_features, include_shap_values=include_shap_values,
                            error_name=None)
    return "".join(header + body)
