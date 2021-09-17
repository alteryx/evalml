"""Prediction explanation tools."""
import sys
import traceback
from collections import namedtuple
from enum import Enum
from timeit import default_timer as timer

import numpy as np
import pandas as pd

from evalml.exceptions import PipelineScoreError
from evalml.model_family import ModelFamily
from evalml.model_understanding.prediction_explanations._report_creator_factory import (
    _report_creator_factory,
)
from evalml.problem_types import ProblemTypes, is_regression, is_time_series
from evalml.utils import infer_feature_types
from evalml.utils.gen_utils import drop_rows_with_nans

# Container for all of the pipeline-related data we need to create reports. Helps standardize APIs of report makers.
_ReportData = namedtuple(
    "ReportData",
    [
        "pipeline",
        "pipeline_features",
        "input_features",
        "y_true",
        "y_pred",
        "y_pred_values",
        "errors",
        "index_list",
        "metric",
    ],
)


def explain_predictions(
    pipeline,
    input_features,
    y,
    indices_to_explain,
    top_k_features=3,
    include_shap_values=False,
    include_expected_value=False,
    output_format="text",
    training_data=None,
    training_target=None,
):
    """Creates a report summarizing the top contributing features for each data point in the input features.

    XGBoost and Stacked Ensemble models, as well as CatBoost multiclass classifiers, are not currently supported.

    Args:
        pipeline (PipelineBase): Fitted pipeline whose predictions we want to explain with SHAP.
        input_features (pd.DataFrame): Dataframe of input data to evaluate the pipeline on.
        y (pd.Series): Labels for the input data.
        indices_to_explain (list[int]): List of integer indices to explain.
        top_k_features (int): How many of the highest/lowest contributing feature to include in the table for each
            data point.  Default is 3.
        include_shap_values (bool): Whether SHAP values should be included in the table. Default is False.
        include_expected_value (bool): Whether the expected value should be included in the table. Default is False.
        output_format (str): Either "text", "dict", or "dataframe". Default is "text".
        training_data (pd.DataFrame, np.ndarray): Data the pipeline was trained on. Required and only used for time series pipelines.
        training_target (pd.Series, np.ndarray): Targets used to train the pipeline. Required and only used for time series pipelines.

    Returns:
        str, dict, or pd.DataFrame: A report explaining the top contributing features to each prediction for each row of input_features.
            The report will include the feature names, prediction contribution, and SHAP Value (optional).

    Raises:
        ValueError: if input_features is empty.
        ValueError: if an output_format outside of "text", "dict" or "dataframe is provided.
        ValueError: if the requested index falls outside the input_feature's boundaries.
    """
    input_features = infer_feature_types(input_features)

    if pipeline.model_family == ModelFamily.ENSEMBLE:
        raise ValueError("Cannot explain predictions for a stacked ensemble pipeline")
    if input_features.empty:
        raise ValueError("Parameter input_features must be a non-empty dataframe.")
    if output_format not in {"text", "dict", "dataframe"}:
        raise ValueError(
            f"Parameter output_format must be either text, dict, or dataframe. Received {output_format}"
        )
    if any([x < 0 or x >= len(input_features) for x in indices_to_explain]):
        raise ValueError(
            f"Explained indices should be between 0 and {len(input_features) - 1}"
        )
    if is_time_series(pipeline.problem_type) and (
        training_target is None or training_data is None
    ):
        raise ValueError(
            "Prediction explanations for time series pipelines requires that training_target and "
            "training_data are not None"
        )

    pipeline_features = pipeline.compute_estimator_features(
        input_features, y, training_data, training_target
    )

    data = _ReportData(
        pipeline,
        pipeline_features,
        input_features,
        y_true=y,
        y_pred=None,
        y_pred_values=None,
        errors=None,
        index_list=indices_to_explain,
        metric=None,
    )

    report_creator = _report_creator_factory(
        data,
        report_type="explain_predictions",
        output_format=output_format,
        top_k_features=top_k_features,
        include_shap_values=include_shap_values,
        include_expected_value=include_expected_value,
    )
    return report_creator(data)


def _update_progress(start_time, current_time, progress_stage, callback_function):
    """Helper function for updating progress of a function and making a call to the user-provided callback function, if provided.

    The callback function should accept the following parameters:
        - progress_stage: stage of computation
        - time_elapsed: total time in seconds that has elapsed since start of call
    """
    if callback_function is not None:
        elapsed_time = current_time - start_time
        callback_function(progress_stage, elapsed_time)


class ExplainPredictionsStage(Enum):
    """Enum for prediction stage."""

    PREPROCESSING_STAGE = "preprocessing_stage"
    PREDICT_STAGE = "predict_stage"
    COMPUTE_FEATURE_STAGE = "compute_feature_stage"
    COMPUTE_SHAP_VALUES_STAGE = "compute_shap_value_stage"
    DONE = "done"


def explain_predictions_best_worst(
    pipeline,
    input_features,
    y_true,
    num_to_explain=5,
    top_k_features=3,
    include_shap_values=False,
    metric=None,
    output_format="text",
    callback=None,
    training_data=None,
    training_target=None,
):
    """Creates a report summarizing the top contributing features for the best and worst points in the dataset as measured by error to true labels.

    XGBoost and Stacked Ensemble models, as well as CatBoost multiclass classifiers, are not currently supported.

    Args:
        pipeline (PipelineBase): Fitted pipeline whose predictions we want to explain with SHAP.
        input_features (pd.DataFrame): Input data to evaluate the pipeline on.
        y_true (pd.Series): True labels for the input data.
        num_to_explain (int): How many of the best, worst, random data points to explain.
        top_k_features (int): How many of the highest/lowest contributing feature to include in the table for each
            data point.
        include_shap_values (bool): Whether SHAP values should be included in the table. Default is False.
        metric (callable): The metric used to identify the best and worst points in the dataset. Function must accept
            the true labels and predicted value or probabilities as the only arguments and lower values
            must be better. By default, this will be the absolute error for regression problems and cross entropy loss
            for classification problems.
        output_format (str): Either "text" or "dict". Default is "text".
        callback (callable): Function to be called with incremental updates. Has the following parameters:
            - progress_stage: stage of computation
            - time_elapsed: total time in seconds that has elapsed since start of call
        training_data (pd.DataFrame, np.ndarray): Data the pipeline was trained on. Required and only used for time series pipelines.
        training_target (pd.Series, np.ndarray): Targets used to train the pipeline. Required and only used for time series pipelines.

    Returns:
        str, dict, or pd.DataFrame: A report explaining the top contributing features for the best/worst predictions in the input_features.
            For each of the best/worst rows of input_features, the predicted values, true labels, metric value,
            feature names, prediction contribution, and SHAP Value (optional) will be listed.

    Raises:
        ValueError: If input_features does not have more than twice the requested features to explain.
        ValueError: If y_true and input_features have mismatched lengths.
        ValueError: If an output_format outside of "text", "dict" or "dataframe is provided.
        PipelineScoreError: If the pipeline errors out while scoring.
    """
    start_time = timer()
    _update_progress(
        start_time, timer(), ExplainPredictionsStage.PREPROCESSING_STAGE, callback
    )

    input_features = infer_feature_types(input_features)
    y_true = infer_feature_types(y_true)

    if not (input_features.shape[0] >= num_to_explain * 2):
        raise ValueError(
            f"Input features must be a dataframe with more than {num_to_explain * 2} rows! "
            "Convert to a dataframe and select a smaller value for num_to_explain if you do not have "
            "enough data."
        )
    if y_true.shape[0] != input_features.shape[0]:
        raise ValueError(
            "Parameters y_true and input_features must have the same number of data points. Received: "
            f"true labels: {y_true.shape[0]} and {input_features.shape[0]}"
        )
    if output_format not in {"text", "dict", "dataframe"}:
        raise ValueError(
            f"Parameter output_format must be either text, dict, or dataframe. Received {output_format}"
        )
    if pipeline.model_family == ModelFamily.ENSEMBLE:
        raise ValueError("Cannot explain predictions for a stacked ensemble pipeline")
    if not metric:
        metric = DEFAULT_METRICS[pipeline.problem_type]
    _update_progress(
        start_time, timer(), ExplainPredictionsStage.PREDICT_STAGE, callback
    )
    if is_time_series(pipeline.problem_type) and (
        training_target is None or training_data is None
    ):
        raise ValueError(
            "Prediction explanations for time series pipelines requires that training_target and "
            "training_data are not None"
        )

    try:
        if is_regression(pipeline.problem_type):
            if is_time_series(pipeline.problem_type):
                y_pred = pipeline.predict_in_sample(
                    input_features, y_true, training_data, training_target
                )
            else:
                y_pred = pipeline.predict(input_features)
            y_pred_values = None
            y_true_no_nan, y_pred_no_nan = drop_rows_with_nans(y_true, y_pred)
            errors = metric(y_true_no_nan, y_pred_no_nan)
        else:
            if is_time_series(pipeline.problem_type):
                y_pred = pipeline.predict_proba_in_sample(
                    input_features, y_true, training_data, training_target
                )
                y_pred_values = pipeline.predict_in_sample(
                    input_features, y_true, training_data, training_target
                )
            else:
                y_pred = pipeline.predict_proba(input_features)
                y_pred_values = pipeline.predict(input_features)
            y_true_no_nan, y_pred_no_nan, y_pred_values_no_nan = drop_rows_with_nans(
                y_true, y_pred, y_pred_values
            )
            errors = metric(pipeline._encode_targets(y_true_no_nan), y_pred_no_nan)
    except Exception as e:
        tb = traceback.format_tb(sys.exc_info()[2])
        raise PipelineScoreError(
            exceptions={metric.__name__: (e, tb)}, scored_successfully={}
        )

    errors = pd.Series(errors)
    sorted_scores = errors.sort_values()
    best_indices = sorted_scores.index[:num_to_explain]
    worst_indices = sorted_scores.index[-num_to_explain:]
    index_list = best_indices.tolist() + worst_indices.tolist()
    _update_progress(
        start_time, timer(), ExplainPredictionsStage.COMPUTE_FEATURE_STAGE, callback
    )

    pipeline_features = pipeline.compute_estimator_features(
        input_features, y_true, training_data, training_target
    )

    _update_progress(
        start_time, timer(), ExplainPredictionsStage.COMPUTE_SHAP_VALUES_STAGE, callback
    )

    data = _ReportData(
        pipeline,
        pipeline_features,
        input_features,
        y_true,
        y_pred,
        y_pred_values,
        errors,
        index_list,
        metric,
    )

    report_creator = _report_creator_factory(
        data,
        report_type="explain_predictions_best_worst",
        output_format=output_format,
        top_k_features=top_k_features,
        include_shap_values=include_shap_values,
        num_to_explain=num_to_explain,
        include_expected_value=True,
    )

    _update_progress(start_time, timer(), ExplainPredictionsStage.DONE, callback)
    return report_creator(data)


def abs_error(y_true, y_pred):
    """Computes the absolute error per data point for regression problems.

    Args:
        y_true (pd.Series): True labels.
        y_pred (pd.Series): Predicted values.

    Returns:
        np.ndarray
    """
    return np.abs(y_true.values - y_pred.values)


def cross_entropy(y_true, y_pred_proba):
    """Computes Cross Entropy Loss per data point for classification problems.

    Args:
        y_true (pd.Series): True labels encoded as ints.
        y_pred_proba (pd.DataFrame): Predicted probabilities. One column per class.

    Returns:
        np.ndarray
    """
    n_data_points = y_pred_proba.shape[0]
    log_likelihood = -np.log(
        y_pred_proba.values[range(n_data_points), y_true.values.astype("int")]
    )
    return log_likelihood


DEFAULT_METRICS = {
    ProblemTypes.BINARY: cross_entropy,
    ProblemTypes.MULTICLASS: cross_entropy,
    ProblemTypes.REGRESSION: abs_error,
    ProblemTypes.TIME_SERIES_BINARY: cross_entropy,
    ProblemTypes.TIME_SERIES_MULTICLASS: cross_entropy,
    ProblemTypes.TIME_SERIES_REGRESSION: abs_error,
}
