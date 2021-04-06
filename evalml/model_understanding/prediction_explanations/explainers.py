import sys
import traceback
from collections import namedtuple

import numpy as np
import pandas as pd

from evalml.exceptions import PipelineScoreError
from evalml.model_family import ModelFamily
from evalml.model_understanding.prediction_explanations._report_creator_factory import (
    _report_creator_factory
)
from evalml.problem_types import ProblemTypes, is_regression, is_time_series
from evalml.utils import _convert_woodwork_types_wrapper, infer_feature_types
from evalml.utils.gen_utils import drop_rows_with_nans

# Container for all of the pipeline-related data we need to create reports. Helps standardize APIs of report makers.
_ReportData = namedtuple("ReportData", ["pipeline", "pipeline_features", "input_features",
                                        "y_true", "y_pred", "y_pred_values", "errors", "index_list", "metric"])


def explain_predictions(pipeline, input_features, y, indices_to_explain, top_k_features=3, include_shap_values=False,
                        output_format="text"):
    """Creates a report summarizing the top contributing features for each data point in the input features.

    XGBoost and Stacked Ensemble models, as well as CatBoost multiclass classifiers, are not currently supported.

    Arguments:
        pipeline (PipelineBase): Fitted pipeline whose predictions we want to explain with SHAP.
        input_features (ww.DataTable, pd.DataFrame): Dataframe of input data to evaluate the pipeline on.
        y (ww.DataColumn, pd.Series): Labels for the input data.
        indices_to_explain (list(int)): List of integer indices to explain.
        top_k_features (int): How many of the highest/lowest contributing feature to include in the table for each
            data point.  Default is 3.
        include_shap_values (bool): Whether SHAP values should be included in the table. Default is False.
        output_format (str): Either "text", "dict", or "dataframe". Default is "text".

    Returns:
        str, dict, or pd.DataFrame - A report explaining the top contributing features to each prediction for each row of input_features.
            The report will include the feature names, prediction contribution, and SHAP Value (optional).

    Raises:
        ValueError: if input_features is empty.
        ValueError: if an output_format outside of "text", "dict" or "dataframe is provided.
        ValueError: if the requested index falls outside the input_feature's boundaries.
    """
    input_features = infer_feature_types(input_features)
    input_features = _convert_woodwork_types_wrapper(input_features.to_dataframe())

    if pipeline.model_family == ModelFamily.ENSEMBLE:
        raise ValueError("Cannot explain predictions for a stacked ensemble pipeline")
    if input_features.empty:
        raise ValueError("Parameter input_features must be a non-empty dataframe.")
    if output_format not in {"text", "dict", "dataframe"}:
        raise ValueError(f"Parameter output_format must be either text, dict, or dataframe. Received {output_format}")
    if any([x < 0 or x >= len(input_features) for x in indices_to_explain]):
        raise ValueError(f"Explained indices should be between 0 and {len(input_features) - 1}")

    pipeline_features = pipeline.compute_estimator_features(input_features, y).to_dataframe()

    data = _ReportData(pipeline, pipeline_features, input_features, y_true=y, y_pred=None,
                       y_pred_values=None, errors=None, index_list=indices_to_explain, metric=None)

    report_creator = _report_creator_factory(data, report_type="explain_predictions",
                                             output_format=output_format, top_k_features=top_k_features,
                                             include_shap_values=include_shap_values)
    return report_creator(data)


def explain_predictions_best_worst(pipeline, input_features, y_true, num_to_explain=5, top_k_features=3,
                                   include_shap_values=False, metric=None, output_format="text"):
    """Creates a report summarizing the top contributing features for the best and worst points in the dataset as measured by error to true labels.

    XGBoost and Stacked Ensemble models, as well as CatBoost multiclass classifiers, are not currently supported.

    Arguments:
        pipeline (PipelineBase): Fitted pipeline whose predictions we want to explain with SHAP.
        input_features (ww.DataTable, pd.DataFrame): Input data to evaluate the pipeline on.
        y_true (ww.DataColumn, pd.Series): True labels for the input data.
        num_to_explain (int): How many of the best, worst, random data points to explain.
        top_k_features (int): How many of the highest/lowest contributing feature to include in the table for each
            data point.
        include_shap_values (bool): Whether SHAP values should be included in the table. Default is False.
        metric (callable): The metric used to identify the best and worst points in the dataset. Function must accept
            the true labels and predicted value or probabilities as the only arguments and lower values
            must be better. By default, this will be the absolute error for regression problems and cross entropy loss
            for classification problems.
        output_format (str): Either "text" or "dict". Default is "text".

    Returns:
        str, dict, or pd.DataFrame - A report explaining the top contributing features for the best/worst predictions in the input_features.
            For each of the best/worst rows of input_features, the predicted values, true labels, metric value,
            feature names, prediction contribution, and SHAP Value (optional) will be listed.

    Raises:
        ValueError: if input_features does not have more than twice the requested features to explain.
        ValueError: if y_true and input_features have mismatched lengths.
        ValueError: if an output_format outside of "text", "dict" or "dataframe is provided.
    """
    input_features = infer_feature_types(input_features)
    input_features = _convert_woodwork_types_wrapper(input_features.to_dataframe())
    y_true = infer_feature_types(y_true)
    y_true = _convert_woodwork_types_wrapper(y_true.to_series())

    if not (input_features.shape[0] >= num_to_explain * 2):
        raise ValueError(f"Input features must be a dataframe with more than {num_to_explain * 2} rows! "
                         "Convert to a dataframe and select a smaller value for num_to_explain if you do not have "
                         "enough data.")
    if y_true.shape[0] != input_features.shape[0]:
        raise ValueError("Parameters y_true and input_features must have the same number of data points. Received: "
                         f"true labels: {y_true.shape[0]} and {input_features.shape[0]}")
    if output_format not in {"text", "dict", "dataframe"}:
        raise ValueError(f"Parameter output_format must be either text, dict, or dataframe. Received {output_format}")
    if pipeline.model_family == ModelFamily.ENSEMBLE:
        raise ValueError("Cannot explain predictions for a stacked ensemble pipeline")
    if not metric:
        metric = DEFAULT_METRICS[pipeline.problem_type]

    try:
        if is_regression(pipeline.problem_type):
            if is_time_series(pipeline.problem_type):
                y_pred = pipeline.predict(input_features, y=y_true).to_series()
            else:
                y_pred = pipeline.predict(input_features).to_series()
            y_pred_values = None
            y_true_no_nan, y_pred_no_nan = drop_rows_with_nans(y_true, y_pred)
            errors = metric(y_true_no_nan, y_pred_no_nan)
        else:
            if is_time_series(pipeline.problem_type):
                y_pred = pipeline.predict_proba(input_features, y=y_true).to_dataframe()
                y_pred_values = pipeline.predict(input_features, y=y_true).to_series()
            else:
                y_pred = pipeline.predict_proba(input_features).to_dataframe()
                y_pred_values = pipeline.predict(input_features).to_series()
            y_true_no_nan, y_pred_no_nan, y_pred_values_no_nan = drop_rows_with_nans(y_true, y_pred, y_pred_values)
            errors = metric(pipeline._encode_targets(y_true_no_nan), y_pred_no_nan)
    except Exception as e:
        tb = traceback.format_tb(sys.exc_info()[2])
        raise PipelineScoreError(exceptions={metric.__name__: (e, tb)}, scored_successfully={})

    errors = pd.Series(errors, index=y_pred_no_nan.index)
    sorted_scores = errors.sort_values()
    best_indices = sorted_scores.index[:num_to_explain]
    worst_indices = sorted_scores.index[-num_to_explain:]
    index_list = best_indices.tolist() + worst_indices.tolist()

    pipeline_features = pipeline.compute_estimator_features(input_features, y_true).to_dataframe()

    data = _ReportData(pipeline, pipeline_features, input_features, y_true, y_pred, y_pred_values, errors, index_list, metric)

    report_creator = _report_creator_factory(data, report_type="explain_predictions_best_worst",
                                             output_format=output_format, top_k_features=top_k_features,
                                             include_shap_values=include_shap_values, num_to_explain=num_to_explain)
    return report_creator(data)


def abs_error(y_true, y_pred):
    """Computes the absolute error per data point for regression problems.

    Arguments:
        y_true (pd.Series): True labels.
        y_pred (pd.Series): Predicted values.

    Returns:
        np.ndarray
    """
    return np.abs(y_true.values - y_pred.values)


def cross_entropy(y_true, y_pred_proba):
    """Computes Cross Entropy Loss per data point for classification problems.

    Arguments:
        y_true (pd.Series): True labels encoded as ints.
        y_pred_proba (pd.DataFrame): Predicted probabilities. One column per class.

    Returns:
        np.ndarray
    """
    n_data_points = y_pred_proba.shape[0]
    log_likelihood = -np.log(y_pred_proba.values[range(n_data_points), y_true.values.astype("int")])
    return log_likelihood


DEFAULT_METRICS = {ProblemTypes.BINARY: cross_entropy,
                   ProblemTypes.MULTICLASS: cross_entropy,
                   ProblemTypes.REGRESSION: abs_error,
                   ProblemTypes.TIME_SERIES_BINARY: cross_entropy,
                   ProblemTypes.TIME_SERIES_MULTICLASS: cross_entropy,
                   ProblemTypes.TIME_SERIES_REGRESSION: abs_error}
