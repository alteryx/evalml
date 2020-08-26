import sys
import traceback
from collections import namedtuple

import numpy as np
import pandas as pd

from evalml.exceptions import PipelineScoreError
from evalml.model_understanding.prediction_explanations._user_interface import (
    _TableClassificationPredictedValuesMaker,
    _TableEmptyPredictedValuesMaker,
    _TableHeadingMaker,
    _make_single_prediction_shap_table,
    _TableRegressionPredictedValuesMaker,
    _TextReportMaker,
    _TableSHAPMaker,
    _DictSHAPMaker,
    _DictHeadingMaker,
    _DictReportMaker,
    _DictRegressionPredictedValuesMaker,
    _DictClassificationPredictedValuesMaker,
)
from evalml.problem_types import ProblemTypes

_PipelineData = namedtuple("PipelineData", ["pipeline", "input_features",
                                            "y_true", "y_pred", "y_pred_values",
                                            "errors", "index_list"])


def explain_prediction(pipeline, input_features, top_k=3, training_data=None, include_shap_values=False,
                       output_format="text"):
    """Creates table summarizing the top_k positive and top_k negative contributing features to the prediction of a single datapoint.

    XGBoost models and CatBoost multiclass classifiers are not currently supported.

    Arguments:
        pipeline (PipelineBase): Fitted pipeline whose predictions we want to explain with SHAP.
        input_features (pd.DataFrame): Dataframe of features - needs to correspond to data the pipeline was fit on.
        top_k (int): How many of the highest/lowest features to include in the table.
        training_data (pd.DataFrame): Training data the pipeline was fit on.
            This is required for non-tree estimators because we need a sample of training data for the KernelSHAP algorithm.
        include_shap_values (bool): Whether the SHAP values should be included in an extra column in the output.
            Default is False.

    Returns:
        str: Table
    """
    return _make_single_prediction_shap_table(pipeline, input_features, top_k, training_data, include_shap_values,
                                              output_format=output_format)


def abs_error(y_true, y_pred):
    """Computes the absolute error per data point for regression problems.

    Arguments:
        y_true (pd.Series): True labels.
        y_pred (pd.Series): Predicted values.

    Returns:
       pd.Series
    """
    return np.abs(y_true - y_pred)


def cross_entropy(y_true, y_pred_proba):
    """Computes Cross Entropy Loss per data point for classification problems.

    Arguments:
        y_true (pd.Series): True labels encoded as ints.
        y_pred_proba (pd.DataFrame): Predicted probabilities. One column per class.

    Returns:
       pd.Series
    """
    n_data_points = y_pred_proba.shape[0]
    log_likelihood = -np.log(y_pred_proba.values[range(n_data_points), y_true.values.astype("int")])
    return pd.Series(log_likelihood)


DEFAULT_METRICS = {ProblemTypes.BINARY: cross_entropy,
                   ProblemTypes.MULTICLASS: cross_entropy,
                   ProblemTypes.REGRESSION: abs_error}


def explain_predictions(pipeline, input_features, training_data=None, top_k_features=3, include_shap_values=False,
                        output_format="text"):
    """Creates a report summarizing the top contributing features for each data point in the input features.

    XGBoost models and CatBoost multiclass classifiers are not currently supported.

    Arguments:
        pipeline (PipelineBase): Fitted pipeline whose predictions we want to explain with SHAP.
        input_features (pd.DataFrame): Dataframe of input data to evaluate the pipeline on.
        training_data (pd.DataFrame): Dataframe of data the pipeline was fit on. This can be omitted for pipelines
            with tree-based estimators.
        top_k_features (int): How many of the highest/lowest contributing feature to include in the table for each
            data point.
        include_shap_values (bool): Whether SHAP values should be included in the table. Default is False.

    Returns:
        str - A report with the pipeline name and parameters and a table for each row of input_features.
            The table will have the following columns: Feature Name, Contribution to Prediction, SHAP Value (optional),
            and each row of the table will be a feature.
    """
    if not (isinstance(input_features, pd.DataFrame) and not input_features.empty):
        raise ValueError("Parameter input_features must be a non-empty dataframe.")
    data = _PipelineData(pipeline, input_features, None, None, None, None, range(input_features.shape[0]))
    if output_format == "text":
        header_maker = _TableHeadingMaker([""], input_features.shape[0])
        prediction_results_maker = _TableEmptyPredictedValuesMaker()
        table_maker = _TableSHAPMaker(top_k_features, include_shap_values, training_data)
        report_maker = _TextReportMaker(header_maker, prediction_results_maker, table_maker)
        return report_maker.make_report(data)
    else:
        table_maker = _DictSHAPMaker(top_k_features, include_shap_values, training_data)
        section_maker = _DictReportMaker(None, None, table_maker)
        return section_maker.make_report(data)


def explain_predictions_best_worst(pipeline, input_features, y_true, num_to_explain=5, top_k_features=3,
                                   include_shap_values=False, metric=None, output_format="text"):
    """Creates a report summarizing the top contributing features for the best and worst points in the dataset as measured by error to true labels.

    XGBoost models and CatBoost multiclass classifiers are not currently supported.

    Arguments:
        pipeline (PipelineBase): Fitted pipeline whose predictions we want to explain with SHAP.
        input_features (pd.DataFrame): Dataframe of input data to evaluate the pipeline on.
        y_true (pd.Series): True labels for the input data.
        num_to_explain (int): How many of the best, worst, random data points to explain.
        top_k_features (int): How many of the highest/lowest contributing feature to include in the table for each
            data point.
        include_shap_values (bool): Whether SHAP values should be included in the table. Default is False.
        metric (callable): The metric used to identify the best and worst points in the dataset. Function must accept
            the true labels and predicted value or probabilities as the only arguments and lower values
            must be better. By default, this will be the absolute error for regression problems and cross entropy loss
            for classification problems.

    Returns:
        str - A report with the pipeline name and parameters. For each of the best/worst rows of input_features, the
            predicted values, true labels, and metric value will be listed along with a table. The table will have the
            following columns: Feature Name, Contribution to Prediction, SHAP Value (optional), and each row of the
            table will correspond to a feature.
    """
    if not (isinstance(input_features, pd.DataFrame) and input_features.shape[0] >= num_to_explain * 2):
        raise ValueError(f"Input features must be a dataframe with more than {num_to_explain * 2} rows! "
                         "Convert to a dataframe and select a smaller value for num_to_explain if you do not have "
                         "enough data.")
    if not isinstance(y_true, pd.Series):
        raise ValueError("Parameter y_true must be a pd.Series.")
    if y_true.shape[0] != input_features.shape[0]:
        raise ValueError("Parameters y_true and input_features must have the same number of data points. Received: "
                         f"true labels: {y_true.shape[0]} and {input_features.shape[0]}")
    if not metric:
        metric = DEFAULT_METRICS[pipeline.problem_type]

    try:
        if pipeline.problem_type == ProblemTypes.REGRESSION:
            y_pred = pipeline.predict(input_features)
            y_pred_values = None
            errors = metric(y_true, y_pred)
        else:
            y_pred = pipeline.predict_proba(input_features)
            y_pred_values = pipeline.predict(input_features)
            errors = metric(pipeline._encode_targets(y_true), y_pred)
    except Exception as e:
        tb = traceback.format_tb(sys.exc_info()[2])
        raise PipelineScoreError(exceptions={metric.__name__: (e, tb)}, scored_successfully={})

    sorted_scores = errors.sort_values()
    best = sorted_scores.index[:num_to_explain]
    worst = sorted_scores.index[-num_to_explain:]
    index_list = best.tolist() + worst.tolist()

    pipeline_data = _PipelineData(pipeline, input_features, y_true, y_pred, y_pred_values, errors, index_list)

    if output_format == "text":
        heading_maker = _TableHeadingMaker(["Best ", "Worst "], n_indices=num_to_explain)
        table_maker = _TableSHAPMaker(top_k_features, include_shap_values, training_data=input_features)
        if pipeline.problem_type == ProblemTypes.REGRESSION:
            prediction_results_class = _TableRegressionPredictedValuesMaker
        else:
            prediction_results_class = _TableClassificationPredictedValuesMaker
        prediction_results_maker = prediction_results_class(metric.__name__, y_pred_values)

        report_maker = _TextReportMaker(heading_maker, prediction_results_maker, table_maker)
    else:
        heading_maker = _DictHeadingMaker(["best", "worst"], n_indices=num_to_explain)
        table_maker = _DictSHAPMaker(top_k_features, include_shap_values, training_data=input_features)
        if pipeline.problem_type == ProblemTypes.REGRESSION:
            prediction_results_class = _DictRegressionPredictedValuesMaker
        else:
            prediction_results_class = _DictClassificationPredictedValuesMaker
        prediction_results_maker = prediction_results_class(metric.__name__, y_pred_values)

        report_maker = _DictReportMaker(heading_maker, prediction_results_maker, table_maker)
    return report_maker.make_report(pipeline_data)
