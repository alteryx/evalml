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
    _ReportSectionMaker,
    _SHAPBinaryTableMaker,
    _SHAPMultiClassTableMaker,
    _SHAPRegressionTableMaker
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

    if pipeline.problem_type == ProblemTypes.REGRESSION:
        table_maker = _SHAPRegressionTableMaker()
    elif pipeline.problem_type == ProblemTypes.BINARY:
        table_maker = _SHAPBinaryTableMaker()
    else:
        table_maker = _SHAPMultiClassTableMaker(pipeline._classes)

    shap_values = _compute_shap_values(pipeline, input_features, training_data)
    normalized_shap_values = _normalize_shap_values(shap_values)
    return table_maker(shap_values, normalized_shap_values, top_k, include_shap_values)


def _abs_error(y_true, y_pred):
    """Computes the absolute error per data point for regression problems.

    Arguments:
        y_true (pd.Series): True labels.
        y_pred (pd.Series): Predicted values.

    Returns:
       pd.Series
    """
    return np.abs(y_true - y_pred)


def _cross_entropy(y_true, y_pred_proba):
    """Computes Cross Entropy Loss per data point for classification problems.

    Arguments:
        y_true (pd.Series): True labels as ints but not one-hot-encoded.
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


class _HeadingMaker:
    """Makes the heading for reports.

    Differences between best/worst reports and reports where user manually specifies the input features subset
    are handled by formatting the value of the prefix parameter in the initialization.
    """

    def __init__(self, prefix, n_indices):
        self.prefix = prefix
        self.n_indices = n_indices

    def __call__(self, rank, index):
        return [f"\t{self.prefix}{rank + 1} of {self.n_indices}\n\n"]


class _SHAPTableMaker:
    """Makes the SHAP table for reports.

    The table is the same whether the user requests a best/worst report or they manually specified the
    subset of the input features.

    Handling the differences in how the table is formatted between regression and classification problems
    is delegated to the explain_prediction function.
    """

    def __init__(self, top_k_features, include_shap_values, training_data):
        self.top_k_features = top_k_features
        self.include_shap_values = include_shap_values
        self.training_data = training_data

    def __call__(self, index, pipeline, input_features):
        table = explain_prediction(pipeline, input_features.iloc[index:(index + 1)],
                                   training_data=self.training_data, top_k=self.top_k_features,
                                   include_shap_values=self.include_shap_values)
        table = table.splitlines()
        # Indent the rows of the table to match the indentation of the entire report.
        return ["\t\t" + line + "\n" for line in table] + ["\n\n"]


class _EmptyPredictedValuesMaker:
    """Omits the predicted values section for reports where the user specifies the subset of the input features."""

    def __call__(self, index, y_pred, y_true, scores):
        return [""]


class _ClassificationPredictedValuesMaker:
    """Makes the predicted values section for classification problem best/worst reports."""

    def __init__(self, error_name):
        self.error_name = error_name

    def __call__(self, index, y_pred, y_true, scores):
        pred_value = [f"{col_name}: {pred}" for col_name, pred in
                      zip(y_pred.columns, round(y_pred.iloc[index], 3).tolist())]
        pred_value = "[" + ", ".join(pred_value) + "]"
        true_value = y_true[index]
        prediction_name = "Predicted Probabilities"

        return [f"\t\t{prediction_name}: {pred_value}\n",
                f"\t\tTarget Value: {true_value}\n",
                f"\t\t{self.error_name}: {round(scores[index], 3)}\n\n"]


class _RegressionPredictedValuesMaker:
    """Makes the predicted values section for regression problem best/worst reports."""

    def __init__(self, error_name):
        self.error_name = error_name

    def __call__(self, index, y_pred, y_true, scores):

        return [f"\t\tPredicted Value: {round(y_pred.iloc[index], 3)}\n",
                f"\t\tTarget Value: {round(y_true[index], 3)}\n",
                f"\t\t{self.error_name}: {round(scores[index], 3)}\n\n"]


def explain_predictions_best_worst(pipeline, input_features, y_true, num_to_explain=5, top_k_features=3,
                                   include_shap_values=False, metric=None):
    """Creates a report summarizing the top contributing features for the best and worst points in the dataset as measured by error to true labels.

    XGBoost models and CatBoost multiclass classifiers are not currently supported.

    Arguments:
        pipeline (PipelineBase): Fitted pipeline whose predictions we want to explain with SHAP.
        input_features (pd.DataFrame): Dataframe of input data to evaluate the pipeline on.
        y_true (pd.Series): True labels for the input data.
        num_to_explain (int): How many of the best, worst, random data points to explain.
        top_k_features (int): How many of the highest/lowest contributing feature to include in the table for each
            data point.
        include_shap_values (bool): Whether SHAP values should be included in the table.
        metric (callable): The metric used to identify the best and worst points in the dataset. Function must accept
            the true labels and predicted value or probabilities as the only arguments and lower values
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
                         f"true labels: {y_true.shape[0]} and {input_features.shape[0]}")
    if not metric:
        metric = _DEFAULT_METRICS[pipeline.problem_type]

    table_maker = _SHAPTableMaker(top_k_features, include_shap_values, training_data=input_features)

    try:
        if pipeline.problem_type == ProblemTypes.REGRESSION:
            y_pred = pipeline.predict(input_features)
            errors = metric(y_true, y_pred)
            prediction_results_maker = _RegressionPredictedValuesMaker(metric.__name__)
        else:
            y_pred = pipeline.predict_proba(input_features)
            errors = metric(pipeline._encode_targets(y_true), y_pred)
            prediction_results_maker = _ClassificationPredictedValuesMaker(metric.__name__)
    except Exception as e:
        tb = traceback.format_tb(sys.exc_info()[2])
        raise PipelineScoreError(exceptions={metric.__name__: (e, tb)}, scored_successfully={})

    sorted_scores = errors.sort_values()
    best = sorted_scores.index[:num_to_explain]
    worst = sorted_scores.index[-num_to_explain:]
    report = [pipeline.name + "\n\n", str(pipeline.parameters) + "\n\n"]

    # The trailing space after Best and Worst is intentional. It makes sure there is a space
    # between the prefix and rank for the _HeadingMaker
    for index_list, prefix in zip([best, worst], ["Best ", "Worst "]):
        header_maker = _HeadingMaker(prefix, n_indices=num_to_explain)
        report_section_maker = _ReportSectionMaker(header_maker, prediction_results_maker, table_maker)
        section = report_section_maker.make_report_section(pipeline, input_features, index_list, y_pred,
                                                           y_true, errors)
        report.extend(section)
    return "".join(report)


def explain_predictions(pipeline, input_features, training_data=None, top_k_features=3, include_shap_values=False):
    """Creates a report summarizing the top contributing features for each data point in the input features.

    XGBoost models and CatBoost multiclass classifiers are not currently supported.

    Arguments:
        pipeline (PipelineBase): Fitted pipeline whose predictions we want to explain with SHAP.
        input_features (pd.DataFrame): Dataframe of input data to evaluate the pipeline on.
        training_data (pd.DataFrame): Dataframe of data the pipeline was fit on. This can be omitted for pipelines
            with tree-based estimators.
        top_k_features (int): How many of the highest/lowest contributing feature to include in the table for each
            data point.
        include_shap_values (bool): Whether SHAP values should be included in the table.

    Returns:
        str
    """
    if not (isinstance(input_features, pd.DataFrame) and not input_features.empty):
        raise ValueError("Parameter input_features must be a non-empty dataframe.")
    report = [pipeline.name + "\n\n", str(pipeline.parameters) + "\n\n"]
    header_maker = _HeadingMaker(prefix="", n_indices=input_features.shape[0])
    prediction_results_maker = _EmptyPredictedValuesMaker()
    table_maker = _SHAPTableMaker(top_k_features, include_shap_values, training_data=training_data)
    section_maker = _ReportSectionMaker(header_maker, prediction_results_maker, table_maker)
    report.extend(section_maker.make_report_section(pipeline, input_features, indices=range(input_features.shape[0]),
                                                    y_true=None, y_pred=None, errors=None))
    return "".join(report)
