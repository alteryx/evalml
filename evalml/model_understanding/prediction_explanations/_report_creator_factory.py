from evalml.model_understanding.prediction_explanations._user_interface import (
    _ClassificationPredictedValues,
    _ExplanationTable,
    _Heading,
    _RegressionPredictedValues,
    _ReportMaker,
)
from evalml.problem_types import is_regression


def _best_worst_predicted_values_section(data, regression, classification):
    """Get and initialize the predicted values section maker given the data."""
    predicted_values_class = (
        regression if is_regression(data.pipeline.problem_type) else classification
    )
    return predicted_values_class(data.metric.__name__, data.y_pred_values)


def _report_creator_factory(
    data,
    report_type,
    output_format,
    top_k_features,
    include_explainer_values,
    include_expected_value,
    num_to_explain=None,
    algorithm="shap",
):
    """Get and initialize the report creator class given the ReportData and parameters passed in by the user.

    Args:
        data (_ReportData): Data about the problem (pipeline/predicted values, etc) needed for the report.
        report_type (str): Either "explain_predictions" or "explain_predictions_best_worst"
        output_format (str): Either "text" or "dict" - passed in by user.
        top_k_features (int): How many best/worst features to include in each explanation table - passed in by user.
        include_explainer_values (bool): Whether to include the explanation values in each explanation table - passed in by user.
        include_expected_value (bool): Whether the expected value should be included in the table - passed in by user.
        num_to_explain (int): How many rows to include in the entire report - passed in by user.
        algorithm (str): Algorithm to use while generating top contributing features, one of "shap" or "lime". Defaults to "shap".

    Returns:
        _ReportCreator method needed to create the desired report.
    """
    if report_type == "explain_predictions" and output_format == "text":
        heading = _Heading([""], len(data.index_list))
        explanation_table = _ExplanationTable(
            top_k_features,
            include_explainer_values,
            algorithm,
        )
        report_maker = _ReportMaker(heading, None, explanation_table).make_text
    elif report_type == "explain_predictions" and output_format == "dict":
        explanation_table = _ExplanationTable(
            top_k_features,
            include_explainer_values,
            algorithm,
        )
        report_maker = _ReportMaker(None, None, explanation_table).make_dict
    elif report_type == "explain_predictions" and output_format == "dataframe":
        explanation_table = _ExplanationTable(
            top_k_features,
            include_explainer_values,
            algorithm,
        )
        report_maker = _ReportMaker(None, None, explanation_table).make_dataframe
    elif report_type == "explain_predictions_best_worst" and output_format == "text":
        heading_maker = _Heading(["Best ", "Worst "], n_indices=num_to_explain)
        predicted_values = _best_worst_predicted_values_section(
            data,
            _RegressionPredictedValues,
            _ClassificationPredictedValues,
        )
        table_maker = _ExplanationTable(
            top_k_features,
            include_explainer_values,
            algorithm,
        )
        report_maker = _ReportMaker(
            heading_maker,
            predicted_values,
            table_maker,
        ).make_text
    elif (
        report_type == "explain_predictions_best_worst" and output_format == "dataframe"
    ):
        heading_maker = _Heading(["best", "worst"], n_indices=num_to_explain)
        table_maker = _ExplanationTable(
            top_k_features,
            include_explainer_values,
            algorithm,
        )
        predicted_values = _best_worst_predicted_values_section(
            data,
            _RegressionPredictedValues,
            _ClassificationPredictedValues,
        )
        report_maker = _ReportMaker(
            heading_maker,
            predicted_values,
            table_maker,
        ).make_dataframe
    else:
        heading_maker = _Heading(["best", "worst"], n_indices=num_to_explain)
        table_maker = _ExplanationTable(
            top_k_features,
            include_explainer_values,
            algorithm,
        )
        predicted_values = _best_worst_predicted_values_section(
            data,
            _RegressionPredictedValues,
            _ClassificationPredictedValues,
        )
        report_maker = _ReportMaker(
            heading_maker,
            predicted_values,
            table_maker,
        ).make_dict

    return report_maker
