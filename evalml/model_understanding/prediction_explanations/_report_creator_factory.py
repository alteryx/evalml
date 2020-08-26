from evalml.model_understanding.prediction_explanations._user_interface import (
    _DictClassificationPredictedValues,
    _DictHeading,
    _DictRegressionPredictedValues,
    _DictReportMaker,
    _DictSHAPTable,
    _TextClassificationPredictedValues,
    _TextEmptyPredictedValues,
    _TextHeading,
    _TextRegressionPredictedValues,
    _TextReportMaker,
    _TextSHAPTable
)
from evalml.problem_types import ProblemTypes


def _best_worst_predicted_values_section(data, regression, classification):
    """Get and initialize the predicted values section maker given the data."""
    predicted_values_class = regression if data.pipeline.problem_type == ProblemTypes.REGRESSION else classification
    return predicted_values_class(data.metric.__name__, data.y_pred_values)


def _report_creator_factory(data, report_type, output_format, top_k_features, include_shap_values, num_to_explain=None):
    """Get and initialize the report creator class given the ReportData and parameters passed in by the user.

    Arguments:
        data (_ReportData): Data about the problem (pipeline/predicted values, etc) needed for the report.
        report_type (str): Either "explain_predictions" or "explain_predictions_best_worst"
        output_format (str): Either "text" or "dict" - passed in by user.
        top_k_features (int): How many best/worst features to include in each SHAP table - passed in by user.
        include_shap_values (bool): Whether to include the SHAP values in each SHAP table - passed in by user.
        num_to_explain (int): How many rows to include in the entire report - passed in by user.

    Returns:
        _ReportCreator instance needed to create the desired report.
    """
    if report_type == "explain_predictions" and output_format == "text":
        heading = _TextHeading([""], data.input_features.shape[0])
        predicted_values = _TextEmptyPredictedValues()
        shap_table = _TextSHAPTable(top_k_features, include_shap_values, data.input_features)
        report_maker = _TextReportMaker(heading, predicted_values, shap_table)
    elif report_type == "explain_predictions" and output_format == "dict":
        shap_table = _DictSHAPTable(top_k_features, include_shap_values, data.input_features)
        report_maker = _DictReportMaker(None, None, shap_table)
    elif report_type == "explain_predictions_best_worst" and output_format == "text":
        heading_maker = _TextHeading(["Best ", "Worst "], n_indices=num_to_explain)
        predicted_values = _best_worst_predicted_values_section(data, _TextRegressionPredictedValues,
                                                                _TextClassificationPredictedValues)
        table_maker = _TextSHAPTable(top_k_features, include_shap_values, training_data=data.input_features)
        report_maker = _TextReportMaker(heading_maker, predicted_values, table_maker)
    else:
        heading_maker = _DictHeading(["best", "worst"], n_indices=num_to_explain)
        table_maker = _DictSHAPTable(top_k_features, include_shap_values, training_data=data.input_features)
        predicted_values = _best_worst_predicted_values_section(data, _DictRegressionPredictedValues,
                                                                _DictClassificationPredictedValues)
        report_maker = _DictReportMaker(heading_maker, predicted_values, table_maker)

    return report_maker
