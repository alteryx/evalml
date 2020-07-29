import pandas as pd

from evalml.pipelines.prediction_explanations._algorithms import (
    _compute_shap_values,
    _normalize_shap_values
)
from evalml.pipelines.prediction_explanations._user_interface import (
    _make_single_prediction_table
)


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
    return _make_single_prediction_table(shap_values, normalized_shap_values, top_k, include_shap_values)
