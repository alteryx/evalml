import pandas as pd

from evalml.pipelines.prediction_explanations._algorithms import (
    _compute_shap_values,
    _normalize_shap_values
)
from evalml.pipelines.prediction_explanations._user_interface import (
    _make_single_prediction_table
)


def _explain_with_shap_values(pipeline, features, top_k=3, training_data=None, include_shap_values=False):
    """Creates table summarizing the top_k positive and top_k negative contributing features to the prediction of a single datapoint.

    Arguments:
        pipeline (PipelineBase): Fitted pipeline whose predictions we want to explain with SHAP.
        features (pd.DataFrame): Dataframe of features - needs to correspond to data the pipeline was fit on.
        top_k (int): How many of the highest/lowest features to include in the table.
        training_data (pd.DataFrame): Training data the pipeline was fit on.
            For non-tree estimators, we need a sample of training data for the KernelSHAP algorithm.
        include_shap_values (bool): Whether the SHAP values should be included in an extra column in the output.

    Returns:
        str: Table
    """
    shap_values = _compute_shap_values(pipeline, features, training_data)
    normalized_shap_values = _normalize_shap_values(shap_values)
    return _make_single_prediction_table(shap_values, normalized_shap_values, top_k, include_shap_values)


def _explain_prediction(pipeline, features, training_data=None):
    """Displays a table summarizing the top 3 positive and rop 3 negative contributing features to the prediction of a single datapoint.

    Arguments:
        pipeline (PipelineBase): Fitted pipeline whose predictions we want to explain.
        features (pd.DataFrame): Dataframe containing the features for a single data point.
        training_data (pd.DataFrame): Training data the pipeline was fit on.
            For non-tree estimators, we need a sample of training data for the KernelSHAP algorithm.

    Returns:
        None: displays a table to std out
    """
    if not (isinstance(features, pd.DataFrame) and features.shape[0] == 1):
        raise ValueError("features must be stored in a dataframe of one row.")
    print(_explain_with_shap_values(pipeline, features, training_data=training_data, top_k=3,
                                    include_shap_values=False))
