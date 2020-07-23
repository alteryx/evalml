from evalml.pipelines.explanations._algorithms import _compute_shap_values, _normalize_shap_values
from evalml.pipelines.explanations._user_interface import _make_single_prediction_table


def _explain_with_shap_values(pipeline, features, top_k=3, training_data=None, include_shap_values=False):
    """Displays a table summarizing the top_k positive and top_k negative contributing features to the prediction of a single datapoint.

    Arguments:
        pipeline (PipelineBase): Trained pipeline whose predictions we want to explain with SHAP.
        features (pd.DataFrame): Dataframe of features - needs to correspond to data the pipeline was fit on.
        training_data (pd.DataFrame): Training data the pipeline was fit on.
            For non-tree estimators, we need a sample of training data for the KernelSHAP algorithm.
    Returns:
        None: Displays a table to standard out
    """

    shap_values = _compute_shap_values(pipeline, features, training_data)
    normalized_shap_values = _normalize_shap_values(shap_values)
    return _make_single_prediction_table(shap_values, normalized_shap_values, top_k, include_shap_values)

