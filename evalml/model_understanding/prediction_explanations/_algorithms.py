import logging
import warnings
from operator import add

import numpy as np
import shap
from sklearn.utils import check_array

from evalml.model_family.model_family import ModelFamily
from evalml.problem_types import is_binary, is_multiclass, is_regression

logger = logging.getLogger(__name__)


def _create_dictionary(shap_values, feature_names):
    """Creates a mapping from a feature name to a list of SHAP values for all points that were queried.

    Args:
        shap_values (np.ndarray): SHAP values stored in an array of shape (n_datapoints, n_features).
        feature_names (Iterable): Iterable storing the feature names as they are ordered in the dataset.

    Returns:
        dict
    """
    if not isinstance(shap_values, np.ndarray):
        raise ValueError("SHAP values must be stored in a numpy array!")
    shap_values = np.atleast_2d(shap_values)
    mapping = {}
    for feature_name, column_index in zip(feature_names, range(shap_values.shape[1])):
        mapping[feature_name] = shap_values[:, column_index].tolist()
    return mapping


def _compute_shap_values(pipeline, features, training_data=None):
    """Computes SHAP values for each feature.

    Args:
        pipeline (PipelineBase): Trained pipeline whose predictions we want to explain with SHAP.
        features (pd.DataFrame): Dataframe of features - needs to correspond to data the pipeline was fit on.
        training_data (pd.DataFrame): Training data the pipeline was fit on.
            For non-tree estimators, we need a sample of training data for the KernelSHAP algorithm.

    Returns:
        dict or list(dict): For regression problems, a dictionary mapping a feature name to a list of SHAP values.
            For classification problems, returns a list of dictionaries. One for each class.
        float: the expected value if return_expected_value is True.
    """
    estimator = pipeline.estimator
    if estimator.model_family == ModelFamily.BASELINE:
        raise ValueError(
            "You passed in a baseline pipeline. These are simple enough that SHAP values are not needed."
        )

    feature_names = features.columns

    # This is to make sure all dtypes are numeric - SHAP algorithms will complain otherwise.
    # Sklearn components do this under-the-hood so we're not changing the data the model was trained on.
    # Catboost can naturally handle string-encoded categorical features so we don't need to convert to numeric.
    if estimator.model_family != ModelFamily.CATBOOST:
        features = check_array(features.values)

    if estimator.model_family.is_tree_estimator():
        # Use tree_path_dependent to avoid linear runtime with dataset size
        with warnings.catch_warnings(record=True) as ws:
            explainer = shap.TreeExplainer(
                estimator._component_obj, feature_perturbation="tree_path_dependent"
            )
        if ws:
            logger.debug(f"_compute_shap_values TreeExplainer: {ws[0].message}")
        shap_values = explainer.shap_values(features, check_additivity=False)
        # shap only outputs values for positive class for Catboost/Xgboost binary estimators.
        # this modifies the output to match the output format of other binary estimators.
        # Ok to fill values of negative class with zeros since the negative class will get dropped
        # in the UI anyways.
        if (
            estimator.model_family
            in {
                ModelFamily.CATBOOST,
                ModelFamily.XGBOOST,
            }
            and is_binary(pipeline.problem_type)
        ):
            shap_values = [np.zeros(shap_values.shape), shap_values]
    else:
        if training_data is None:
            raise ValueError(
                "You must pass in a value for parameter 'training_data' when the pipeline "
                "does not have a tree-based estimator. "
                f"Current estimator model family is {estimator.model_family}."
            )

        # More than 100 datapoints can negatively impact runtime according to SHAP
        # https://github.com/slundberg/shap/blob/master/shap/explainers/kernel.py#L114
        sampled_training_data_features = shap.sample(training_data, 100)
        sampled_training_data_features = check_array(sampled_training_data_features)
        if is_regression(pipeline.problem_type):
            link_function = "identity"
            decision_function = estimator._component_obj.predict
        else:
            link_function = "logit"
            decision_function = estimator._component_obj.predict_proba
        with warnings.catch_warnings(record=True) as ws:
            explainer = shap.KernelExplainer(
                decision_function, sampled_training_data_features, link_function
            )
            shap_values = explainer.shap_values(features)
        if ws:
            logger.debug(f"_compute_shap_values KernelExplainer: {ws[0].message}")

    if is_multiclass(pipeline.problem_type) or is_regression(pipeline.problem_type):
        expected_value = explainer.expected_value
    elif is_binary(pipeline.problem_type):
        try:
            # Accounts for CatBoost and XGBoost returning expected_value as float
            # for positive class
            expected_value = explainer.expected_value[1]
        except IndexError:
            expected_value = explainer.expected_value

    # classification problem
    if isinstance(shap_values, list):
        mappings = []
        for class_shap_values in shap_values:
            mappings.append(_create_dictionary(class_shap_values, feature_names))
        return (mappings, expected_value)
    # regression problem
    elif isinstance(shap_values, np.ndarray):
        dic = _create_dictionary(shap_values, feature_names)
        return (dic, expected_value)
    else:
        raise ValueError(f"Unknown shap_values datatype {str(type(shap_values))}!")


def _aggreggate_shap_values_dict(values, provenance):
    """Aggregates shap values across features created from a common feature.

    For example, let's say the pipeline has a text featurizer that creates the columns: LSA_1, LSA_2, PolarityScore,
    MeanCharacter, and DiversityScore from a column called "text_feature".

    The values dictionary input to this function will have a key for each of the features created by the text featurizer,
    but it will not have a key for the original "text_feature" column. It will look like this:

    {"LSA_1": [0.2], "LSA_0": [0.3], "PolarityScore": [0.1], "MeanCharacters": [0.05], "DiversityScore": [-0.1], ...}

    After this function, the values dictionary will look like: {"text_feature": [0.55]}

    This aggregation will happen for all features for which we know the provenance/lineage. Other features will
    be left as they are.

    Args:
        values (dict): A mapping of feature names to a list of SHAP values for each data point.
        provenance (dict): A mapping from a feature in the original data to the names of the features that were created
            from that feature.

    Returns:
        dict: Dictionary mapping from feature name to shap values.
    """
    child_to_parent = {}
    for parent_feature, children in provenance.items():
        for child in children:
            if child in values:
                child_to_parent[child] = parent_feature

    agg_values = {}
    for feature_name, shap_list in values.items():
        # Only aggregate features for which we know the parent-feature
        if feature_name in child_to_parent:
            parent = child_to_parent[feature_name]
            if parent not in agg_values:
                agg_values[parent] = [0] * len(shap_list)
            # Elementwise-sum without numpy
            agg_values[parent] = list(map(add, agg_values[parent], shap_list))
        else:
            agg_values[feature_name] = shap_list
    return agg_values


def _aggregate_shap_values(values, provenance):
    """Aggregates shap values across features created from a common feature.

    Args:
        values (dict):  A mapping of feature names to a list of SHAP values for each data point.
        provenance (dict): A mapping from a feature in the original data to the names of the features that were created
            from that feature

    Returns:
        dict or list(dict)
    """
    if isinstance(values, dict):
        return _aggreggate_shap_values_dict(values, provenance)
    else:
        return [
            _aggreggate_shap_values_dict(class_values, provenance)
            for class_values in values
        ]


def _normalize_values_dict(values):
    """Normalizes SHAP values by dividing by the sum of absolute values for each feature.

    Args:
        values (dict): A mapping of feature names to a list of SHAP values for each data point.

    Returns:
        dict

    Examples:
        >>> values = {"a": [1, -1, 3], "b": [3, -2, 0], "c": [-1, 3, 4]}
        >>> normalized_values = _normalize_values_dict(values)
        >>> assert normalized_values == {"a": [1/5, -1/6, 3/7], "b": [3/5, -2/6, 0/7], "c": [-1/5, 3/6, 4/7]}
    """
    # Store in matrix of shape (len(values), n_features)
    feature_names = list(values.keys())
    all_values = np.stack([values[feature_name] for feature_name in feature_names]).T

    if not all_values.any():
        return values

    scaled_values = all_values / np.abs(all_values).sum(axis=1)[:, np.newaxis]

    return {
        feature_name: scaled_values[:, i].tolist()
        for i, feature_name in enumerate(feature_names)
    }


def _normalize_shap_values(values):
    """Normalizes the SHAP values by the absolute value of their sum for each data point.

    Args:
        values (dict or list(dict)): Dictionary mapping feature name to list of values,
            or a list of dictionaries (each mapping a feature name to a list of values).

    Returns:
        dict or list(dict)
    """
    if isinstance(values, dict):
        return _normalize_values_dict(values)
    elif isinstance(values, list):
        return [_normalize_values_dict(class_values) for class_values in values]
    else:
        raise ValueError(
            f"Unsupported data type for _normalize_shap_values: {str(type(values))}."
        )
