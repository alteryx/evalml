import shap
import numpy as np

from evalml.model_family import ModelFamily
from evalml.model_understanding.prediction_explanations import explain_predictions
from evalml.model_understanding.prediction_explanations._algorithms import _compute_shap_values
from evalml.problem_types import ProblemTypes


def graph_force_plot(pipeline, rows_to_explain, training_data, y, matplotlib=False):
    """ Function to generate a force plot for a pipeline.

    Args:
        pipeline (PipelineBase): the pipeline to generate the force plot for.
        rows_to_explain (list(int)): a list of the indices of the training_data to explain
        training_data (pandas.DataFrame): the data used to train the pipeline
        y (pandas.Series): target data for the pipeline
        matplotlib (bool): flag to display the force plot using matplotlib (outside of jupyter)

    Returns:
        list(dict(shap.AdditiveForceVisualizer)): list of dictionaries where each dict
            contains the class label and the force plot for classification problems or
            a single dict with the force plot for a regression problem.
            e.x. For single row binary force plots:
                    {"class": 0, "force_plot": AdditiveForceVisualizerObject,
                     "class": 1, "force_plot": AdditiveForceVisualizerObject}
                 For multi row multi-class force plots:
                    {"class": 0, "force_plot": AdditiveForceArrayVisualizerObject,
                     "class": 1, "force_plot": AdditiveForceArrayVisualizerObject,
                     "class": 2, "force_plot": AdditiveForceArrayVisualizerObject}

    Raises:
        TypeError: if rows_to_explain is not a list.
        TypeError: if all values in rows_to_explain aren't integers.
    """

    def gen_force_plot(shap_values, training_data, expected_value, matplotlib):
        """ Helper function to generate a single force plot. """
        # Ensure the training data sample shape matches the shap values shape.
        assert training_data.shape[1] == len(shap_values)
        training_data_sample = training_data.iloc[0]
        shap_plot = shap.force_plot(expected_value, np.array(shap_values), training_data_sample, matplotlib=matplotlib)
        return shap_plot

    shap_plots = force_plot(pipeline, rows_to_explain, training_data, y)
    for cls in shap_plots:
        cls_dict = shap_plots[cls]
        cls_dict["plot"] = gen_force_plot(cls_dict["shap_values"],
                                          training_data[cls_dict["feature_names"]],
                                          cls_dict["expected_value"],
                                          matplotlib=False)

    return shap_plots


def force_plot(pipeline, rows_to_explain, training_data, y):
    """ Function to generate a force plot for a pipeline.

    Args:
        pipeline (PipelineBase): the pipeline to generate the force plot for.
        rows_to_explain (list(int)): a list of the indices of the training_data to explain
        training_data (pandas.DataFrame): the data used to train the pipeline

    Returns:
        list(dict()): list of dictionaries where each dict
            contains the data to generate the force plot for classification problems or
            a single dict with the force plot for a regression problem.
            e.x. For single row binary force plots with return_data == True:
                    {"class": 0, "data": dict,
                     "class": 1, "data": dict}
                 For multi row multi-class force plots with return_data == True:
                    {"class": 0, "data": dict,
                     "class": 1, "data": dict,
                     "class": 2, "data": dict}

    Raises:
        TypeError: if rows_to_explain is not a list.
        TypeError: if all values in rows_to_explain aren't integers.
    """

    if not isinstance(rows_to_explain, list):
        raise TypeError("rows_to_explain should be provided as a list of row index integers!")
    if not all([isinstance(x, int) for x in rows_to_explain]):
        raise TypeError("rows_to_explain should only contain integers!")

    explanations = {}
    prediction_explanations = explain_predictions(pipeline, training_data, y, rows_to_explain,
                            top_k_features=3, include_shap_values=True,
                            output_format="dict")
    row_explanations = prediction_explanations["explanations"]
    for row_explanation in row_explanations:
        row_exp = row_explanation["explanations"]
        for cls_exp in row_exp:
            cls = cls_exp["class_name"] if cls_exp["class_name"] is not None else "regression"
            expected_value = cls_exp["expected_value"]
            feature_names = cls_exp["feature_names"]
            shap_values = cls_exp["quantitative_explanation"]
            explanations[cls] = {"expected_value": expected_value, "feature_names": feature_names,
                               "shap_values": shap_values}

    return explanations
