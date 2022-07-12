"""Force plots."""
import numpy as np
import shap
from shap import initjs

from evalml.model_understanding.prediction_explanations import explain_predictions
from evalml.utils import jupyter_check


def graph_force_plot(pipeline, rows_to_explain, training_data, y, matplotlib=False):
    """Function to generate force plots for the desired rows of the training data.

    Args:
        pipeline (PipelineBase): The pipeline to generate the force plot for.
        rows_to_explain (list[int]): A list of the indices indicating which of the rows of
            the training_data to explain.
        training_data (pandas.DataFrame): The data used to train the pipeline.
        y (pandas.Series): The target data for the pipeline.
        matplotlib (bool): flag to display the force plot using matplotlib (outside of jupyter)
            Defaults to False.

    Returns:
        list[dict[shap.AdditiveForceVisualizer]]: The same as force_plot(), but with an additional
            key in each dictionary for the plot.
    """

    def gen_force_plot(shap_values, training_data, expected_value, matplotlib):
        """Helper function to generate a single force plot."""
        shap_plot = shap.force_plot(
            expected_value,
            np.array(shap_values),
            training_data,
            matplotlib=matplotlib,
        )
        return shap_plot

    if jupyter_check():
        initjs()

    shap_plots = force_plot(pipeline, rows_to_explain, training_data, y)
    for ix, row in enumerate(shap_plots):
        row_id = rows_to_explain[ix]
        for cls in row:
            cls_dict = row[cls]
            cls_dict["plot"] = gen_force_plot(
                cls_dict["shap_values"],
                training_data[cls_dict["feature_names"]].iloc[row_id],
                cls_dict["expected_value"],
                matplotlib=matplotlib,
            )
    return shap_plots


def force_plot(pipeline, rows_to_explain, training_data, y):
    """Function to generate the data required to build a force plot.

    Args:
        pipeline (PipelineBase): The pipeline to generate the force plot for.
        rows_to_explain (list[int]): A list of the indices of the training_data to explain.
        training_data (pandas.DataFrame): The data used to train the pipeline.
        y (pandas.Series): The target data.

    Returns:
        list[dict]: list of dictionaries where each dict contains force plot data.  Each dictionary
            entry represents the explanations for a single row.

            For single row binary force plots:
                [{'malignant': {'expected_value': 0.37,
                                'feature_names': ['worst concave points', 'worst perimeter', 'worst radius'],
                                'shap_values': [0.09, 0.09, 0.08],
                                'plot': AdditiveForceVisualizer}]

            For two row binary force plots:
                [{'malignant': {'expected_value': 0.37,
                                'feature_names': ['worst concave points', 'worst perimeter', 'worst radius'],
                                'shap_values': [0.09, 0.09, 0.08],
                                'plot': AdditiveForceVisualizer},
                {'malignant': {'expected_value': 0.29,
                                'feature_names': ['worst concave points', 'worst perimeter', 'worst radius'],
                                'shap_values': [0.05, 0.03, 0.02],
                                'plot': AdditiveForceVisualizer}]

    Raises:
        TypeError: If rows_to_explain is not a list.
        TypeError: If all values in rows_to_explain aren't integers.
    """
    if not isinstance(rows_to_explain, list):
        raise TypeError(
            "rows_to_explain should be provided as a list of row index integers!",
        )
    if not all([isinstance(x, int) for x in rows_to_explain]):
        raise TypeError("rows_to_explain should only contain integers!")

    explanations = []
    prediction_explanations = explain_predictions(
        pipeline,
        training_data,
        y,
        rows_to_explain,
        top_k_features=len(training_data.columns),
        include_explainer_values=True,
        output_format="dict",
    )
    row_explanations = prediction_explanations["explanations"]
    for row_explanation in row_explanations:
        row_exp = row_explanation["explanations"]
        row_exp_dict = {}
        for cls_exp in row_exp:
            cls = (
                cls_exp["class_name"]
                if cls_exp["class_name"] is not None
                else "regression"
            )
            expected_value = cls_exp["expected_value"]
            feature_names = cls_exp["feature_names"]
            shap_values = cls_exp["quantitative_explanation"]
            row_exp_dict[cls] = {
                "expected_value": expected_value,
                "feature_names": feature_names,
                "shap_values": shap_values,
            }
        explanations.append(row_exp_dict)
    return explanations
