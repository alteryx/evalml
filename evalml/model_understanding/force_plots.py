import shap
import numpy as np

from evalml.model_family import ModelFamily
from evalml.model_understanding.prediction_explanations import explain_predictions
from evalml.model_understanding.prediction_explanations._algorithms import _compute_shap_values
from evalml.problem_types import ProblemTypes


def graph_force_plot(pipeline, rows_to_explain, training_data, matplotlib=False):
    """ Function to generate a force plot for a pipeline.

    Args:
        pipeline (PipelineBase): the pipeline to generate the force plot for.
        rows_to_explain (list(int)): a list of the indices of the training_data to explain
        training_data (pandas.DataFrame): the data used to train the pipeline
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
    return force_plot(pipeline, rows_to_explain, training_data, return_data=False, matplotlib=matplotlib)


def force_plot(pipeline, rows_to_explain, training_data, y, return_data=True, matplotlib=False):
    """ Function to generate a force plot for a pipeline.

    Args:
        pipeline (PipelineBase): the pipeline to generate the force plot for.
        rows_to_explain (list(int)): a list of the indices of the training_data to explain
        training_data (pandas.DataFrame): the data used to train the pipeline
        return_data (bool): whether to return a dictionary of force plot data (True) or
            the actual plots (False)
        matplotlib (bool): flag to display the force plot using matplotlib (outside of jupyter)

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
    def gen_force_plot(shap_values, training_data, expected_value, matplotlib):
        """ Helper function to generate a single force plot. """
        # Ensure the training data sample shape matches the shap values shape.
        # training_data_sample = training_data.iloc[:len(shap_values)]
        assert training_data.shape[1] == len(shap_values)
        import pdb; pdb.set_trace()
        shap_plot = shap.force_plot(expected_value, np.array(shap_values), training_data, matplotlib=matplotlib)
        return shap_plot

    if not isinstance(rows_to_explain, list):
        raise TypeError("rows_to_explain should be provided as a list of row index integers!")
    if not all([isinstance(x, int) for x in rows_to_explain]):
        raise TypeError("rows_to_explain should only contain integers!")

    points_to_explain = training_data.iloc[rows_to_explain]

    x = explain_predictions(pipeline, training_data, y, rows_to_explain,
                                   top_k_features=3, include_shap_values=True,
                                   output_format="dict")
    row_explanations = x["explanations"]
    for row_explanation in row_explanations:
        row_exp = row_explanation["explanations"]
        for cls_exp in row_exp:
            cls = cls_exp["class_name"]
            expected_value = cls_exp["expected_value"]
            feature_names = cls_exp["feature_names"]
            shap_values = cls_exp["quantitative_explanation"]
            gen_force_plot(shap_values, training_data[feature_names], expected_value, False)

    # shap_values =
    import pdb; pdb.set_trace()

    shap_plots = []
    # classification returns shap_values as a list with shap values for each class
    if isinstance(shap_values, list):
        expected_values = explainer.expected_value
        # Coerce expected values as catboost/binary doesn't return the same types as the other explainers
        if pipeline.estimator.model_family == ModelFamily.CATBOOST and pipeline.problem_type == ProblemTypes.BINARY:
            expected_values = [0, expected_values]

        for idx, s_v in enumerate(shap_values):
            result = {}
            result["class"] = idx
            force_plot = gen_force_plot(shap_values=s_v, training_data=training_data,
                                        expected_value=expected_values[idx], matplotlib=False)
            if return_data:
                result["data"] = force_plot.data
            else:
                result["force_plot"] = force_plot
            shap_plots.append(result)
    # regression problems return shap values as a numpy array of values
    else:
        result = {}
        result["class"] = "regression"
        force_plot = gen_force_plot(shap_values=shap_values, training_data=training_data,
                                    expected_value=explainer.expected_value, matplotlib=matplotlib)
        if return_data:
            result["data"] = force_plot.data

            import pdb;
            pdb.set_trace()
        else:
            result["force_plot"] = force_plot
        shap_plots.append(result)

    return shap_plots