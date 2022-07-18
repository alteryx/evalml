"""Visualization functions for model understanding."""
import copy
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.manifold import TSNE
from sklearn.tree import export_graphviz

from evalml.model_family import ModelFamily
from evalml.objectives.utils import get_objective
from evalml.problem_types import ProblemTypes
from evalml.utils import import_or_raise, infer_feature_types, jupyter_check


def binary_objective_vs_threshold(pipeline, X, y, objective, steps=100):
    """Computes objective score as a function of potential binary classification decision thresholds for a fitted binary classification pipeline.

    Args:
        pipeline (BinaryClassificationPipeline obj): Fitted binary classification pipeline.
        X (pd.DataFrame): The input data used to compute objective score.
        y (pd.Series): The target labels.
        objective (ObjectiveBase obj, str): Objective used to score.
        steps (int): Number of intervals to divide and calculate objective score at.

    Returns:
        pd.DataFrame: DataFrame with thresholds and the corresponding objective score calculated at each threshold.

    Raises:
        ValueError: If objective is not a binary classification objective.
        ValueError: If objective's `score_needs_proba` is not False.
    """
    objective = get_objective(objective, return_instance=True)
    if not objective.is_defined_for_problem_type(ProblemTypes.BINARY):
        raise ValueError(
            "`binary_objective_vs_threshold` can only be calculated for binary classification objectives",
        )
    if objective.score_needs_proba:
        raise ValueError("Objective `score_needs_proba` must be False")

    pipeline_tmp = copy.copy(pipeline)
    thresholds = np.linspace(0, 1, steps + 1)
    costs = []
    for threshold in thresholds:
        pipeline_tmp.threshold = threshold
        scores = pipeline_tmp.score(X, y, [objective])
        costs.append(scores[objective.name])
    df = pd.DataFrame({"threshold": thresholds, "score": costs})
    return df


def graph_binary_objective_vs_threshold(pipeline, X, y, objective, steps=100):
    """Generates a plot graphing objective score vs. decision thresholds for a fitted binary classification pipeline.

    Args:
        pipeline (PipelineBase or subclass): Fitted pipeline
        X (pd.DataFrame): The input data used to score and compute scores
        y (pd.Series): The target labels
        objective (ObjectiveBase obj, str): Objective used to score, shown on the y-axis of the graph
        steps (int): Number of intervals to divide and calculate objective score at

    Returns:
        plotly.Figure representing the objective score vs. threshold graph generated

    """
    _go = import_or_raise(
        "plotly.graph_objects",
        error_msg="Cannot find dependency plotly.graph_objects",
    )
    if jupyter_check():
        import_or_raise("ipywidgets", warning=True)

    objective = get_objective(objective, return_instance=True)
    df = binary_objective_vs_threshold(pipeline, X, y, objective, steps)
    title = f"{objective.name} Scores vs. Thresholds"
    layout = _go.Layout(
        title={"text": title},
        xaxis={"title": "Threshold", "range": _calculate_axis_range(df["threshold"])},
        yaxis={
            "title": f"{objective.name} Scores vs. Binary Classification Decision Threshold",
            "range": _calculate_axis_range(df["score"]),
        },
    )
    data = []
    data.append(_go.Scatter(x=df["threshold"], y=df["score"], line=dict(width=3)))
    return _go.Figure(layout=layout, data=data)


def get_prediction_vs_actual_data(y_true, y_pred, outlier_threshold=None):
    """Combines y_true and y_pred into a single dataframe and adds a column for outliers. Used in `graph_prediction_vs_actual()`.

    Args:
        y_true (pd.Series, or np.ndarray): The real target values of the data
        y_pred (pd.Series, or np.ndarray): The predicted values outputted by the regression model.
        outlier_threshold (int, float): A positive threshold for what is considered an outlier value. This value is compared to the absolute difference
                                 between each value of y_true and y_pred. Values within this threshold will be blue, otherwise they will be yellow.
                                 Defaults to None.

    Returns:
        pd.DataFrame with the following columns:
                * `prediction`: Predicted values from regression model.
                * `actual`: Real target values.
                * `outlier`: Colors indicating which values are in the threshold for what is considered an outlier value.

    Raises:
        ValueError: If threshold is not positive.
    """
    if outlier_threshold and outlier_threshold <= 0:
        raise ValueError(
            f"Threshold must be positive! Provided threshold is {outlier_threshold}",
        )

    y_true = infer_feature_types(y_true)
    y_pred = infer_feature_types(y_pred)

    predictions = y_pred.reset_index(drop=True)
    actual = y_true.reset_index(drop=True)
    data = pd.concat([pd.Series(predictions), pd.Series(actual)], axis=1)
    data.columns = ["prediction", "actual"]
    if outlier_threshold:
        data["outlier"] = np.where(
            (abs(data["prediction"] - data["actual"]) >= outlier_threshold),
            "#ffff00",
            "#0000ff",
        )
    else:
        data["outlier"] = "#0000ff"
    return data


def graph_prediction_vs_actual(y_true, y_pred, outlier_threshold=None):
    """Generate a scatter plot comparing the true and predicted values. Used for regression plotting.

    Args:
        y_true (pd.Series): The real target values of the data.
        y_pred (pd.Series): The predicted values outputted by the regression model.
        outlier_threshold (int, float): A positive threshold for what is considered an outlier value. This value is compared to the absolute difference
                                 between each value of y_true and y_pred. Values within this threshold will be blue, otherwise they will be yellow.
                                 Defaults to None.

    Returns:
        plotly.Figure representing the predicted vs. actual values graph

    Raises:
        ValueError: If threshold is not positive.
    """
    _go = import_or_raise(
        "plotly.graph_objects",
        error_msg="Cannot find dependency plotly.graph_objects",
    )
    if jupyter_check():
        import_or_raise("ipywidgets", warning=True)

    if outlier_threshold and outlier_threshold <= 0:
        raise ValueError(
            f"Threshold must be positive! Provided threshold is {outlier_threshold}",
        )

    df = get_prediction_vs_actual_data(y_true, y_pred, outlier_threshold)
    data = []

    x_axis = _calculate_axis_range(df["prediction"])
    y_axis = _calculate_axis_range(df["actual"])
    x_y_line = [min(x_axis[0], y_axis[0]), max(x_axis[1], y_axis[1])]
    data.append(
        _go.Scatter(x=x_y_line, y=x_y_line, name="y = x line", line_color="grey"),
    )

    title = "Predicted vs Actual Values Scatter Plot"
    layout = _go.Layout(
        title={"text": title},
        xaxis={"title": "Prediction", "range": x_y_line},
        yaxis={"title": "Actual", "range": x_y_line},
    )

    for color, outlier_group in df.groupby("outlier"):
        if outlier_threshold:
            name = (
                "< outlier_threshold" if color == "#0000ff" else ">= outlier_threshold"
            )
        else:
            name = "Values"
        data.append(
            _go.Scatter(
                x=outlier_group["prediction"],
                y=outlier_group["actual"],
                mode="markers",
                marker=_go.scatter.Marker(color=color),
                name=name,
            ),
        )
    return _go.Figure(layout=layout, data=data)


def _tree_parse(est, feature_names):
    children_left = est.tree_.children_left
    children_right = est.tree_.children_right
    features = est.tree_.feature
    thresholds = est.tree_.threshold
    values = est.tree_.value

    def recurse(i):
        if children_left[i] == children_right[i]:
            return {"Value": values[i]}
        return OrderedDict(
            {
                "Feature": feature_names[features[i]],
                "Threshold": thresholds[i],
                "Value": values[i],
                "Left_Child": recurse(children_left[i]),
                "Right_Child": recurse(children_right[i]),
            },
        )

    return recurse(0)


def decision_tree_data_from_estimator(estimator):
    """Return data for a fitted tree in a restructured format.

    Args:
        estimator (ComponentBase): A fitted DecisionTree-based estimator.

    Returns:
        OrderedDict: An OrderedDict of OrderedDicts describing a tree structure.

    Raises:
        ValueError: If estimator is not a decision tree-based estimator.
        NotFittedError: If estimator is not yet fitted.
    """
    if not estimator.model_family == ModelFamily.DECISION_TREE:
        raise ValueError(
            "Tree structure reformatting is only supported for decision tree estimators",
        )
    if not estimator._is_fitted:
        raise NotFittedError(
            "This DecisionTree estimator is not fitted yet. Call 'fit' with appropriate arguments "
            "before using this estimator.",
        )
    est = estimator._component_obj
    feature_names = estimator.input_feature_names
    return _tree_parse(est, feature_names)


def decision_tree_data_from_pipeline(pipeline_):
    """Return data for a fitted pipeline in a restructured format.

    Args:
        pipeline_ (PipelineBase): A pipeline with a DecisionTree-based estimator.

    Returns:
        OrderedDict: An OrderedDict of OrderedDicts describing a tree structure.

    Raises:
        ValueError: If estimator is not a decision tree-based estimator.
        NotFittedError: If estimator is not yet fitted.
    """
    if not pipeline_.model_family == ModelFamily.DECISION_TREE:
        raise ValueError(
            "Tree structure reformatting is only supported for decision tree estimators",
        )
    if not pipeline_._is_fitted:
        raise NotFittedError(
            "The DecisionTree estimator associated with this pipeline is not fitted yet. Call 'fit' "
            "with appropriate arguments before using this estimator.",
        )
    est = pipeline_.estimator._component_obj
    feature_names = pipeline_.input_feature_names[pipeline_.estimator.name]

    return _tree_parse(est, feature_names)


def visualize_decision_tree(
    estimator,
    max_depth=None,
    rotate=False,
    filled=False,
    filepath=None,
):
    """Generate an image visualizing the decision tree.

    Args:
        estimator (ComponentBase): A fitted DecisionTree-based estimator.
        max_depth (int, optional): The depth to which the tree should be displayed. If set to None (as by default), tree is fully generated.
        rotate (bool, optional): Orient tree left to right rather than top-down.
        filled (bool, optional): Paint nodes to indicate majority class for classification, extremity of values for regression, or purity of node for multi-output.
        filepath (str, optional): Path to where the graph should be saved. If set to None (as by default), the graph will not be saved.

    Returns:
        graphviz.Source: DOT object that can be directly displayed in Jupyter notebooks.

    Raises:
        ValueError: If estimator is not a decision tree-based estimator.
        NotFittedError: If estimator is not yet fitted.
    """
    if not estimator.model_family == ModelFamily.DECISION_TREE:
        raise ValueError(
            "Tree visualizations are only supported for decision tree estimators",
        )
    if max_depth and (not isinstance(max_depth, int) or not max_depth >= 0):
        raise ValueError(
            "Unknown value: '{}'. The parameter max_depth has to be a non-negative integer".format(
                max_depth,
            ),
        )
    if not estimator._is_fitted:
        raise NotFittedError(
            "This DecisionTree estimator is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.",
        )

    est = estimator._component_obj

    graphviz = import_or_raise(
        "graphviz",
        error_msg="Please install graphviz to visualize trees.",
    )

    graph_format = None
    if filepath:
        # Cast to str in case a Path object was passed in
        filepath = str(filepath)
        try:
            f = open(filepath, "w")
            f.close()
        except (IOError, FileNotFoundError):
            raise ValueError(
                ("Specified filepath is not writeable: {}".format(filepath)),
            )
        path_and_name, graph_format = os.path.splitext(filepath)
        if graph_format:
            graph_format = graph_format[1:].lower()  # ignore the dot
            supported_filetypes = graphviz.FORMATS
            if graph_format not in supported_filetypes:
                raise ValueError(
                    (
                        "Unknown format '{}'. Make sure your format is one of the "
                        + "following: {}"
                    ).format(graph_format, supported_filetypes),
                )
        else:
            graph_format = "pdf"  # If the filepath has no extension default to pdf

    dot_data = export_graphviz(
        decision_tree=est,
        max_depth=max_depth,
        rotate=rotate,
        filled=filled,
        feature_names=estimator.input_feature_names,
    )
    source_obj = graphviz.Source(source=dot_data, format=graph_format)
    if filepath:
        source_obj.render(filename=path_and_name, cleanup=True)

    return source_obj


def get_prediction_vs_actual_over_time_data(pipeline, X, y, X_train, y_train, dates):
    """Get the data needed for the prediction_vs_actual_over_time plot.

    Args:
        pipeline (TimeSeriesRegressionPipeline): Fitted time series regression pipeline.
        X (pd.DataFrame): Features used to generate new predictions.
        y (pd.Series): Target values to compare predictions against.
        X_train (pd.DataFrame): Data the pipeline was trained on.
        y_train (pd.Series): Target values for training data.
        dates (pd.Series): Dates corresponding to target values and predictions.

    Returns:
        pd.DataFrame: Predictions vs. time.
    """
    dates = infer_feature_types(dates)
    prediction = pipeline.predict_in_sample(X, y, X_train=X_train, y_train=y_train)

    return pd.DataFrame(
        {
            "dates": dates.reset_index(drop=True),
            "target": y.reset_index(drop=True),
            "prediction": prediction.reset_index(drop=True),
        },
    )


def graph_prediction_vs_actual_over_time(pipeline, X, y, X_train, y_train, dates):
    """Plot the target values and predictions against time on the x-axis.

    Args:
        pipeline (TimeSeriesRegressionPipeline): Fitted time series regression pipeline.
        X (pd.DataFrame): Features used to generate new predictions.
        y (pd.Series): Target values to compare predictions against.
        X_train (pd.DataFrame): Data the pipeline was trained on.
        y_train (pd.Series): Target values for training data.
        dates (pd.Series): Dates corresponding to target values and predictions.

    Returns:
        plotly.Figure: Showing the prediction vs actual over time.

    Raises:
        ValueError: If the pipeline is not a time-series regression pipeline.
    """
    _go = import_or_raise(
        "plotly.graph_objects",
        error_msg="Cannot find dependency plotly.graph_objects",
    )

    if pipeline.problem_type != ProblemTypes.TIME_SERIES_REGRESSION:
        raise ValueError(
            "graph_prediction_vs_actual_over_time only supports time series regression pipelines! "
            f"Received {str(pipeline.problem_type)}.",
        )

    data = get_prediction_vs_actual_over_time_data(
        pipeline,
        X,
        y,
        X_train,
        y_train,
        dates,
    )

    data = [
        _go.Scatter(
            x=data["dates"],
            y=data["target"],
            mode="lines+markers",
            name="Target",
            line=dict(color="#1f77b4"),
        ),
        _go.Scatter(
            x=data["dates"],
            y=data["prediction"],
            mode="lines+markers",
            name="Prediction",
            line=dict(color="#d62728"),
        ),
    ]
    # Let plotly pick the best date format.
    layout = _go.Layout(
        title={"text": "Prediction vs Target over time"},
        xaxis={"title": "Time"},
        yaxis={"title": "Target Values and Predictions"},
    )

    return _go.Figure(data=data, layout=layout)


def get_linear_coefficients(estimator, features=None):
    """Returns a dataframe showing the features with the greatest predictive power for a linear model.

    Args:
        estimator (Estimator): Fitted linear model family estimator.
        features (list[str]): List of feature names associated with the underlying data.

    Returns:
        pd.DataFrame: Displaying the features by importance.

    Raises:
        ValueError: If the model is not a linear model.
        NotFittedError: If the model is not yet fitted.
    """
    if not estimator.model_family == ModelFamily.LINEAR_MODEL:
        raise ValueError(
            "Linear coefficients are only available for linear family models",
        )
    if not estimator._is_fitted:
        raise NotFittedError(
            "This linear estimator is not fitted yet. Call 'fit' with appropriate arguments "
            "before using this estimator.",
        )
    coef_ = estimator.feature_importance
    coef_.name = "Coefficients"
    coef_.index = features
    coef_ = coef_.sort_values()
    coef_ = pd.Series(estimator._component_obj.intercept_, index=["Intercept"]).append(
        coef_,
    )

    return coef_


def t_sne(
    X,
    n_components=2,
    perplexity=30.0,
    learning_rate=200.0,
    metric="euclidean",
    **kwargs,
):
    """Get the transformed output after fitting X to the embedded space using t-SNE.

     Args:
        X (np.ndarray, pd.DataFrame): Data to be transformed. Must be numeric.
        n_components (int, optional): Dimension of the embedded space.
        perplexity (float, optional): Related to the number of nearest neighbors that is used in other manifold learning algorithms. Larger datasets usually require a larger perplexity. Consider selecting a value between 5 and 50.
        learning_rate (float, optional): Usually in the range [10.0, 1000.0]. If the cost function gets stuck in a bad local minimum, increasing the learning rate may help.
        metric (str, optional): The metric to use when calculating distance between instances in a feature array.
        kwargs: Arbitrary keyword arguments.

    Returns:
        np.ndarray (n_samples, n_components): TSNE output.

    Raises:
        ValueError: If specified parameters are not valid values.
    """
    if not isinstance(n_components, int) or not n_components > 0:
        raise ValueError(
            "The parameter n_components must be of type integer and greater than 0",
        )
    if not perplexity >= 0:
        raise ValueError("The parameter perplexity must be non-negative")

    X = infer_feature_types(X)
    t_sne_ = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        metric=metric,
        **kwargs,
    )
    X_new = t_sne_.fit_transform(X)
    return X_new


def graph_t_sne(
    X,
    n_components=2,
    perplexity=30.0,
    learning_rate=200.0,
    metric="euclidean",
    marker_line_width=2,
    marker_size=7,
    **kwargs,
):
    """Plot high dimensional data into lower dimensional space using t-SNE.

    Args:
        X (np.ndarray, pd.DataFrame): Data to be transformed. Must be numeric.
        n_components (int): Dimension of the embedded space. Defaults to 2.
        perplexity (float): Related to the number of nearest neighbors that is used in other manifold learning algorithms. Larger datasets usually require a larger perplexity. Consider selecting a value between 5 and 50. Defaults to 30.
        learning_rate (float): Usually in the range [10.0, 1000.0]. If the cost function gets stuck in a bad local minimum, increasing the learning rate may help. Must be positive. Defaults to 200.
        metric (str): The metric to use when calculating distance between instances in a feature array. The default is "euclidean" which is interpreted as the squared euclidean distance.
        marker_line_width (int): Determines the line width of the marker boundary. Defaults to 2.
        marker_size (int): Determines the size of the marker. Defaults to 7.
        kwargs: Arbitrary keyword arguments.

    Returns:
        plotly.Figure: Figure representing the transformed data.

    Raises:
        ValueError: If marker_line_width or marker_size are not valid values.
    """
    _go = import_or_raise(
        "plotly.graph_objects",
        error_msg="Cannot find dependency plotly.graph_objects",
    )

    if not marker_line_width >= 0:
        raise ValueError("The parameter marker_line_width must be non-negative")
    if not marker_size >= 0:
        raise ValueError("The parameter marker_size must be non-negative")

    X_embedded = t_sne(
        X,
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        metric=metric,
        **kwargs,
    )

    fig = _go.Figure()
    fig.add_trace(_go.Scatter(x=X_embedded[:, 0], y=X_embedded[:, 1], mode="markers"))
    fig.update_traces(
        mode="markers",
        marker_line_width=marker_line_width,
        marker_size=marker_size,
    )
    fig.update_layout(title="t-SNE", yaxis_zeroline=False, xaxis_zeroline=False)

    return fig


def _calculate_axis_range(arr):
    """Helper method to help calculate the appropriate range for an axis based on the data to graph."""
    max_value = arr.max()
    min_value = arr.min()
    margins = abs(max_value - min_value) * 0.05
    return [min_value - margins, max_value + margins]
