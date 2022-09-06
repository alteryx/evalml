import os
from collections import OrderedDict
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import woodwork as ww
from sklearn.exceptions import NotFittedError

from evalml.model_understanding.visualizations import (
    binary_objective_vs_threshold,
    decision_tree_data_from_estimator,
    decision_tree_data_from_pipeline,
    get_linear_coefficients,
    get_prediction_vs_actual_data,
    get_prediction_vs_actual_over_time_data,
    graph_binary_objective_vs_threshold,
    graph_prediction_vs_actual,
    graph_prediction_vs_actual_over_time,
    graph_t_sne,
    t_sne,
    visualize_decision_tree,
)
from evalml.objectives import CostBenefitMatrix
from evalml.pipelines import (
    DecisionTreeRegressor,
    ElasticNetRegressor,
    LinearRegressor,
    MulticlassClassificationPipeline,
    RegressionPipeline,
    TimeSeriesRegressionPipeline,
)
from evalml.problem_types import ProblemTypes
from evalml.utils import get_random_state, infer_feature_types


@pytest.mark.parametrize("data_type", ["np", "pd", "ww"])
def test_cost_benefit_matrix_vs_threshold(
    data_type,
    X_y_binary,
    logistic_regression_binary_pipeline,
    make_data_type,
):
    X, y = X_y_binary
    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)

    cbm = CostBenefitMatrix(
        true_positive=1,
        true_negative=-1,
        false_positive=-7,
        false_negative=-2,
    )
    logistic_regression_binary_pipeline.fit(X, y)
    original_pipeline_threshold = logistic_regression_binary_pipeline.threshold
    cost_benefit_df = binary_objective_vs_threshold(
        logistic_regression_binary_pipeline,
        X,
        y,
        cbm,
        steps=5,
    )
    assert list(cost_benefit_df.columns) == ["threshold", "score"]
    assert cost_benefit_df.shape == (6, 2)
    assert not cost_benefit_df.isnull().all().all()
    assert logistic_regression_binary_pipeline.threshold == original_pipeline_threshold


@pytest.mark.parametrize("data_type", ["np", "pd", "ww"])
def test_binary_objective_vs_threshold(
    data_type,
    X_y_binary,
    logistic_regression_binary_pipeline,
    make_data_type,
):
    X, y = X_y_binary
    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)

    logistic_regression_binary_pipeline.fit(X, y)

    # test objective with score_needs_proba == True
    with pytest.raises(ValueError, match="Objective `score_needs_proba` must be False"):
        binary_objective_vs_threshold(
            logistic_regression_binary_pipeline,
            X,
            y,
            "Log Loss Binary",
        )

    # test with non-binary objective
    with pytest.raises(
        ValueError,
        match="can only be calculated for binary classification objectives",
    ):
        binary_objective_vs_threshold(
            logistic_regression_binary_pipeline,
            X,
            y,
            "f1 micro",
        )

    # test objective with score_needs_proba == False
    results_df = binary_objective_vs_threshold(
        logistic_regression_binary_pipeline,
        X,
        y,
        "f1",
        steps=5,
    )
    assert list(results_df.columns) == ["threshold", "score"]
    assert results_df.shape == (6, 2)
    assert not results_df.isnull().all().all()


@patch("evalml.pipelines.BinaryClassificationPipeline.score")
def test_binary_objective_vs_threshold_steps(
    mock_score,
    X_y_binary,
    logistic_regression_binary_pipeline,
):
    X, y = X_y_binary
    cbm = CostBenefitMatrix(
        true_positive=1,
        true_negative=-1,
        false_positive=-7,
        false_negative=-2,
    )
    logistic_regression_binary_pipeline.fit(X, y)
    mock_score.return_value = {"Cost Benefit Matrix": 0.2}
    cost_benefit_df = binary_objective_vs_threshold(
        logistic_regression_binary_pipeline,
        X,
        y,
        cbm,
        steps=234,
    )
    mock_score.assert_called()
    assert list(cost_benefit_df.columns) == ["threshold", "score"]
    assert cost_benefit_df.shape == (235, 2)


@pytest.mark.parametrize("data_type", ["np", "pd", "ww"])
@patch("evalml.model_understanding.visualizations.binary_objective_vs_threshold")
def test_graph_binary_objective_vs_threshold(
    mock_cb_thresholds,
    data_type,
    X_y_binary,
    logistic_regression_binary_pipeline,
    make_data_type,
    go,
):

    X, y = X_y_binary
    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)

    cbm = CostBenefitMatrix(
        true_positive=1,
        true_negative=-1,
        false_positive=-7,
        false_negative=-2,
    )

    mock_cb_thresholds.return_value = pd.DataFrame(
        {"threshold": [0, 0.5, 1.0], "score": [100, -20, 5]},
    )

    figure = graph_binary_objective_vs_threshold(
        logistic_regression_binary_pipeline,
        X,
        y,
        cbm,
    )
    assert isinstance(figure, go.Figure)
    data = figure.data[0]
    assert not np.any(np.isnan(data["x"]))
    assert not np.any(np.isnan(data["y"]))
    assert np.array_equal(data["x"], mock_cb_thresholds.return_value["threshold"])
    assert np.array_equal(data["y"], mock_cb_thresholds.return_value["score"])


@patch("evalml.model_understanding.visualizations.jupyter_check")
@patch("evalml.model_understanding.visualizations.import_or_raise")
def test_jupyter_graph_check(
    import_check,
    jupyter_check,
    X_y_binary,
    X_y_regression,
    logistic_regression_binary_pipeline,
):
    X, y = X_y_binary
    X = X.ww.iloc[:20, :5]
    y = y.ww.iloc[:20]
    logistic_regression_binary_pipeline.fit(X, y)
    cbm = CostBenefitMatrix(
        true_positive=1,
        true_negative=-1,
        false_positive=-7,
        false_negative=-2,
    )
    jupyter_check.return_value = True
    with pytest.warns(None) as graph_valid:
        graph_binary_objective_vs_threshold(
            logistic_regression_binary_pipeline,
            X,
            y,
            cbm,
            steps=5,
        )
        assert len(graph_valid) == 0
        import_check.assert_called_with("ipywidgets", warning=True)

    Xr, yr = X_y_regression
    with pytest.warns(None) as graph_valid:
        rs = get_random_state(42)
        y_preds = yr * rs.random(yr.shape)
        graph_prediction_vs_actual(yr, y_preds)
        assert len(graph_valid) == 0
        import_check.assert_called_with("ipywidgets", warning=True)


@pytest.mark.parametrize("data_type", ["np", "pd", "ww"])
def test_get_prediction_vs_actual_data(data_type, make_data_type):
    y_true = np.array([1, 2, 3000, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    y_pred = np.array([5, 4, 2, 8, 6, 6, 5, 1, 7, 2, 1, 3000])

    y_true_in = make_data_type(data_type, y_true)
    y_pred_in = make_data_type(data_type, y_pred)

    with pytest.raises(ValueError, match="Threshold must be positive!"):
        get_prediction_vs_actual_data(y_true_in, y_pred_in, outlier_threshold=-1)

    outlier_loc = [2, 11]
    results = get_prediction_vs_actual_data(
        y_true_in,
        y_pred_in,
        outlier_threshold=2000,
    )
    assert isinstance(results, pd.DataFrame)
    assert np.array_equal(results["prediction"], y_pred)
    assert np.array_equal(results["actual"], y_true)
    for i, value in enumerate(results["outlier"]):
        if i in outlier_loc:
            assert value == "#ffff00"
        else:
            assert value == "#0000ff"

    results = get_prediction_vs_actual_data(y_true_in, y_pred_in)
    assert isinstance(results, pd.DataFrame)
    assert np.array_equal(results["prediction"], y_pred)
    assert np.array_equal(results["actual"], y_true)
    assert (results["outlier"] == "#0000ff").all()


def test_graph_prediction_vs_actual_default(go):

    y_true = [1, 2, 3000, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    y_pred = [5, 4, 2, 8, 6, 6, 5, 1, 7, 2, 1, 3000]

    fig = graph_prediction_vs_actual(y_true, y_pred)
    assert isinstance(fig, type(go.Figure()))
    fig_dict = fig.to_dict()
    assert (
        fig_dict["layout"]["title"]["text"] == "Predicted vs Actual Values Scatter Plot"
    )
    assert fig_dict["layout"]["xaxis"]["title"]["text"] == "Prediction"
    assert fig_dict["layout"]["yaxis"]["title"]["text"] == "Actual"
    assert len(fig_dict["data"]) == 2
    assert fig_dict["data"][0]["name"] == "y = x line"
    assert fig_dict["data"][0]["x"] == fig_dict["data"][0]["y"]
    assert len(fig_dict["data"][1]["x"]) == len(y_true)
    assert fig_dict["data"][1]["marker"]["color"] == "#0000ff"
    assert fig_dict["data"][1]["name"] == "Values"


@pytest.mark.parametrize("data_type", ["pd", "ww"])
def test_graph_prediction_vs_actual(data_type, go):

    y_true = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    y_pred = [5, 4, 3, 8, 6, 3, 5, 9, 7, 12, 1, 2]

    with pytest.raises(ValueError, match="Threshold must be positive!"):
        graph_prediction_vs_actual(y_true, y_pred, outlier_threshold=-1)

    fig = graph_prediction_vs_actual(y_true, y_pred, outlier_threshold=100)
    assert isinstance(fig, type(go.Figure()))
    fig_dict = fig.to_dict()
    assert (
        fig_dict["layout"]["title"]["text"] == "Predicted vs Actual Values Scatter Plot"
    )
    assert fig_dict["layout"]["xaxis"]["title"]["text"] == "Prediction"
    assert fig_dict["layout"]["yaxis"]["title"]["text"] == "Actual"
    assert len(fig_dict["data"]) == 2
    assert fig_dict["data"][1]["marker"]["color"] == "#0000ff"

    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)
    if data_type == "ww":
        y_true = ww.init_series(y_true)
        y_pred = ww.init_series(y_pred)
    fig = graph_prediction_vs_actual(y_true, y_pred, outlier_threshold=6.1)
    assert isinstance(fig, type(go.Figure()))
    fig_dict = fig.to_dict()
    assert (
        fig_dict["layout"]["title"]["text"] == "Predicted vs Actual Values Scatter Plot"
    )
    assert fig_dict["layout"]["xaxis"]["title"]["text"] == "Prediction"
    assert fig_dict["layout"]["yaxis"]["title"]["text"] == "Actual"
    assert len(fig_dict["data"]) == 3
    assert fig_dict["data"][1]["marker"]["color"] == "#0000ff"
    assert len(fig_dict["data"][1]["x"]) == 10
    assert len(fig_dict["data"][1]["y"]) == 10
    assert fig_dict["data"][1]["name"] == "< outlier_threshold"
    assert fig_dict["data"][2]["marker"]["color"] == "#ffff00"
    assert len(fig_dict["data"][2]["x"]) == 2
    assert len(fig_dict["data"][2]["y"]) == 2
    assert fig_dict["data"][2]["name"] == ">= outlier_threshold"


def test_get_prediction_vs_actual_over_time_data(ts_data):
    X, _, y = ts_data()
    X_train, y_train = X.iloc[:30], y.iloc[:30]
    X_test, y_test = X.iloc[30:], y.iloc[30:]

    pipeline = TimeSeriesRegressionPipeline(
        ["DateTime Featurizer", "Elastic Net Regressor"],
        parameters={
            "pipeline": {
                "gap": 0,
                "max_delay": 2,
                "forecast_horizon": 1,
                "time_index": "date",
            },
        },
    )

    pipeline.fit(X_train, y_train)
    results = get_prediction_vs_actual_over_time_data(
        pipeline,
        X_test,
        y_test,
        X_train,
        y_train,
        pd.Series(X_test.index),
    )
    assert isinstance(results, pd.DataFrame)
    assert list(results.columns) == ["dates", "target", "prediction"]


def test_graph_prediction_vs_actual_over_time(ts_data, go):
    X, _, y = ts_data()
    X_train, y_train = X.iloc[:30], y.iloc[:30]
    X_test, y_test = X.iloc[30:], y.iloc[30:]

    pipeline = TimeSeriesRegressionPipeline(
        ["DateTime Featurizer", "Elastic Net Regressor"],
        parameters={
            "pipeline": {
                "gap": 0,
                "max_delay": 2,
                "forecast_horizon": 1,
                "time_index": "date",
            },
        },
    )
    pipeline.fit(X_train, y_train)

    fig = graph_prediction_vs_actual_over_time(
        pipeline,
        X_test,
        y_test,
        X_train,
        y_train,
        pd.Series(X_test.index),
    )

    assert isinstance(fig, go.Figure)
    fig_dict = fig.to_dict()
    assert fig_dict["layout"]["title"]["text"] == "Prediction vs Target over time"
    assert fig_dict["layout"]["xaxis"]["title"]["text"] == "Time"
    assert (
        fig_dict["layout"]["yaxis"]["title"]["text"] == "Target Values and Predictions"
    )
    assert len(fig_dict["data"]) == 2
    assert fig_dict["data"][0]["line"]["color"] == "#1f77b4"
    assert len(fig_dict["data"][0]["x"]) == X_test.shape[0]
    assert not np.isnan(fig_dict["data"][0]["y"]).all()
    assert len(fig_dict["data"][0]["y"]) == X_test.shape[0]
    assert fig_dict["data"][1]["line"]["color"] == "#d62728"
    assert len(fig_dict["data"][1]["x"]) == X_test.shape[0]
    assert len(fig_dict["data"][1]["y"]) == X_test.shape[0]
    assert not np.isnan(fig_dict["data"][1]["y"]).all()


def test_graph_prediction_vs_actual_over_time_value_error():
    class NotTSPipeline:
        problem_type = ProblemTypes.REGRESSION

    error_msg = "graph_prediction_vs_actual_over_time only supports time series regression pipelines! Received regression."
    with pytest.raises(ValueError, match=error_msg):
        graph_prediction_vs_actual_over_time(
            NotTSPipeline(),
            None,
            None,
            None,
            None,
            None,
        )


def test_decision_tree_data_from_estimator_not_fitted(tree_estimators):
    est_class, _ = tree_estimators
    with pytest.raises(
        NotFittedError,
        match="This DecisionTree estimator is not fitted yet. Call 'fit' with "
        "appropriate arguments before using this estimator.",
    ):
        decision_tree_data_from_estimator(est_class)


def test_decision_tree_data_from_estimator_wrong_type(logit_estimator):
    est_logit = logit_estimator
    with pytest.raises(
        ValueError,
        match="Tree structure reformatting is only supported for decision tree estimators",
    ):
        decision_tree_data_from_estimator(est_logit)


def test_decision_tree_data_from_estimator(fitted_tree_estimators):
    est_class, est_reg = fitted_tree_estimators

    formatted_ = decision_tree_data_from_estimator(est_reg)
    tree_ = est_reg._component_obj.tree_

    assert isinstance(formatted_, OrderedDict)
    assert formatted_["Feature"] == f"Testing_{tree_.feature[0]}"
    assert formatted_["Threshold"] == tree_.threshold[0]
    assert all([a == b for a, b in zip(formatted_["Value"][0], tree_.value[0][0])])
    left_child_feature_ = formatted_["Left_Child"]["Feature"]
    right_child_feature_ = formatted_["Right_Child"]["Feature"]
    left_child_threshold_ = formatted_["Left_Child"]["Threshold"]
    right_child_threshold_ = formatted_["Right_Child"]["Threshold"]
    left_child_value_ = formatted_["Left_Child"]["Value"]
    right_child_value_ = formatted_["Right_Child"]["Value"]
    assert left_child_feature_ == f"Testing_{tree_.feature[tree_.children_left[0]]}"
    assert right_child_feature_ == f"Testing_{tree_.feature[tree_.children_right[0]]}"
    assert left_child_threshold_ == tree_.threshold[tree_.children_left[0]]
    assert right_child_threshold_ == tree_.threshold[tree_.children_right[0]]
    # Check that the immediate left and right child of the root node have the correct values
    assert all(
        [
            a == b
            for a, b in zip(
                left_child_value_[0],
                tree_.value[tree_.children_left[0]][0],
            )
        ],
    )
    assert all(
        [
            a == b
            for a, b in zip(
                right_child_value_[0],
                tree_.value[tree_.children_right[0]][0],
            )
        ],
    )


def test_decision_tree_data_from_pipeline_not_fitted():
    mock_pipeline = MulticlassClassificationPipeline(
        component_graph=["Decision Tree Classifier"],
    )
    with pytest.raises(
        NotFittedError,
        match="The DecisionTree estimator associated with this pipeline is not fitted yet. "
        "Call 'fit' with appropriate arguments before using this estimator.",
    ):
        decision_tree_data_from_pipeline(mock_pipeline)


def test_decision_tree_data_from_pipeline_wrong_type():
    mock_pipeline = MulticlassClassificationPipeline(
        component_graph=["Logistic Regression Classifier"],
    )
    with pytest.raises(
        ValueError,
        match="Tree structure reformatting is only supported for decision tree estimators",
    ):
        decision_tree_data_from_pipeline(mock_pipeline)


def test_decision_tree_data_from_pipeline_feature_length(X_y_categorical_regression):
    mock_pipeline = RegressionPipeline(
        component_graph=["One Hot Encoder", "Imputer", "Decision Tree Regressor"],
    )
    X, y = X_y_categorical_regression
    mock_pipeline.fit(X, y)
    assert (
        len(mock_pipeline.input_feature_names[mock_pipeline.estimator.name])
        == mock_pipeline.estimator._component_obj.n_features_
    )


def test_decision_tree_data_from_pipeline(X_y_categorical_regression):
    mock_pipeline = RegressionPipeline(
        component_graph=["One Hot Encoder", "Imputer", "Decision Tree Regressor"],
    )
    X, y = X_y_categorical_regression
    mock_pipeline.fit(X, y)
    formatted_ = decision_tree_data_from_pipeline(mock_pipeline)
    tree_ = mock_pipeline.estimator._component_obj.tree_
    feature_names = mock_pipeline.input_feature_names[mock_pipeline.estimator.name]

    assert isinstance(formatted_, OrderedDict)
    assert formatted_["Feature"] == feature_names[tree_.feature[0]]
    assert formatted_["Threshold"] == tree_.threshold[0]
    assert all([a == b for a, b in zip(formatted_["Value"][0], tree_.value[0][0])])
    left_child_feature_ = formatted_["Left_Child"]["Feature"]
    right_child_feature_ = formatted_["Right_Child"]["Feature"]
    left_child_threshold_ = formatted_["Left_Child"]["Threshold"]
    right_child_threshold_ = formatted_["Right_Child"]["Threshold"]
    left_child_value_ = formatted_["Left_Child"]["Value"]
    right_child_value_ = formatted_["Right_Child"]["Value"]
    assert left_child_feature_ == feature_names[tree_.feature[tree_.children_left[0]]]
    assert right_child_feature_ == feature_names[tree_.feature[tree_.children_right[0]]]
    assert left_child_threshold_ == tree_.threshold[tree_.children_left[0]]
    assert right_child_threshold_ == tree_.threshold[tree_.children_right[0]]
    # Check that the immediate left and right child of the root node have the correct values
    assert all(
        [
            a == b
            for a, b in zip(
                left_child_value_[0],
                tree_.value[tree_.children_left[0]][0],
            )
        ],
    )
    assert all(
        [
            a == b
            for a, b in zip(
                right_child_value_[0],
                tree_.value[tree_.children_right[0]][0],
            )
        ],
    )


def test_visualize_decision_trees_filepath(fitted_tree_estimators, tmpdir):
    import graphviz

    est_class, _ = fitted_tree_estimators
    filepath = os.path.join(str(tmpdir), "invalid", "path", "test.png")

    assert not os.path.exists(filepath)
    with pytest.raises(ValueError, match="Specified filepath is not writeable"):
        visualize_decision_tree(estimator=est_class, filepath=filepath)

    filepath = os.path.join(str(tmpdir), "test_0.png")
    src = visualize_decision_tree(estimator=est_class, filepath=filepath)
    assert os.path.exists(filepath)
    assert src.format == "png"
    assert isinstance(src, graphviz.Source)


def test_visualize_decision_trees_wrong_format(fitted_tree_estimators, tmpdir):
    import graphviz

    est_class, _ = fitted_tree_estimators
    filepath = os.path.join(str(tmpdir), "test_0.xyz")
    with pytest.raises(
        ValueError,
        match=f"Unknown format 'xyz'. Make sure your format is one of the following: "
        f"{graphviz.FORMATS}",
    ):
        visualize_decision_tree(estimator=est_class, filepath=filepath)


def test_visualize_decision_trees_est_wrong_type(logit_estimator, tmpdir):
    est_logit = logit_estimator
    filepath = os.path.join(str(tmpdir), "test_1.png")
    with pytest.raises(
        ValueError,
        match="Tree visualizations are only supported for decision tree estimators",
    ):
        visualize_decision_tree(estimator=est_logit, filepath=filepath)


def test_visualize_decision_trees_max_depth(tree_estimators, tmpdir):
    est_class, _ = tree_estimators
    filepath = os.path.join(str(tmpdir), "test_1.png")
    with pytest.raises(
        ValueError,
        match="Unknown value: '-1'. The parameter max_depth has to be a non-negative integer",
    ):
        visualize_decision_tree(estimator=est_class, max_depth=-1, filepath=filepath)


def test_visualize_decision_trees_not_fitted(tree_estimators, tmpdir):
    est_class, _ = tree_estimators
    filepath = os.path.join(str(tmpdir), "test_1.png")
    with pytest.raises(
        NotFittedError,
        match="This DecisionTree estimator is not fitted yet. Call 'fit' with "
        "appropriate arguments before using this estimator.",
    ):
        visualize_decision_tree(estimator=est_class, max_depth=3, filepath=filepath)


def test_visualize_decision_trees(fitted_tree_estimators, tmpdir):
    import graphviz

    est_class, est_reg = fitted_tree_estimators

    filepath = os.path.join(str(tmpdir), "test_2")
    src = visualize_decision_tree(
        estimator=est_class,
        filled=True,
        max_depth=3,
        rotate=True,
        filepath=filepath,
    )
    assert src.format == "pdf"  # Check that extension defaults to pdf
    assert isinstance(src, graphviz.Source)

    filepath = os.path.join(str(tmpdir), "test_3.pdf")
    src = visualize_decision_tree(estimator=est_reg, filled=True, filepath=filepath)
    assert src.format == "pdf"
    assert isinstance(src, graphviz.Source)

    src = visualize_decision_tree(estimator=est_reg, filled=True, max_depth=2)
    assert src.format == "pdf"
    assert isinstance(src, graphviz.Source)


def test_linear_coefficients_errors():
    dt = DecisionTreeRegressor()

    with pytest.raises(
        ValueError,
        match="Linear coefficients are only available for linear family models",
    ):
        get_linear_coefficients(dt)

    lin = LinearRegressor()

    with pytest.raises(ValueError, match="This linear estimator is not fitted yet."):
        get_linear_coefficients(lin)


@pytest.mark.parametrize("estimator", [LinearRegressor, ElasticNetRegressor])
def test_linear_coefficients_output(estimator):
    X = pd.DataFrame(
        [[1, 2, 3, 5], [3, 5, 2, 1], [5, 2, 2, 2], [3, 2, 3, 3]],
        columns=["First", "Second", "Third", "Fourth"],
    )
    y = pd.Series([2, 1, 3, 4])

    est_ = estimator()
    est_.fit(X, y)

    output_ = get_linear_coefficients(
        est_,
        features=["First", "Second", "Third", "Fourth"],
    )
    assert list(output_.index) == ["Intercept", "Second", "Fourth", "First", "Third"]
    assert output_.shape[0] == X.shape[1] + 1
    assert (
        pd.Series(est_._component_obj.intercept_, index=["Intercept"]).append(
            pd.Series(est_.feature_importance).sort_values(),
        )
        == output_.values
    ).all()


@pytest.mark.parametrize("n_components", [2.0, -2, 0])
def test_t_sne_errors_n_components(n_components):
    X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

    with pytest.raises(
        ValueError,
        match=f"The parameter n_components must be of type integer and greater than 0",
    ):
        t_sne(X, n_components=n_components)


@pytest.mark.parametrize("perplexity", [-2, -1.2])
def test_t_sne_errors_perplexity(perplexity):
    X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

    with pytest.raises(
        ValueError,
        match=f"The parameter perplexity must be non-negative",
    ):
        t_sne(X, perplexity=perplexity)


@pytest.mark.parametrize("data_type", ["np", "pd", "ww"])
def test_t_sne(data_type):
    if data_type == "np":
        X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    elif data_type == "pd":
        X = pd.DataFrame([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    elif data_type == "ww":
        X = pd.DataFrame(np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]]))
        X.ww.init()

    output_ = t_sne(X, n_components=2, perplexity=2, learning_rate=200.0)
    assert isinstance(output_, np.ndarray)


@pytest.mark.parametrize("marker_line_width", [-2, -1.2])
def test_t_sne_errors_marker_line_width(marker_line_width):
    X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

    with pytest.raises(
        ValueError,
        match=f"The parameter marker_line_width must be non-negative",
    ):
        graph_t_sne(X, marker_line_width=marker_line_width)


@pytest.mark.parametrize("marker_size", [-2, -1.2])
def test_t_sne_errors_marker_size(marker_size):
    X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

    with pytest.raises(
        ValueError,
        match=f"The parameter marker_size must be non-negative",
    ):
        graph_t_sne(X, marker_size=marker_size)


@pytest.mark.parametrize("data_type", ["np", "pd", "ww"])
@pytest.mark.parametrize("perplexity", [0, 2.6, 3])
@pytest.mark.parametrize("learning_rate", [100.0, 0.1])
def test_graph_t_sne(data_type, perplexity, learning_rate, go):

    if data_type == "np":
        X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    elif data_type == "pd":
        X = pd.DataFrame([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    elif data_type == "ww":
        X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
        X = infer_feature_types(X)

    for width_, size_ in [(3, 2), (2, 3), (1, 4)]:
        fig = graph_t_sne(
            X,
            n_components=2,
            perplexity=perplexity,
            learning_rate=learning_rate,
            marker_line_width=width_,
            marker_size=size_,
        )
        assert isinstance(fig, go.Figure)
        fig_dict_data = fig.to_dict()["data"][0]
        assert fig_dict_data["marker"]["line"]["width"] == width_
        assert fig_dict_data["marker"]["size"] == size_
        assert fig_dict_data["mode"] == "markers"
        assert fig_dict_data["type"] == "scatter"
