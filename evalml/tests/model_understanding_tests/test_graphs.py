import os
import warnings
from collections import OrderedDict
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import woodwork as ww
from sklearn.exceptions import NotFittedError, UndefinedMetricWarning
from sklearn.preprocessing import label_binarize

from evalml.exceptions import NoPositiveLabelException
from evalml.model_understanding.graphs import (
    binary_objective_vs_threshold,
    calculate_permutation_importance,
    confusion_matrix,
    decision_tree_data_from_estimator,
    decision_tree_data_from_pipeline,
    get_linear_coefficients,
    get_prediction_vs_actual_data,
    get_prediction_vs_actual_over_time_data,
    graph_binary_objective_vs_threshold,
    graph_confusion_matrix,
    graph_partial_dependence,
    graph_permutation_importance,
    graph_precision_recall_curve,
    graph_prediction_vs_actual,
    graph_prediction_vs_actual_over_time,
    graph_roc_curve,
    graph_t_sne,
    normalize_confusion_matrix,
    precision_recall_curve,
    roc_curve,
    t_sne,
    visualize_decision_tree,
)
from evalml.objectives import CostBenefitMatrix
from evalml.pipelines import (
    BinaryClassificationPipeline,
    DecisionTreeRegressor,
    ElasticNetRegressor,
    LinearRegressor,
    MulticlassClassificationPipeline,
    RegressionPipeline,
    TimeSeriesRegressionPipeline,
)
from evalml.problem_types import ProblemTypes
from evalml.utils import get_random_state, infer_feature_types


@pytest.fixture
def test_pipeline():
    class TestPipeline(BinaryClassificationPipeline):
        component_graph = [
            "Simple Imputer",
            "One Hot Encoder",
            "Standard Scaler",
            "Logistic Regression Classifier",
        ]

        def __init__(self, parameters):
            super().__init__(self.component_graph, parameters=parameters)

    return TestPipeline(parameters={"Logistic Regression Classifier": {"n_jobs": 1}})


@pytest.mark.parametrize("data_type", ["np", "pd", "ww"])
def test_confusion_matrix(data_type, make_data_type):
    y_true = np.array([2, 0, 2, 2, 0, 1, 1, 0, 2])
    y_predicted = np.array([0, 0, 2, 2, 0, 2, 1, 1, 1])
    y_true = make_data_type(data_type, y_true)
    y_predicted = make_data_type(data_type, y_predicted)

    conf_mat = confusion_matrix(y_true, y_predicted, normalize_method=None)
    conf_mat_expected = np.array([[2, 1, 0], [0, 1, 1], [1, 1, 2]])
    assert np.array_equal(conf_mat_expected, conf_mat.to_numpy())
    assert isinstance(conf_mat, pd.DataFrame)

    conf_mat = confusion_matrix(y_true, y_predicted, normalize_method="all")
    conf_mat_expected = conf_mat_expected / 9.0
    assert np.array_equal(conf_mat_expected, conf_mat.to_numpy())
    assert isinstance(conf_mat, pd.DataFrame)

    conf_mat = confusion_matrix(y_true, y_predicted, normalize_method="true")
    conf_mat_expected = np.array(
        [[2 / 3.0, 1 / 3.0, 0], [0, 0.5, 0.5], [0.25, 0.25, 0.5]]
    )
    assert np.array_equal(conf_mat_expected, conf_mat.to_numpy())
    assert isinstance(conf_mat, pd.DataFrame)

    conf_mat = confusion_matrix(y_true, y_predicted, normalize_method="pred")
    conf_mat_expected = np.array(
        [[2 / 3.0, 1 / 3.0, 0], [0, 1 / 3.0, 1 / 3.0], [1 / 3.0, 1 / 3.0, 2 / 3.0]]
    )
    assert np.allclose(conf_mat_expected, conf_mat.to_numpy(), equal_nan=True)
    assert isinstance(conf_mat, pd.DataFrame)

    with pytest.raises(ValueError, match="Invalid value provided"):
        conf_mat = confusion_matrix(
            y_true, y_predicted, normalize_method="Invalid Option"
        )


@pytest.mark.parametrize("data_type", ["ww", "np", "pd"])
def test_normalize_confusion_matrix(data_type, make_data_type):
    conf_mat = np.array([[2, 3, 0], [0, 1, 1], [1, 0, 2]])
    conf_mat = make_data_type(data_type, conf_mat)

    conf_mat_normalized = normalize_confusion_matrix(conf_mat)
    assert all(conf_mat_normalized.sum(axis=1) == 1.0)
    assert isinstance(conf_mat_normalized, pd.DataFrame)

    conf_mat_normalized = normalize_confusion_matrix(conf_mat, "pred")
    for col_sum in conf_mat_normalized.sum(axis=0):
        assert col_sum == 1.0 or col_sum == 0.0

    conf_mat_normalized = normalize_confusion_matrix(conf_mat, "all")
    assert conf_mat_normalized.sum().sum() == 1.0

    # testing with named pd.DataFrames
    conf_mat_df = pd.DataFrame()
    conf_mat_df["col_1"] = [0, 1, 2]
    conf_mat_df["col_2"] = [0, 0, 3]
    conf_mat_df["col_3"] = [2, 0, 0]
    conf_mat_normalized = normalize_confusion_matrix(conf_mat_df)
    assert all(conf_mat_normalized.sum(axis=1) == 1.0)
    assert list(conf_mat_normalized.columns) == ["col_1", "col_2", "col_3"]

    conf_mat_normalized = normalize_confusion_matrix(conf_mat_df, "pred")
    for col_sum in conf_mat_normalized.sum(axis=0):
        assert col_sum == 1.0 or col_sum == 0.0

    conf_mat_normalized = normalize_confusion_matrix(conf_mat_df, "all")
    assert conf_mat_normalized.sum().sum() == 1.0


@pytest.mark.parametrize("data_type", ["ww", "np", "pd"])
def test_normalize_confusion_matrix_error(data_type, make_data_type):
    conf_mat = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    conf_mat = make_data_type(data_type, conf_mat)

    warnings.simplefilter("default", category=RuntimeWarning)

    with pytest.raises(
        ValueError,
        match='Invalid value provided for "normalize_method": invalid option',
    ):
        normalize_confusion_matrix(conf_mat, normalize_method="invalid option")
    with pytest.raises(ValueError, match="Invalid value provided"):
        normalize_confusion_matrix(conf_mat, normalize_method=None)

    with pytest.raises(ValueError, match="Sum of given axis is 0"):
        normalize_confusion_matrix(conf_mat, "true")
    with pytest.raises(ValueError, match="Sum of given axis is 0"):
        normalize_confusion_matrix(conf_mat, "pred")
    with pytest.raises(ValueError, match="Sum of given axis is 0"):
        normalize_confusion_matrix(conf_mat, "all")


@pytest.mark.parametrize("data_type", ["ww", "pd", "np"])
def test_confusion_matrix_labels(data_type, make_data_type):
    y_true = np.array([True, False, True, True, False, False])
    y_pred = np.array([False, False, True, True, False, False])
    y_true = make_data_type(data_type, y_true)
    y_pred = make_data_type(data_type, y_pred)

    conf_mat = confusion_matrix(y_true=y_true, y_predicted=y_pred)
    labels = [False, True]
    assert np.array_equal(conf_mat.index, labels)
    assert np.array_equal(conf_mat.columns, labels)

    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 1, 1])
    y_true = make_data_type(data_type, y_true)
    y_pred = make_data_type(data_type, y_pred)
    conf_mat = confusion_matrix(y_true=y_true, y_predicted=y_pred)
    labels = [0, 1]
    assert np.array_equal(conf_mat.index, labels)
    assert np.array_equal(conf_mat.columns, labels)

    y_true = np.array(["blue", "red", "blue", "red"])
    y_pred = np.array(["blue", "red", "red", "red"])
    y_true = make_data_type(data_type, y_true)
    y_pred = make_data_type(data_type, y_pred)
    conf_mat = confusion_matrix(y_true=y_true, y_predicted=y_pred)
    labels = ["blue", "red"]
    assert np.array_equal(conf_mat.index, labels)
    assert np.array_equal(conf_mat.columns, labels)

    y_true = np.array(["blue", "red", "red", "red", "orange", "orange"])
    y_pred = np.array(["red", "blue", "blue", "red", "orange", "orange"])
    y_true = make_data_type(data_type, y_true)
    y_pred = make_data_type(data_type, y_pred)
    conf_mat = confusion_matrix(y_true=y_true, y_predicted=y_pred)
    labels = ["blue", "orange", "red"]
    assert np.array_equal(conf_mat.index, labels)
    assert np.array_equal(conf_mat.columns, labels)

    y_true = np.array([0, 1, 2, 1, 2, 1, 2, 3])
    y_pred = np.array([0, 1, 1, 1, 1, 1, 3, 3])
    y_true = make_data_type(data_type, y_true)
    y_pred = make_data_type(data_type, y_pred)
    conf_mat = confusion_matrix(y_true=y_true, y_predicted=y_pred)
    labels = [0, 1, 2, 3]
    assert np.array_equal(conf_mat.index, labels)
    assert np.array_equal(conf_mat.columns, labels)


@pytest.fixture
def binarized_ys(X_y_multi):
    _, y_true = X_y_multi
    rs = get_random_state(42)
    y_tr = label_binarize(y_true, classes=[0, 1, 2])
    y_pred_proba = y_tr * rs.random(y_tr.shape)
    return y_true, y_tr, y_pred_proba


def test_precision_recall_curve_return_type():
    y_true = np.array([0, 0, 1, 1])
    y_predict_proba = np.array([0.1, 0.4, 0.35, 0.8])
    precision_recall_curve_data = precision_recall_curve(y_true, y_predict_proba)
    assert isinstance(precision_recall_curve_data["precision"], np.ndarray)
    assert isinstance(precision_recall_curve_data["recall"], np.ndarray)
    assert isinstance(precision_recall_curve_data["thresholds"], np.ndarray)
    assert isinstance(precision_recall_curve_data["auc_score"], float)


@pytest.mark.parametrize("data_type", ["np", "pd", "pd2d", "li", "ww"])
def test_precision_recall_curve(data_type, make_data_type):
    y_true = np.array([0, 0, 1, 1])
    y_predict_proba = np.array([0.1, 0.4, 0.35, 0.8])
    if data_type == "pd2d":
        data_type = "pd"
        y_predict_proba = np.array([[0.9, 0.1], [0.6, 0.4], [0.65, 0.35], [0.2, 0.8]])
    y_true = make_data_type(data_type, y_true)
    y_predict_proba = make_data_type(data_type, y_predict_proba)

    precision_recall_curve_data = precision_recall_curve(y_true, y_predict_proba)

    precision = precision_recall_curve_data.get("precision")
    recall = precision_recall_curve_data.get("recall")
    thresholds = precision_recall_curve_data.get("thresholds")

    precision_expected = np.array([0.66666667, 0.5, 1, 1])
    recall_expected = np.array([1, 0.5, 0.5, 0])
    thresholds_expected = np.array([0.35, 0.4, 0.8])

    np.testing.assert_almost_equal(precision_expected, precision, decimal=5)
    np.testing.assert_almost_equal(recall_expected, recall, decimal=5)
    np.testing.assert_almost_equal(thresholds_expected, thresholds, decimal=5)


def test_precision_recall_curve_pos_label_idx():
    y_true = pd.Series(np.array([0, 0, 1, 1]))
    y_predict_proba = pd.DataFrame(
        np.array([[0.9, 0.1], [0.6, 0.4], [0.65, 0.35], [0.2, 0.8]])
    )
    precision_recall_curve_data = precision_recall_curve(
        y_true, y_predict_proba, pos_label_idx=1
    )

    precision = precision_recall_curve_data.get("precision")
    recall = precision_recall_curve_data.get("recall")
    thresholds = precision_recall_curve_data.get("thresholds")

    precision_expected = np.array([0.66666667, 0.5, 1, 1])
    recall_expected = np.array([1, 0.5, 0.5, 0])
    thresholds_expected = np.array([0.35, 0.4, 0.8])
    np.testing.assert_almost_equal(precision_expected, precision, decimal=5)
    np.testing.assert_almost_equal(recall_expected, recall, decimal=5)
    np.testing.assert_almost_equal(thresholds_expected, thresholds, decimal=5)

    y_predict_proba = pd.DataFrame(
        np.array([[0.1, 0.9], [0.4, 0.6], [0.35, 0.65], [0.8, 0.2]])
    )
    precision_recall_curve_data = precision_recall_curve(
        y_true, y_predict_proba, pos_label_idx=0
    )
    np.testing.assert_almost_equal(precision_expected, precision, decimal=5)
    np.testing.assert_almost_equal(recall_expected, recall, decimal=5)
    np.testing.assert_almost_equal(thresholds_expected, thresholds, decimal=5)


def test_precision_recall_curve_pos_label_idx_error(make_data_type):
    y_true = np.array([0, 0, 1, 1])
    y_predict_proba = np.array([[0.9, 0.1], [0.6, 0.4], [0.65, 0.35], [0.2, 0.8]])
    with pytest.raises(
        NoPositiveLabelException,
        match="Predicted probabilities of shape \\(4, 2\\) don't contain a column at index 9001",
    ):
        precision_recall_curve(y_true, y_predict_proba, pos_label_idx=9001)


@pytest.mark.parametrize("data_type", ["np", "pd", "ww"])
def test_graph_precision_recall_curve(X_y_binary, data_type, make_data_type):
    go = pytest.importorskip(
        "plotly.graph_objects",
        reason="Skipping plotting test because plotly not installed",
    )
    X, y_true = X_y_binary
    rs = get_random_state(42)
    y_pred_proba = y_true * rs.random(y_true.shape)
    X = make_data_type(data_type, X)
    y_true = make_data_type(data_type, y_true)
    fig = graph_precision_recall_curve(y_true, y_pred_proba)
    assert isinstance(fig, type(go.Figure()))

    fig_dict = fig.to_dict()
    assert fig_dict["layout"]["title"]["text"] == "Precision-Recall"
    assert len(fig_dict["data"]) == 1

    precision_recall_curve_data = precision_recall_curve(y_true, y_pred_proba)
    assert np.array_equal(
        fig_dict["data"][0]["x"], precision_recall_curve_data["recall"]
    )
    assert np.array_equal(
        fig_dict["data"][0]["y"], precision_recall_curve_data["precision"]
    )
    assert fig_dict["data"][0]["name"] == "Precision-Recall (AUC {:06f})".format(
        precision_recall_curve_data["auc_score"]
    )


def test_graph_precision_recall_curve_title_addition(X_y_binary):
    go = pytest.importorskip(
        "plotly.graph_objects",
        reason="Skipping plotting test because plotly not installed",
    )
    X, y_true = X_y_binary
    rs = get_random_state(42)
    y_pred_proba = y_true * rs.random(y_true.shape)
    fig = graph_precision_recall_curve(
        y_true, y_pred_proba, title_addition="with added title text"
    )
    assert isinstance(fig, type(go.Figure()))
    fig_dict = fig.to_dict()
    assert (
        fig_dict["layout"]["title"]["text"] == "Precision-Recall with added title text"
    )


@pytest.mark.parametrize("data_type", ["np", "pd", "ww"])
def test_roc_curve_binary(data_type, make_data_type):
    y_true = np.array([1, 1, 0, 0])
    y_predict_proba = np.array([0.1, 0.4, 0.35, 0.8])
    y_true = make_data_type(data_type, y_true)
    y_predict_proba = make_data_type(data_type, y_predict_proba)

    roc_curve_data = roc_curve(y_true, y_predict_proba)[0]
    fpr_rates = roc_curve_data.get("fpr_rates")
    tpr_rates = roc_curve_data.get("tpr_rates")
    thresholds = roc_curve_data.get("thresholds")
    auc_score = roc_curve_data.get("auc_score")
    fpr_expected = np.array([0, 0.5, 0.5, 1, 1])
    tpr_expected = np.array([0, 0, 0.5, 0.5, 1])
    thresholds_expected = np.array([1.8, 0.8, 0.4, 0.35, 0.1])
    assert np.array_equal(fpr_expected, fpr_rates)
    assert np.array_equal(tpr_expected, tpr_rates)
    assert np.array_equal(thresholds_expected, thresholds)
    assert auc_score == pytest.approx(0.25, 1e-5)
    assert isinstance(roc_curve_data["fpr_rates"], np.ndarray)
    assert isinstance(roc_curve_data["tpr_rates"], np.ndarray)
    assert isinstance(roc_curve_data["thresholds"], np.ndarray)

    y_true = np.array([1, 1, 0, 0])
    y_predict_proba = np.array([[0.9, 0.1], [0.6, 0.4], [0.65, 0.35], [0.2, 0.8]])
    y_predict_proba = make_data_type(data_type, y_predict_proba)
    y_true = make_data_type(data_type, y_true)

    roc_curve_data = roc_curve(y_true, y_predict_proba)[0]
    fpr_rates = roc_curve_data.get("fpr_rates")
    tpr_rates = roc_curve_data.get("tpr_rates")
    thresholds = roc_curve_data.get("thresholds")
    auc_score = roc_curve_data.get("auc_score")
    fpr_expected = np.array([0, 0.5, 0.5, 1, 1])
    tpr_expected = np.array([0, 0, 0.5, 0.5, 1])
    thresholds_expected = np.array([1.8, 0.8, 0.4, 0.35, 0.1])
    assert np.array_equal(fpr_expected, fpr_rates)
    assert np.array_equal(tpr_expected, tpr_rates)
    assert np.array_equal(thresholds_expected, thresholds)
    assert auc_score == pytest.approx(0.25, 1e-5)
    assert isinstance(roc_curve_data["fpr_rates"], np.ndarray)
    assert isinstance(roc_curve_data["tpr_rates"], np.ndarray)
    assert isinstance(roc_curve_data["thresholds"], np.ndarray)


@pytest.mark.parametrize("data_type", ["np", "pd", "ww"])
def test_roc_curve_multiclass(data_type, make_data_type):
    y_true = np.array([1, 2, 0, 0, 2, 1])
    y_predict_proba = np.array(
        [
            [0.33, 0.33, 0.33],
            [0.05, 0.05, 0.90],
            [0.75, 0.15, 0.10],
            [0.8, 0.1, 0.1],
            [0.1, 0.1, 0.8],
            [0.3, 0.4, 0.3],
        ]
    )
    y_true = make_data_type(data_type, y_true)
    y_predict_proba = make_data_type(data_type, y_predict_proba)

    roc_curve_data = roc_curve(y_true, y_predict_proba)
    fpr_expected = [[0, 0, 0, 1], [0, 0, 0, 0.25, 0.75, 1], [0, 0, 0, 0.5, 1]]
    tpr_expected = [[0, 0.5, 1, 1], [0, 0.5, 1, 1, 1, 1], [0, 0.5, 1, 1, 1]]
    thresholds_expected = [
        [1.8, 0.8, 0.75, 0.05],
        [1.4, 0.4, 0.33, 0.15, 0.1, 0.05],
        [1.9, 0.9, 0.8, 0.3, 0.1],
    ]
    auc_expected = [1, 1, 1]

    y_true_unique = y_true
    if data_type == "ww":
        y_true_unique = y_true

    for i in np.unique(y_true_unique):
        fpr_rates = roc_curve_data[i].get("fpr_rates")
        tpr_rates = roc_curve_data[i].get("tpr_rates")
        thresholds = roc_curve_data[i].get("thresholds")
        auc_score = roc_curve_data[i].get("auc_score")
        assert np.array_equal(fpr_expected[i], fpr_rates)
        assert np.array_equal(tpr_expected[i], tpr_rates)
        assert np.array_equal(thresholds_expected[i], thresholds)
        assert auc_expected[i] == pytest.approx(auc_score, 1e-5)
        assert isinstance(roc_curve_data[i]["fpr_rates"], np.ndarray)
        assert isinstance(roc_curve_data[i]["tpr_rates"], np.ndarray)
        assert isinstance(roc_curve_data[i]["thresholds"], np.ndarray)


@pytest.mark.parametrize("data_type", ["np", "pd", "ww"])
def test_graph_roc_curve_binary(X_y_binary, data_type, make_data_type):
    go = pytest.importorskip(
        "plotly.graph_objects",
        reason="Skipping plotting test because plotly not installed",
    )
    X, y_true = X_y_binary
    rs = get_random_state(42)
    y_pred_proba = y_true * rs.random(y_true.shape)
    y_true = make_data_type(data_type, y_true)
    y_pred_proba = make_data_type(data_type, y_pred_proba)

    fig = graph_roc_curve(y_true, y_pred_proba)
    assert isinstance(fig, type(go.Figure()))
    fig_dict = fig.to_dict()
    assert fig_dict["layout"]["title"]["text"] == "Receiver Operating Characteristic"
    assert len(fig_dict["data"]) == 2
    roc_curve_data = roc_curve(y_true, y_pred_proba)[0]
    assert np.array_equal(fig_dict["data"][0]["x"], roc_curve_data["fpr_rates"])
    assert np.array_equal(fig_dict["data"][0]["y"], roc_curve_data["tpr_rates"])
    assert np.allclose(
        np.array(fig_dict["data"][0]["text"]).astype(float),
        np.array(roc_curve_data["thresholds"]).astype(float),
    )
    assert fig_dict["data"][0]["name"] == "Class 1 (AUC {:06f})".format(
        roc_curve_data["auc_score"]
    )
    assert np.array_equal(fig_dict["data"][1]["x"], np.array([0, 1]))
    assert np.array_equal(fig_dict["data"][1]["y"], np.array([0, 1]))
    assert fig_dict["data"][1]["name"] == "Trivial Model (AUC 0.5)"


def test_graph_roc_curve_nans():
    go = pytest.importorskip(
        "plotly.graph_objects",
        reason="Skipping plotting test because plotly not installed",
    )
    one_val_y_zero = np.array([0])
    with pytest.warns(UndefinedMetricWarning):
        fig = graph_roc_curve(one_val_y_zero, one_val_y_zero)
    assert isinstance(fig, type(go.Figure()))
    fig_dict = fig.to_dict()
    assert np.array_equal(fig_dict["data"][0]["x"], np.array([0.0, 1.0]))
    assert np.allclose(
        fig_dict["data"][0]["y"], np.array([np.nan, np.nan]), equal_nan=True
    )
    fig1 = graph_roc_curve(
        np.array([np.nan, 1, 1, 0, 1]), np.array([0, 0, 0.5, 0.1, 0.9])
    )
    fig2 = graph_roc_curve(
        np.array([1, 0, 1, 0, 1]), np.array([0, np.nan, 0.5, 0.1, 0.9])
    )
    assert fig1 == fig2


def test_graph_roc_curve_multiclass(binarized_ys):
    go = pytest.importorskip(
        "plotly.graph_objects",
        reason="Skipping plotting test because plotly not installed",
    )
    y_true, y_tr, y_pred_proba = binarized_ys
    fig = graph_roc_curve(y_true, y_pred_proba)
    assert isinstance(fig, type(go.Figure()))
    fig_dict = fig.to_dict()
    assert fig_dict["layout"]["title"]["text"] == "Receiver Operating Characteristic"
    assert len(fig_dict["data"]) == 4
    for i in range(3):
        roc_curve_data = roc_curve(y_tr[:, i], y_pred_proba[:, i])[0]
        assert np.array_equal(fig_dict["data"][i]["x"], roc_curve_data["fpr_rates"])
        assert np.array_equal(fig_dict["data"][i]["y"], roc_curve_data["tpr_rates"])
        assert np.allclose(
            np.array(fig_dict["data"][i]["text"]).astype(float),
            np.array(roc_curve_data["thresholds"]).astype(float),
        )
        assert fig_dict["data"][i]["name"] == "Class {name} (AUC {:06f})".format(
            roc_curve_data["auc_score"], name=i + 1
        )
    assert np.array_equal(fig_dict["data"][3]["x"], np.array([0, 1]))
    assert np.array_equal(fig_dict["data"][3]["y"], np.array([0, 1]))
    assert fig_dict["data"][3]["name"] == "Trivial Model (AUC 0.5)"

    with pytest.raises(
        ValueError,
        match="Number of custom class names does not match number of classes",
    ):
        graph_roc_curve(y_true, y_pred_proba, custom_class_names=["one", "two"])


def test_graph_roc_curve_multiclass_custom_class_names(binarized_ys):
    go = pytest.importorskip(
        "plotly.graph_objects",
        reason="Skipping plotting test because plotly not installed",
    )
    y_true, y_tr, y_pred_proba = binarized_ys
    custom_class_names = ["one", "two", "three"]
    fig = graph_roc_curve(y_true, y_pred_proba, custom_class_names=custom_class_names)
    assert isinstance(fig, type(go.Figure()))
    fig_dict = fig.to_dict()
    assert fig_dict["layout"]["title"]["text"] == "Receiver Operating Characteristic"
    for i in range(3):
        roc_curve_data = roc_curve(y_tr[:, i], y_pred_proba[:, i])[0]
        assert np.array_equal(fig_dict["data"][i]["x"], roc_curve_data["fpr_rates"])
        assert np.array_equal(fig_dict["data"][i]["y"], roc_curve_data["tpr_rates"])
        assert fig_dict["data"][i]["name"] == "Class {name} (AUC {:06f})".format(
            roc_curve_data["auc_score"], name=custom_class_names[i]
        )
    assert np.array_equal(fig_dict["data"][3]["x"], np.array([0, 1]))
    assert np.array_equal(fig_dict["data"][3]["y"], np.array([0, 1]))
    assert fig_dict["data"][3]["name"] == "Trivial Model (AUC 0.5)"


def test_graph_roc_curve_title_addition(X_y_binary):
    go = pytest.importorskip(
        "plotly.graph_objects",
        reason="Skipping plotting test because plotly not installed",
    )
    X, y_true = X_y_binary
    rs = get_random_state(42)
    y_pred_proba = y_true * rs.random(y_true.shape)
    fig = graph_roc_curve(y_true, y_pred_proba, title_addition="with added title text")
    assert isinstance(fig, type(go.Figure()))
    fig_dict = fig.to_dict()
    assert (
        fig_dict["layout"]["title"]["text"]
        == "Receiver Operating Characteristic with added title text"
    )


@pytest.mark.parametrize("data_type", ["np", "pd", "ww"])
def test_graph_confusion_matrix_default(X_y_binary, data_type, make_data_type):
    go = pytest.importorskip(
        "plotly.graph_objects",
        reason="Skipping plotting test because plotly not installed",
    )
    X, y_true = X_y_binary
    rs = get_random_state(42)
    y_pred = np.round(y_true * rs.random(y_true.shape)).astype(int)
    y_true = make_data_type(data_type, y_true)
    y_pred = make_data_type(data_type, y_pred)

    fig = graph_confusion_matrix(y_true, y_pred)
    assert isinstance(fig, type(go.Figure()))
    fig_dict = fig.to_dict()
    assert (
        fig_dict["layout"]["title"]["text"]
        == 'Confusion matrix, normalized using method "true"'
    )
    assert fig_dict["layout"]["xaxis"]["title"]["text"] == "Predicted Label"
    assert np.all(fig_dict["layout"]["xaxis"]["tickvals"] == np.array([0, 1]))
    assert fig_dict["layout"]["yaxis"]["title"]["text"] == "True Label"
    assert np.all(fig_dict["layout"]["yaxis"]["tickvals"] == np.array([0, 1]))
    assert fig_dict["layout"]["yaxis"]["autorange"] == "reversed"
    heatmap = fig_dict["data"][0]
    conf_mat = confusion_matrix(y_true, y_pred, normalize_method="true")
    conf_mat_unnormalized = confusion_matrix(y_true, y_pred, normalize_method=None)
    assert np.array_equal(heatmap["x"], conf_mat.columns)
    assert np.array_equal(heatmap["y"], conf_mat.columns)
    assert np.array_equal(heatmap["z"], conf_mat)
    assert np.array_equal(heatmap["customdata"], conf_mat_unnormalized)
    assert (
        heatmap["hovertemplate"]
        == "<b>True</b>: %{y}<br><b>Predicted</b>: %{x}<br><b>Normalized Count</b>: %{z}<br><b>Raw Count</b>: %{customdata} <br><extra></extra>"
    )
    annotations = fig.__dict__["_layout_obj"]["annotations"]
    # check that the figure has text annotations for the confusion matrix
    for i in range(len(annotations)):
        assert "text" in annotations[i]


def test_graph_confusion_matrix_norm_disabled(X_y_binary):
    go = pytest.importorskip(
        "plotly.graph_objects",
        reason="Skipping plotting test because plotly not installed",
    )
    X, y_true = X_y_binary
    rs = get_random_state(42)
    y_pred = np.round(y_true * rs.random(y_true.shape)).astype(int)
    fig = graph_confusion_matrix(y_true, y_pred, normalize_method=None)
    assert isinstance(fig, type(go.Figure()))
    fig_dict = fig.to_dict()
    assert fig_dict["layout"]["title"]["text"] == "Confusion matrix"
    assert fig_dict["layout"]["xaxis"]["title"]["text"] == "Predicted Label"
    assert np.all(fig_dict["layout"]["xaxis"]["tickvals"] == np.array([0, 1]))
    assert fig_dict["layout"]["yaxis"]["title"]["text"] == "True Label"
    assert np.all(fig_dict["layout"]["yaxis"]["tickvals"] == np.array([0, 1]))
    assert fig_dict["layout"]["yaxis"]["autorange"] == "reversed"
    heatmap = fig_dict["data"][0]
    conf_mat = confusion_matrix(y_true, y_pred, normalize_method=None)
    conf_mat_normalized = confusion_matrix(y_true, y_pred, normalize_method="true")
    assert np.array_equal(heatmap["x"], conf_mat.columns)
    assert np.array_equal(heatmap["y"], conf_mat.columns)
    assert np.array_equal(heatmap["z"], conf_mat)
    assert np.array_equal(heatmap["customdata"], conf_mat_normalized)
    assert (
        heatmap["hovertemplate"]
        == "<b>True</b>: %{y}<br><b>Predicted</b>: %{x}<br><b>Raw Count</b>: %{z}<br><b>Normalized Count</b>: %{customdata} <br><extra></extra>"
    )


def test_graph_confusion_matrix_title_addition(X_y_binary):
    go = pytest.importorskip(
        "plotly.graph_objects",
        reason="Skipping plotting test because plotly not installed",
    )
    X, y_true = X_y_binary
    rs = get_random_state(42)
    y_pred = np.round(y_true * rs.random(y_true.shape)).astype(int)
    fig = graph_confusion_matrix(y_true, y_pred, title_addition="with added title text")
    assert isinstance(fig, type(go.Figure()))
    fig_dict = fig.to_dict()
    assert (
        fig_dict["layout"]["title"]["text"]
        == 'Confusion matrix with added title text, normalized using method "true"'
    )


def test_graph_permutation_importance(X_y_binary, test_pipeline):
    go = pytest.importorskip(
        "plotly.graph_objects",
        reason="Skipping plotting test because plotly not installed",
    )
    X, y = X_y_binary
    clf = test_pipeline
    clf.fit(X, y)
    fig = graph_permutation_importance(test_pipeline, X, y, "Log Loss Binary")
    assert isinstance(fig, go.Figure)
    fig_dict = fig.to_dict()
    assert (
        fig_dict["layout"]["title"]["text"] == "Permutation Importance<br><sub>"
        "The relative importance of each input feature's overall "
        "influence on the pipelines' predictions, computed using the "
        "permutation importance algorithm.</sub>"
    )
    assert len(fig_dict["data"]) == 1

    perm_importance_data = calculate_permutation_importance(
        clf, X, y, "Log Loss Binary"
    )
    assert np.array_equal(
        fig_dict["data"][0]["x"][::-1], perm_importance_data["importance"].values
    )
    assert np.array_equal(
        fig_dict["data"][0]["y"][::-1], perm_importance_data["feature"]
    )


@patch("evalml.model_understanding.graphs.calculate_permutation_importance")
def test_graph_permutation_importance_show_all_features(mock_perm_importance):
    go = pytest.importorskip(
        "plotly.graph_objects",
        reason="Skipping plotting test because plotly not installed",
    )
    mock_perm_importance.return_value = pd.DataFrame(
        {"feature": ["f1", "f2"], "importance": [0.0, 0.6]}
    )

    figure = graph_permutation_importance(
        test_pipeline, pd.DataFrame(), pd.Series(), "Log Loss Binary"
    )
    assert isinstance(figure, go.Figure)

    data = figure.data[0]
    assert np.any(data["x"] == 0.0)


@patch("evalml.model_understanding.graphs.calculate_permutation_importance")
def test_graph_permutation_importance_threshold(mock_perm_importance):
    go = pytest.importorskip(
        "plotly.graph_objects",
        reason="Skipping plotting test because plotly not installed",
    )
    mock_perm_importance.return_value = pd.DataFrame(
        {"feature": ["f1", "f2"], "importance": [0.0, 0.6]}
    )

    with pytest.raises(
        ValueError,
        match="Provided importance threshold of -0.1 must be greater than or equal to 0",
    ):
        fig = graph_permutation_importance(
            test_pipeline,
            pd.DataFrame(),
            pd.Series(),
            "Log Loss Binary",
            importance_threshold=-0.1,
        )
    fig = graph_permutation_importance(
        test_pipeline,
        pd.DataFrame(),
        pd.Series(),
        "Log Loss Binary",
        importance_threshold=0.5,
    )
    assert isinstance(fig, go.Figure)

    data = fig.data[0]
    assert np.all(data["x"] >= 0.5)


@pytest.mark.parametrize("data_type", ["np", "pd", "ww"])
def test_cost_benefit_matrix_vs_threshold(
    data_type, X_y_binary, logistic_regression_binary_pipeline_class, make_data_type
):
    X, y = X_y_binary
    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)

    cbm = CostBenefitMatrix(
        true_positive=1, true_negative=-1, false_positive=-7, false_negative=-2
    )
    pipeline = logistic_regression_binary_pipeline_class(parameters={})
    pipeline.fit(X, y)
    original_pipeline_threshold = pipeline.threshold
    cost_benefit_df = binary_objective_vs_threshold(pipeline, X, y, cbm, steps=5)
    assert list(cost_benefit_df.columns) == ["threshold", "score"]
    assert cost_benefit_df.shape == (6, 2)
    assert not cost_benefit_df.isnull().all().all()
    assert pipeline.threshold == original_pipeline_threshold


@pytest.mark.parametrize("data_type", ["np", "pd", "ww"])
def test_binary_objective_vs_threshold(
    data_type, X_y_binary, logistic_regression_binary_pipeline_class, make_data_type
):
    X, y = X_y_binary
    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)

    pipeline = logistic_regression_binary_pipeline_class(parameters={})
    pipeline.fit(X, y)

    # test objective with score_needs_proba == True
    with pytest.raises(ValueError, match="Objective `score_needs_proba` must be False"):
        binary_objective_vs_threshold(pipeline, X, y, "Log Loss Binary")

    # test with non-binary objective
    with pytest.raises(
        ValueError, match="can only be calculated for binary classification objectives"
    ):
        binary_objective_vs_threshold(pipeline, X, y, "f1 micro")

    # test objective with score_needs_proba == False
    results_df = binary_objective_vs_threshold(pipeline, X, y, "f1", steps=5)
    assert list(results_df.columns) == ["threshold", "score"]
    assert results_df.shape == (6, 2)
    assert not results_df.isnull().all().all()


@patch("evalml.pipelines.BinaryClassificationPipeline.score")
def test_binary_objective_vs_threshold_steps(
    mock_score, X_y_binary, logistic_regression_binary_pipeline_class
):
    X, y = X_y_binary
    cbm = CostBenefitMatrix(
        true_positive=1, true_negative=-1, false_positive=-7, false_negative=-2
    )
    pipeline = logistic_regression_binary_pipeline_class(parameters={})
    pipeline.fit(X, y)
    mock_score.return_value = {"Cost Benefit Matrix": 0.2}
    cost_benefit_df = binary_objective_vs_threshold(pipeline, X, y, cbm, steps=234)
    mock_score.assert_called()
    assert list(cost_benefit_df.columns) == ["threshold", "score"]
    assert cost_benefit_df.shape == (235, 2)


@pytest.mark.parametrize("data_type", ["np", "pd", "ww"])
@patch("evalml.model_understanding.graphs.binary_objective_vs_threshold")
def test_graph_binary_objective_vs_threshold(
    mock_cb_thresholds,
    data_type,
    X_y_binary,
    logistic_regression_binary_pipeline_class,
    make_data_type,
):
    go = pytest.importorskip(
        "plotly.graph_objects",
        reason="Skipping plotting test because plotly not installed",
    )
    X, y = X_y_binary
    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)

    pipeline = logistic_regression_binary_pipeline_class(parameters={})
    cbm = CostBenefitMatrix(
        true_positive=1, true_negative=-1, false_positive=-7, false_negative=-2
    )

    mock_cb_thresholds.return_value = pd.DataFrame(
        {"threshold": [0, 0.5, 1.0], "score": [100, -20, 5]}
    )

    figure = graph_binary_objective_vs_threshold(pipeline, X, y, cbm)
    assert isinstance(figure, go.Figure)
    data = figure.data[0]
    assert not np.any(np.isnan(data["x"]))
    assert not np.any(np.isnan(data["y"]))
    assert np.array_equal(data["x"], mock_cb_thresholds.return_value["threshold"])
    assert np.array_equal(data["y"], mock_cb_thresholds.return_value["score"])


@patch("evalml.model_understanding.graphs.jupyter_check")
@patch("evalml.model_understanding.graphs.import_or_raise")
def test_jupyter_graph_check(
    import_check, jupyter_check, X_y_binary, X_y_regression, test_pipeline
):
    pytest.importorskip(
        "plotly.graph_objects",
        reason="Skipping plotting test because plotly not installed",
    )
    X, y = X_y_binary
    X = X[:20, :5]
    y = y[:20]
    clf = test_pipeline
    clf.fit(X, y)
    cbm = CostBenefitMatrix(
        true_positive=1, true_negative=-1, false_positive=-7, false_negative=-2
    )
    jupyter_check.return_value = False
    with pytest.warns(None) as graph_valid:
        graph_permutation_importance(test_pipeline, X, y, "log loss binary")
        assert len(graph_valid) == 0
    with pytest.warns(None) as graph_valid:
        graph_confusion_matrix(y, y)
        assert len(graph_valid) == 0

    jupyter_check.return_value = True
    with pytest.warns(None) as graph_valid:
        graph_partial_dependence(clf, X, features=0, grid_resolution=20)
        assert len(graph_valid) == 0
        import_check.assert_called_with("ipywidgets", warning=True)
    with pytest.warns(None) as graph_valid:
        graph_binary_objective_vs_threshold(test_pipeline, X, y, cbm, steps=5)
        assert len(graph_valid) == 0
        import_check.assert_called_with("ipywidgets", warning=True)
    with pytest.warns(None) as graph_valid:
        rs = get_random_state(42)
        y_pred_proba = y * rs.random(y.shape)
        graph_precision_recall_curve(y, y_pred_proba)
        assert len(graph_valid) == 0
        import_check.assert_called_with("ipywidgets", warning=True)
    with pytest.warns(None) as graph_valid:
        graph_permutation_importance(test_pipeline, X, y, "log loss binary")
        assert len(graph_valid) == 0
        import_check.assert_called_with("ipywidgets", warning=True)
    with pytest.warns(None) as graph_valid:
        graph_confusion_matrix(y, y)
        assert len(graph_valid) == 0
        import_check.assert_called_with("ipywidgets", warning=True)
    with pytest.warns(None) as graph_valid:
        rs = get_random_state(42)
        y_pred_proba = y * rs.random(y.shape)
        graph_roc_curve(y, y_pred_proba)
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
        y_true_in, y_pred_in, outlier_threshold=2000
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


def test_graph_prediction_vs_actual_default():
    go = pytest.importorskip(
        "plotly.graph_objects",
        reason="Skipping plotting test because plotly not installed",
    )
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
def test_graph_prediction_vs_actual(data_type):
    go = pytest.importorskip(
        "plotly.graph_objects",
        reason="Skipping plotting test because plotly not installed",
    )
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
    X, y = ts_data
    X_train, y_train = X.iloc[:15], y.iloc[:15]
    X_test, y_test = X.iloc[15:], y.iloc[15:]

    pipeline = TimeSeriesRegressionPipeline(
        ["Elastic Net Regressor"],
        parameters={
            "pipeline": {
                "gap": 0,
                "max_delay": 2,
                "forecast_horizon": 1,
                "date_index": None,
            }
        },
    )

    pipeline.fit(X_train, y_train)
    results = get_prediction_vs_actual_over_time_data(
        pipeline, X_test, y_test, X_train, y_train, pd.Series(X_test.index)
    )
    assert isinstance(results, pd.DataFrame)
    assert list(results.columns) == ["dates", "target", "prediction"]


def test_graph_prediction_vs_actual_over_time(ts_data):
    go = pytest.importorskip(
        "plotly.graph_objects",
        reason="Skipping plotting test because plotly not installed",
    )

    X, y = ts_data
    X_train, y_train = X.iloc[:15], y.iloc[:15]
    X_test, y_test = X.iloc[15:], y.iloc[15:]

    pipeline = TimeSeriesRegressionPipeline(
        ["Elastic Net Regressor"],
        parameters={
            "pipeline": {
                "gap": 0,
                "max_delay": 2,
                "forecast_horizon": 1,
                "date_index": None,
            }
        },
    )
    pipeline.fit(X_train, y_train)

    fig = graph_prediction_vs_actual_over_time(
        pipeline, X_test, y_test, X_train, y_train, pd.Series(X_test.index)
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
    pytest.importorskip(
        "plotly.graph_objects",
        reason="Skipping plotting test because plotly not installed",
    )

    class NotTSPipeline:
        problem_type = ProblemTypes.REGRESSION

    error_msg = "graph_prediction_vs_actual_over_time only supports time series regression pipelines! Received regression."
    with pytest.raises(ValueError, match=error_msg):
        graph_prediction_vs_actual_over_time(
            NotTSPipeline(), None, None, None, None, None
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
                left_child_value_[0], tree_.value[tree_.children_left[0]][0]
            )
        ]
    )
    assert all(
        [
            a == b
            for a, b in zip(
                right_child_value_[0], tree_.value[tree_.children_right[0]][0]
            )
        ]
    )


def test_decision_tree_data_from_pipeline_not_fitted():
    mock_pipeline = MulticlassClassificationPipeline(
        component_graph=["Decision Tree Classifier"]
    )
    with pytest.raises(
        NotFittedError,
        match="The DecisionTree estimator associated with this pipeline is not fitted yet. "
        "Call 'fit' with appropriate arguments before using this estimator.",
    ):
        decision_tree_data_from_pipeline(mock_pipeline)


def test_decision_tree_data_from_pipeline_wrong_type():
    mock_pipeline = MulticlassClassificationPipeline(
        component_graph=["Logistic Regression Classifier"]
    )
    with pytest.raises(
        ValueError,
        match="Tree structure reformatting is only supported for decision tree estimators",
    ):
        decision_tree_data_from_pipeline(mock_pipeline)


def test_decision_tree_data_from_pipeline_feature_length(X_y_categorical_regression):
    mock_pipeline = RegressionPipeline(
        component_graph=["One Hot Encoder", "Imputer", "Decision Tree Regressor"]
    )
    X, y = X_y_categorical_regression
    mock_pipeline.fit(X, y)
    assert (
        len(mock_pipeline.input_feature_names[mock_pipeline.estimator.name])
        == mock_pipeline.estimator._component_obj.n_features_
    )


def test_decision_tree_data_from_pipeline(X_y_categorical_regression):
    mock_pipeline = RegressionPipeline(
        component_graph=["One Hot Encoder", "Imputer", "Decision Tree Regressor"]
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
                left_child_value_[0], tree_.value[tree_.children_left[0]][0]
            )
        ]
    )
    assert all(
        [
            a == b
            for a, b in zip(
                right_child_value_[0], tree_.value[tree_.children_right[0]][0]
            )
        ]
    )


def test_visualize_decision_trees_filepath(fitted_tree_estimators, tmpdir):
    graphviz = pytest.importorskip(
        "graphviz", reason="Skipping visualizing test because graphviz not installed"
    )
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
    graphviz = pytest.importorskip(
        "graphviz", reason="Skipping visualizing test because graphviz not installed"
    )
    est_class, _ = fitted_tree_estimators
    filepath = os.path.join(str(tmpdir), "test_0.xyz")
    with pytest.raises(
        ValueError,
        match=f"Unknown format 'xyz'. Make sure your format is one of the following: "
        f"{graphviz.backend.FORMATS}",
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
    graphviz = pytest.importorskip(
        "graphviz", reason="Skipping visualizing test because graphviz not installed"
    )
    est_class, est_reg = fitted_tree_estimators

    filepath = os.path.join(str(tmpdir), "test_2")
    src = visualize_decision_tree(
        estimator=est_class, filled=True, max_depth=3, rotate=True, filepath=filepath
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
        est_, features=["First", "Second", "Third", "Fourth"]
    )
    assert (output_.index == ["Intercept", "Second", "Fourth", "First", "Third"]).all()
    assert output_.shape[0] == X.shape[1] + 1
    assert (
        pd.Series(est_._component_obj.intercept_, index=["Intercept"]).append(
            pd.Series(est_.feature_importance).sort_values()
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
        ValueError, match=f"The parameter perplexity must be non-negative"
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

    output_ = t_sne(X, n_components=2, perplexity=50, learning_rate=200.0)
    assert isinstance(output_, np.ndarray)


@pytest.mark.parametrize("marker_line_width", [-2, -1.2])
def test_t_sne_errors_marker_line_width(marker_line_width, has_minimal_dependencies):
    if has_minimal_dependencies:
        pytest.skip("Skipping plotting test because plotly not installed")
    X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

    with pytest.raises(
        ValueError, match=f"The parameter marker_line_width must be non-negative"
    ):
        graph_t_sne(X, marker_line_width=marker_line_width)


@pytest.mark.parametrize("marker_size", [-2, -1.2])
def test_t_sne_errors_marker_size(marker_size, has_minimal_dependencies):
    if has_minimal_dependencies:
        pytest.skip("Skipping plotting test because plotly not installed")
    X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

    with pytest.raises(
        ValueError, match=f"The parameter marker_size must be non-negative"
    ):
        graph_t_sne(X, marker_size=marker_size)


@pytest.mark.parametrize("data_type", ["np", "pd", "ww"])
@pytest.mark.parametrize("perplexity", [0, 4.6, 100])
@pytest.mark.parametrize("learning_rate", [100.0, -15, 0])
def test_graph_t_sne(data_type, perplexity, learning_rate):
    go = pytest.importorskip(
        "plotly.graph_objects",
        reason="Skipping plotting test because plotly not installed",
    )
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
