import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import label_binarize

from evalml.exceptions import NoPositiveLabelException
from evalml.model_understanding.metrics import (
    confusion_matrix,
    graph_confusion_matrix,
    graph_precision_recall_curve,
    graph_roc_curve,
    normalize_confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from evalml.utils import get_random_state


@pytest.mark.parametrize("test_nullable", [True, False])
@pytest.mark.parametrize("dtype", ["int", "bool"])
@pytest.mark.parametrize("data_type", ["np", "pd", "ww"])
def test_confusion_matrix(data_type, test_nullable, dtype, make_data_type):
    if dtype == "int":
        y_true = np.array([2, 0, 2, 2, 0, 1, 1, 0, 2])
    elif dtype == "bool":
        y_true = np.array([1, 0, 1, 1, 0, 1, 1, 0, 1]).astype(bool)

    if dtype == "int":
        y_predicted = np.array([0, 0, 2, 2, 0, 2, 1, 1, 1])
    elif dtype == "bool":
        y_predicted = np.array([0, 0, 1, 1, 0, 1, 1, 1, 1]).astype(bool)

    y_true = make_data_type(data_type, y_true, nullable=test_nullable)
    y_predicted = make_data_type(data_type, y_predicted)

    conf_mat = confusion_matrix(y_true, y_predicted, normalize_method=None)
    if dtype == "int":
        conf_mat_expected = np.array([[2, 1, 0], [0, 1, 1], [1, 1, 2]])
    elif dtype == "bool":
        conf_mat_expected = np.array([[2, 1], [1, 5]])
    assert np.array_equal(conf_mat_expected, conf_mat.to_numpy())
    assert isinstance(conf_mat, pd.DataFrame)

    conf_mat = confusion_matrix(y_true, y_predicted, normalize_method="all")
    conf_mat_expected = conf_mat_expected / 9.0
    assert np.allclose(conf_mat_expected, conf_mat.to_numpy())
    assert isinstance(conf_mat, pd.DataFrame)

    conf_mat = confusion_matrix(y_true, y_predicted, normalize_method="true")
    if dtype == "int":
        conf_mat_expected = np.array(
            [[2 / 3.0, 1 / 3.0, 0], [0, 0.5, 0.5], [0.25, 0.25, 0.5]],
        )
    elif dtype == "bool":
        conf_mat_expected = np.array([[0.666667, 0.33333], [0.166667, 0.83333]])
    assert np.allclose(conf_mat_expected, conf_mat.to_numpy())
    assert isinstance(conf_mat, pd.DataFrame)

    conf_mat = confusion_matrix(y_true, y_predicted, normalize_method="pred")
    if dtype == "int":
        conf_mat_expected = np.array(
            [[2 / 3.0, 1 / 3.0, 0], [0, 1 / 3.0, 1 / 3.0], [1 / 3.0, 1 / 3.0, 2 / 3.0]],
        )
    elif dtype == "bool":
        conf_mat_expected = np.array([[0.666667, 0.166667], [0.333333, 0.83333]])
    assert np.allclose(conf_mat_expected, conf_mat.to_numpy(), equal_nan=True)
    assert isinstance(conf_mat, pd.DataFrame)

    with pytest.raises(ValueError, match="Invalid value provided"):
        conf_mat = confusion_matrix(
            y_true,
            y_predicted,
            normalize_method="Invalid Option",
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

    precision_expected = np.array([0.5, 0.66666667, 0.5, 1, 1])
    recall_expected = np.array([1, 1, 0.5, 0.5, 0])
    thresholds_expected = np.array([0.1, 0.35, 0.4, 0.8])

    np.testing.assert_almost_equal(precision_expected, precision, decimal=5)
    np.testing.assert_almost_equal(recall_expected, recall, decimal=5)
    np.testing.assert_almost_equal(thresholds_expected, thresholds, decimal=5)


def test_precision_recall_curve_pos_label_idx():
    y_true = pd.Series(np.array([0, 0, 1, 1]))
    y_predict_proba = pd.DataFrame(
        np.array([[0.9, 0.1], [0.6, 0.4], [0.65, 0.35], [0.2, 0.8]]),
    )
    precision_recall_curve_data = precision_recall_curve(
        y_true,
        y_predict_proba,
        pos_label_idx=1,
    )

    precision = precision_recall_curve_data.get("precision")
    recall = precision_recall_curve_data.get("recall")
    thresholds = precision_recall_curve_data.get("thresholds")

    precision_expected = np.array([0.5, 0.66666667, 0.5, 1, 1])
    recall_expected = np.array([1, 1, 0.5, 0.5, 0])
    thresholds_expected = np.array([0.1, 0.35, 0.4, 0.8])
    np.testing.assert_almost_equal(precision_expected, precision, decimal=5)
    np.testing.assert_almost_equal(recall_expected, recall, decimal=5)
    np.testing.assert_almost_equal(thresholds_expected, thresholds, decimal=5)

    y_predict_proba = pd.DataFrame(
        np.array([[0.1, 0.9], [0.4, 0.6], [0.35, 0.65], [0.8, 0.2]]),
    )
    precision_recall_curve_data = precision_recall_curve(
        y_true,
        y_predict_proba,
        pos_label_idx=0,
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
def test_graph_precision_recall_curve(X_y_binary, data_type, make_data_type, go):

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
        fig_dict["data"][0]["x"],
        precision_recall_curve_data["recall"],
    )
    assert np.array_equal(
        fig_dict["data"][0]["y"],
        precision_recall_curve_data["precision"],
    )
    assert fig_dict["data"][0]["name"] == "Precision-Recall (AUC {:06f})".format(
        precision_recall_curve_data["auc_score"],
    )


def test_graph_precision_recall_curve_title_addition(X_y_binary, go):

    X, y_true = X_y_binary
    rs = get_random_state(42)
    y_pred_proba = y_true * rs.random(y_true.shape)
    fig = graph_precision_recall_curve(
        y_true,
        y_pred_proba,
        title_addition="with added title text",
    )
    assert isinstance(fig, type(go.Figure()))
    fig_dict = fig.to_dict()
    assert (
        fig_dict["layout"]["title"]["text"] == "Precision-Recall with added title text"
    )


@pytest.mark.parametrize("test_nullable", [True, False])
@pytest.mark.parametrize("dtype", ["int", "bool"])
@pytest.mark.parametrize("data_type", ["np", "pd", "ww"])
def test_roc_curve_binary(test_nullable, dtype, data_type, make_data_type):
    if dtype == "int":
        y_true = np.array([1, 1, 0, 0])
    elif dtype == "bool":
        y_true = np.array([1, 1, 0, 0]).astype(bool)
    y_predict_proba = np.array([0.1, 0.4, 0.35, 0.8])
    y_true = make_data_type(data_type, y_true, nullable=test_nullable)
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
        ],
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
def test_graph_roc_curve_binary(X_y_binary, data_type, make_data_type, go):

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
        roc_curve_data["auc_score"],
    )
    assert np.array_equal(fig_dict["data"][1]["x"], np.array([0, 1]))
    assert np.array_equal(fig_dict["data"][1]["y"], np.array([0, 1]))
    assert fig_dict["data"][1]["name"] == "Trivial Model (AUC 0.5)"


def test_graph_roc_curve_nans(go):

    one_val_y_zero = np.array([0])
    with pytest.warns(UndefinedMetricWarning):
        fig = graph_roc_curve(one_val_y_zero, one_val_y_zero)
    assert isinstance(fig, type(go.Figure()))
    fig_dict = fig.to_dict()
    assert np.array_equal(fig_dict["data"][0]["x"], np.array([0.0, 1.0]))
    assert np.allclose(
        fig_dict["data"][0]["y"],
        np.array([np.nan, np.nan]),
        equal_nan=True,
    )
    fig1 = graph_roc_curve(
        np.array([np.nan, 1, 1, 0, 1]),
        np.array([0, 0, 0.5, 0.1, 0.9]),
    )
    fig2 = graph_roc_curve(
        np.array([1, 0, 1, 0, 1]),
        np.array([0, np.nan, 0.5, 0.1, 0.9]),
    )
    assert fig1 == fig2


def test_graph_roc_curve_multiclass(binarized_ys, go):

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
            roc_curve_data["auc_score"],
            name=i + 1,
        )
    assert np.array_equal(fig_dict["data"][3]["x"], np.array([0, 1]))
    assert np.array_equal(fig_dict["data"][3]["y"], np.array([0, 1]))
    assert fig_dict["data"][3]["name"] == "Trivial Model (AUC 0.5)"

    with pytest.raises(
        ValueError,
        match="Number of custom class names does not match number of classes",
    ):
        graph_roc_curve(y_true, y_pred_proba, custom_class_names=["one", "two"])


def test_graph_roc_curve_multiclass_custom_class_names(binarized_ys, go):

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
            roc_curve_data["auc_score"],
            name=custom_class_names[i],
        )
    assert np.array_equal(fig_dict["data"][3]["x"], np.array([0, 1]))
    assert np.array_equal(fig_dict["data"][3]["y"], np.array([0, 1]))
    assert fig_dict["data"][3]["name"] == "Trivial Model (AUC 0.5)"


def test_graph_roc_curve_title_addition(X_y_binary, go):

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
def test_graph_confusion_matrix_default(X_y_binary, data_type, make_data_type, go):

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


def test_graph_confusion_matrix_norm_disabled(X_y_binary, go):

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


def test_graph_confusion_matrix_title_addition(X_y_binary, go):

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


@patch("evalml.model_understanding.metrics.jupyter_check")
@patch("evalml.model_understanding.metrics.import_or_raise")
def test_jupyter_graph_check(
    import_check,
    jupyter_check,
    X_y_binary,
    logistic_regression_binary_pipeline,
):
    X, y = X_y_binary
    X = X.ww.iloc[:20, :5]
    y = y.ww.iloc[:20]
    logistic_regression_binary_pipeline.fit(X, y)
    jupyter_check.return_value = False
    with pytest.warns(None) as graph_valid:
        graph_confusion_matrix(y, y)
        assert len(graph_valid) == 0

    jupyter_check.return_value = True
    with pytest.warns(None) as graph_valid:
        rs = get_random_state(42)
        y_pred_proba = y * rs.random(y.shape)
        graph_precision_recall_curve(y, y_pred_proba)
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
