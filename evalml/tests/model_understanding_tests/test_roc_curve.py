import numpy as np
import pytest
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import label_binarize

from evalml.model_understanding import graph_roc_curve, roc_curve
from evalml.utils import get_random_state


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


@pytest.mark.noncore_dependency
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
        roc_curve_data["auc_score"]
    )
    assert np.array_equal(fig_dict["data"][1]["x"], np.array([0, 1]))
    assert np.array_equal(fig_dict["data"][1]["y"], np.array([0, 1]))
    assert fig_dict["data"][1]["name"] == "Trivial Model (AUC 0.5)"


@pytest.mark.noncore_dependency
def test_graph_roc_curve_nans(go):

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


@pytest.mark.noncore_dependency
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


@pytest.mark.noncore_dependency
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
            roc_curve_data["auc_score"], name=custom_class_names[i]
        )
    assert np.array_equal(fig_dict["data"][3]["x"], np.array([0, 1]))
    assert np.array_equal(fig_dict["data"][3]["y"], np.array([0, 1]))
    assert fig_dict["data"][3]["name"] == "Trivial Model (AUC 0.5)"


@pytest.mark.noncore_dependency
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


@pytest.fixture
def binarized_ys(X_y_multi):
    _, y_true = X_y_multi
    rs = get_random_state(42)
    y_tr = label_binarize(y_true, classes=[0, 1, 2])
    y_pred_proba = y_tr * rs.random(y_tr.shape)
    return y_true, y_tr, y_pred_proba
