import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from evalml.model_understanding import find_confusion_matrix_per_thresholds
from evalml.model_understanding.decision_boundary import (
    _accuracy,
    _balanced_accuracy,
    _f1,
    _find_confusion_matrix_objective_threshold,
    _find_data_between_ranges,
    _precision,
    _recall,
)
from evalml.objectives import AccuracyBinary


@pytest.mark.parametrize(
    "val_list,expected_val",
    [
        ([0, 0, 100, 100], 0.0),
        ([100, 0, 0, 100], 0.5),
        ([50, 50, 50, 50], 0.5),
        ([40, 20, 10, 30], 0.6),
    ],
)
def test_accuracy(val_list, expected_val):
    val = _accuracy(val_list)
    assert val == expected_val


@pytest.mark.parametrize(
    "val_list,expected_val",
    [
        ([0, 0, 100, 100], 0.0),
        ([100, 0, 0, 100], 0.25),
        ([50, 50, 50, 50], 0.5),
        ([40, 20, 10, 30], 13 / 21),
    ],
)
def test_balanced_accuracy(val_list, expected_val):
    val = _balanced_accuracy(val_list)
    assert val == expected_val


@pytest.mark.parametrize(
    "val_list,expected_val",
    [
        ([0, 0, 100, 100], 0.0),
        ([100, 0, 0, 100], 0.5),
        ([50, 50, 50, 50], 0.5),
        ([40, 20, 10, 30], 4 / 7),
    ],
)
def test_recall(val_list, expected_val):
    val = _recall(val_list)
    assert val == expected_val


@pytest.mark.parametrize(
    "val_list,expected_val",
    [
        ([0, 0, 100, 100], 0.0),
        ([100, 0, 0, 100], 1.0),
        ([50, 50, 50, 50], 0.5),
        ([40, 20, 10, 30], 0.8),
    ],
)
def test_precision(val_list, expected_val):
    val = _precision(val_list)
    assert val == expected_val


@pytest.mark.parametrize(
    "val_list,expected_val",
    [
        ([0, 0, 100, 100], 0.0),
        ([100, 0, 0, 100], 2 / 3),
        ([50, 50, 50, 50], 0.5),
        ([40, 20, 10, 30], 2 / 3),
    ],
)
def test_f1(val_list, expected_val):
    val = _f1(val_list)
    assert val == expected_val


def test_find_confusion_matrix_per_threshold_errors(
    dummy_binary_pipeline,
    dummy_multiclass_pipeline,
):
    X = pd.DataFrame()
    y = pd.Series()

    with pytest.raises(
        ValueError,
        match="Expected a fitted binary classification pipeline",
    ):
        find_confusion_matrix_per_thresholds(dummy_binary_pipeline, X, y)

    with pytest.raises(
        ValueError,
        match="Expected a fitted binary classification pipeline",
    ):
        find_confusion_matrix_per_thresholds(dummy_multiclass_pipeline, X, y)

    dummy_multiclass_pipeline._is_fitted = True
    with pytest.raises(
        ValueError,
        match="Expected a fitted binary classification pipeline",
    ):
        find_confusion_matrix_per_thresholds(dummy_multiclass_pipeline, X, y)


@patch("evalml.pipelines.BinaryClassificationPipeline.fit")
@patch("evalml.pipelines.BinaryClassificationPipeline.predict_proba")
@patch(
    "evalml.model_understanding.decision_boundary._find_confusion_matrix_objective_threshold",
)
@patch("evalml.model_understanding.decision_boundary._find_data_between_ranges")
def test_find_confusion_matrix_per_threshold_args_pass_through(
    mock_ranges,
    mock_threshold,
    mock_pred_proba,
    mock_fit,
    dummy_binary_pipeline,
):
    n_bins = 100
    X = pd.DataFrame()
    y = pd.Series([0] * 500 + [1] * 500)
    dummy_binary_pipeline._is_fitted = True

    # set return predicted proba
    preds = [0.1] * 250 + [0.8] * 500 + [0.6] * 250
    pred_proba = pd.DataFrame({0: [1 - v for v in preds], 1: preds})
    mock_pred_proba.return_value = pred_proba

    # set the output for the thresholding private method
    obj_dict = {
        "accuracy": [{"objective score": 0.5, "threshold value": 0.5}, "some function"],
        "balanced_accuracy": [
            {"objective score": 0.5, "threshold value": 0.25},
            "some function",
        ],
    }
    conf_matrix = np.array([[0, 100, 280, 0] for i in range(n_bins)])
    mock_threshold.return_value = (conf_matrix, obj_dict)

    # set the output for data between ranges
    range_result = [[range(5)] for i in range(n_bins)]
    mock_ranges.return_value = range_result

    # calculate the expected output results
    bins = [i / n_bins for i in range(n_bins + 1)]
    expected_pos_skew, pos_range = np.histogram(pred_proba.iloc[:, -1][500:], bins=bins)
    expected_neg_skew, _ = np.histogram(pred_proba.iloc[:, -1][:500], bins=bins)
    expected_result_df = pd.DataFrame(
        {
            "true_pos_count": expected_pos_skew,
            "true_neg_count": expected_neg_skew,
            "true_positives": conf_matrix[:, 0].tolist(),
            "true_negatives": conf_matrix[:, 1].tolist(),
            "false_positives": conf_matrix[:, 2].tolist(),
            "false_negatives": conf_matrix[:, 3].tolist(),
            "data_in_bins": range_result,
        },
        index=pos_range[1:],
    )
    final_obj_dict = {
        "accuracy": {"objective score": 0.5, "threshold value": 0.5},
        "balanced_accuracy": {"objective score": 0.5, "threshold value": 0.25},
    }

    returned_result = find_confusion_matrix_per_thresholds(
        dummy_binary_pipeline,
        X,
        y,
        n_bins,
    )
    call_args = mock_threshold.call_args
    assert all(call_args[0][0] == expected_pos_skew)
    assert all(call_args[0][1] == expected_neg_skew)
    assert all(call_args[0][2] == pos_range)

    assert isinstance(returned_result, tuple)
    pd.testing.assert_frame_equal(returned_result[0], expected_result_df)
    assert returned_result[1] == final_obj_dict


@patch("evalml.pipelines.BinaryClassificationPipeline.fit")
@patch("evalml.pipelines.BinaryClassificationPipeline.predict_proba")
@pytest.mark.parametrize("n_bins", [100, 10, None])
def test_find_confusion_matrix_per_threshold_n_bins(
    mock_pred_proba,
    mock_fit,
    n_bins,
    dummy_binary_pipeline,
):
    X = pd.DataFrame()
    y = pd.Series([0] * 1200 + [1] * 800)
    dummy_binary_pipeline._is_fitted = True
    top_k = 5

    # set return predicted proba
    preds = [0.1] * 400 + [0.8] * 400 + [0.6] * 400 + [0.4] * 400 + [0.5] * 400
    pred_proba = pd.DataFrame({0: [1 - v for v in preds], 1: preds})
    mock_pred_proba.return_value = pred_proba

    # calculate the expected output results
    returned_result = find_confusion_matrix_per_thresholds(
        dummy_binary_pipeline,
        X,
        y,
        n_bins,
        top_k=top_k,
    )
    assert isinstance(returned_result, tuple)
    if n_bins is not None:
        assert len(returned_result[0]) == n_bins
    assert returned_result[0].columns.tolist() == [
        "true_pos_count",
        "true_neg_count",
        "true_positives",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "data_in_bins",
    ]
    assert sum(returned_result[0]["true_pos_count"]) == 800
    assert sum(returned_result[0]["true_neg_count"]) == 1200
    assert all([len(v) <= top_k for v in returned_result[0]["data_in_bins"]])
    assert isinstance(returned_result[1], dict)
    assert set(returned_result[1].keys()) == {
        "accuracy",
        "balanced_accuracy",
        "precision",
        "f1",
    }


@patch("evalml.pipelines.BinaryClassificationPipeline.fit")
@patch("evalml.pipelines.BinaryClassificationPipeline.predict_proba")
@pytest.mark.parametrize("top_k", [-1, 4])
@pytest.mark.parametrize("n_bins", [100, None])
def test_find_confusion_matrix_per_threshold_k_(
    mock_pred_proba,
    mock_fit,
    n_bins,
    top_k,
    dummy_binary_pipeline,
):
    X = pd.DataFrame()
    y = pd.Series([0] * 1200 + [1] * 800)
    dummy_binary_pipeline._is_fitted = True

    # set return predicted proba
    preds = [0.1] * 400 + [0.8] * 400 + [0.6] * 400 + [0.4] * 400 + [0.5] * 400
    pred_proba = pd.DataFrame({0: [1 - v for v in preds], 1: preds})
    mock_pred_proba.return_value = pred_proba

    # calculate the expected output results
    returned_result = find_confusion_matrix_per_thresholds(
        dummy_binary_pipeline,
        X,
        y,
        n_bins=n_bins,
        top_k=top_k,
    )
    assert isinstance(returned_result, tuple)
    if n_bins is not None:
        assert len(returned_result[0]) == n_bins
    n_bins = len(returned_result[0])
    if top_k == -1:
        assert sum([len(v) for v in returned_result[0]["data_in_bins"]]) == 2000
    else:
        assert (
            sum([len(v) for v in returned_result[0]["data_in_bins"]]) <= top_k * n_bins
        )


@pytest.mark.parametrize(
    "ranges",
    [[i / 10 for i in range(11)], [i / 50 for i in range(51)]],
)
@pytest.mark.parametrize("top_k", [-1, 5])
def test_find_data_between_ranges(top_k, ranges):
    data = pd.Series([(i % 100) / 100 for i in range(10000)])
    res = _find_data_between_ranges(data, ranges, top_k)
    lens = 10000 / (len(ranges) - 1) if top_k == -1 else top_k
    assert all([len(v) == lens for v in res])
    total_len = sum([len(v) for v in res])
    # check that the values are all unique here
    res = np.ravel(res)
    assert len(set(res)) == total_len


@pytest.mark.parametrize(
    "pos_skew",
    [
        [0, 0, 2, 3, 5, 10, 20, 20, 20, 20],
        [2, 1, 5, 15, 17, 20, 20, 20, 0, 0],
        [0, 0, 5, 5, 15, 15, 40, 20, 0, 0],
        [20, 20, 0, 5, 10, 5, 0, 0, 20, 20],
    ],
)
@pytest.mark.parametrize(
    "neg_skew",
    [
        [20, 30, 15, 15, 10, 5, 3, 2, 0, 0],
        [0, 0, 15, 15, 10, 5, 30, 20, 5, 0],
        [0, 0, 0, 15, 10, 25, 20, 10, 10, 10],
    ],
)
def test_find_confusion_matrix_objective_threshold(pos_skew, neg_skew):
    # test a variety of bin skews
    ranges = [i / 10 for i in range(11)]
    conf_mat_list, obj_dict = _find_confusion_matrix_objective_threshold(
        pos_skew,
        neg_skew,
        ranges,
    )
    total_pos, total_neg = 100, 100
    pos, neg = 0, 0
    objective_dict = {
        "accuracy": [{"objective score": 0, "threshold value": 0}, _accuracy],
        "balanced_accuracy": [
            {"objective score": 0, "threshold value": 0},
            _balanced_accuracy,
        ],
        "precision": [{"objective score": 0, "threshold value": 0}, _precision],
        "f1": [{"objective score": 0, "threshold value": 0}, _f1],
    }
    expected_conf_mat = []
    for i, range_val in enumerate(ranges[1:]):
        pos += pos_skew[i]
        neg += neg_skew[i]
        tp = total_pos - pos
        fp = total_neg - neg
        cm = [tp, neg, fp, pos]
        assert sum(cm) == 200
        expected_conf_mat.append(cm)

        for k, v in objective_dict.items():
            obj_val = v[1](cm)
            if obj_val > v[0]["objective score"]:
                v[0]["objective score"] = obj_val
                v[0]["threshold value"] = range_val

    assert conf_mat_list == expected_conf_mat
    assert obj_dict == objective_dict


@pytest.mark.parametrize("top_k", [3, -1])
def test_find_confusion_matrix_per_threshold(
    top_k,
    logistic_regression_binary_pipeline,
    X_y_binary,
):
    X, y = X_y_binary
    logistic_regression_binary_pipeline.fit(X, y)
    res_df, obj_dict = find_confusion_matrix_per_thresholds(
        logistic_regression_binary_pipeline,
        X,
        y,
        n_bins=10,
        top_k=top_k,
    )
    assert len(res_df) == 10
    if top_k == 3:
        assert sum([len(s) for s in res_df["data_in_bins"]]) <= 30
    else:
        assert sum([len(s) for s in res_df["data_in_bins"]]) == len(y)
    # assert all([sum(v) == 100 for v in res_df[["true_positives", "true_negatives", "false_postives", "false_negatives"]])
    assert len(obj_dict) == 4


def test_find_confusion_matrix_encode(logistic_regression_binary_pipeline, X_y_binary):
    bcp = logistic_regression_binary_pipeline
    bcp_new = logistic_regression_binary_pipeline.new(parameters={})
    X, y = X_y_binary
    y_new = pd.Series(["Value_1" if s == 1 else "Value_0" for s in y])
    bcp.fit(X, y)
    bcp_new.fit(X, y_new)
    res_df, obj_dict = find_confusion_matrix_per_thresholds(
        logistic_regression_binary_pipeline,
        X,
        y,
    )
    res_df_new, obj_dict_new = find_confusion_matrix_per_thresholds(bcp_new, X, y_new)
    pd.testing.assert_frame_equal(res_df, res_df_new)
    assert obj_dict == obj_dict_new


def test_find_confusion_matrix_values():
    pos_skew = [0, 5, 5, 15, 25]
    neg_skew = [25, 15, 5, 5, 0]
    ranges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    expected_cm = [
        [50, 25, 25, 0],
        [45, 40, 10, 5],
        [40, 45, 5, 10],
        [25, 50, 0, 25],
        [0, 50, 0, 50],
    ]
    expected_objective_dic = {
        "accuracy": [{"objective score": 0.85, "threshold value": 0.4}, _accuracy],
        "balanced_accuracy": [
            {"objective score": 17 / 20, "threshold value": 0.4},
            _balanced_accuracy,
        ],
        "precision": [{"objective score": 1.0, "threshold value": 0.8}, _precision],
        "f1": [{"objective score": 6 / 7, "threshold value": 0.4}, _f1],
    }
    cm, ob_dic = _find_confusion_matrix_objective_threshold(pos_skew, neg_skew, ranges)
    assert cm == expected_cm
    for k, v in ob_dic.items():
        np.testing.assert_allclose(
            list(v[0].values()),
            list(expected_objective_dic[k][0].values()),
        )
        assert v[1] == expected_objective_dic[k][1]


def test_find_confusion_matrix_json(logistic_regression_binary_pipeline, X_y_binary):
    X, y = X_y_binary
    logistic_regression_binary_pipeline.fit(X, y)
    res_df, obj_dict = find_confusion_matrix_per_thresholds(
        logistic_regression_binary_pipeline,
        X,
        y,
    )
    json_result = find_confusion_matrix_per_thresholds(
        logistic_regression_binary_pipeline,
        X,
        y,
        to_json=True,
    )

    result = json.loads(json_result)
    df = pd.DataFrame(result["results"], index=result["thresholds"])
    object_dict = result["objectives"]
    assert object_dict == obj_dict
    pd.testing.assert_frame_equal(res_df, df)


@pytest.mark.parametrize(
    "threshold,expected_len",
    [(0.5, 20), (None, 20), (0.6789012, 21)],
)
def test_find_confusion_matrix_pipeline_threshold(
    threshold,
    expected_len,
    logistic_regression_binary_pipeline,
    X_y_binary,
):
    X, y = X_y_binary
    logistic_regression_binary_pipeline.fit(X, y)
    logistic_regression_binary_pipeline.threshold = threshold
    res_df, _ = find_confusion_matrix_per_thresholds(
        logistic_regression_binary_pipeline,
        X,
        y,
        n_bins=20,
    )
    assert len(res_df) == expected_len
    if threshold is not None:
        assert threshold in res_df.index


def test_find_confusion_matrix_pipeline_threshold_tester(
    logistic_regression_binary_pipeline,
    X_y_binary,
):
    bcp = logistic_regression_binary_pipeline
    X, y = X_y_binary
    y = pd.Series(y)
    bcp.fit(X, y)
    preds = bcp.predict_proba(X).iloc[:, 1]
    bcp.optimize_threshold(X, y, preds, AccuracyBinary())
    first_accuracy = bcp.score(X, y, [AccuracyBinary()])["Accuracy Binary"]
    best_search = 0
    for i in range(20):
        bcp.threshold = i / 20
        search = bcp.score(X, y, [AccuracyBinary()])["Accuracy Binary"]
        if search > best_search:
            best_search = search
    assert first_accuracy >= best_search
