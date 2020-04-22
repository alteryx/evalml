import numpy as np
import pandas as pd
import pytest

from evalml.pipelines.plot_utils import (
    confusion_matrix,
    normalize_confusion_matrix,
    roc_curve
)


def test_roc_curve():
    y_true = np.array([1, 1, 0, 0])
    y_predict_proba = np.array([0.1, 0.4, 0.35, 0.8])
    fpr, tpr, thresholds = roc_curve(y_true, y_predict_proba)
    assert not np.isnan(fpr).any()
    assert not np.isnan(tpr).any()
    assert not np.isnan(thresholds).any()


def test_confusion_matrix():
    y_true = [2, 0, 2, 2, 0, 1]
    y_predicted = [0, 0, 2, 2, 0, 2]
    score = confusion_matrix(y_true, y_predicted)
    cm_expected = np.array([[2, 0, 0], [0, 0, 1], [1, 0, 2]])
    assert np.array_equal(cm_expected, score)


def test_normalize_confusion_matrix():
    conf_mat = np.array([[2, 3, 0], [0, 1, 1], [1, 0, 2]])
    conf_mat_normalized = normalize_confusion_matrix(conf_mat)
    assert all(conf_mat_normalized.sum(axis=1) == 1.0)

    conf_mat_normalized = normalize_confusion_matrix(conf_mat, 'pred')
    for col_sum in conf_mat_normalized.sum(axis=0):
        assert col_sum == 1.0 or col_sum == 0.0

    conf_mat_normalized = normalize_confusion_matrix(conf_mat, 'all')
    assert conf_mat_normalized.sum() == 1.0

    # testing with pd.DataFrames
    conf_mat_df = pd.DataFrame()
    conf_mat_df["col_1"] = [0, 1, 2]
    conf_mat_df["col_2"] = [0, 0, 3]
    conf_mat_df["col_3"] = [2, 0, 0]
    conf_mat_normalized = normalize_confusion_matrix(conf_mat_df)
    assert all(conf_mat_normalized.sum(axis=1) == 1.0)
    assert list(conf_mat_normalized.columns) == ['col_1', 'col_2', 'col_3']

    conf_mat_normalized = normalize_confusion_matrix(conf_mat_df, 'pred')
    for col_sum in conf_mat_normalized.sum(axis=0):
        assert col_sum == 1.0 or col_sum == 0.0

    conf_mat_normalized = normalize_confusion_matrix(conf_mat_df, 'all')
    assert conf_mat_normalized.sum().sum() == 1.0


def test_normalize_confusion_matrix_error():
    conf_mat = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    with pytest.raises(ValueError, match='Invalid value provided'):
        normalize_confusion_matrix(conf_mat, option='invalid option')
    with pytest.raises(ValueError, match='Invalid value provided'):
        normalize_confusion_matrix(conf_mat, option=None)

    with pytest.raises(ValueError, match="Sum of given axis is 0"):
        normalize_confusion_matrix(conf_mat, 'true')
    with pytest.raises(ValueError, match="Sum of given axis is 0"):
        normalize_confusion_matrix(conf_mat, 'pred')
    with pytest.raises(ValueError, match="Sum of given axis is 0"):
        normalize_confusion_matrix(conf_mat, 'all')
