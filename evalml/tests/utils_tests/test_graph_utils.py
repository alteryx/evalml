import warnings

import numpy as np
import pandas as pd
import pytest

from evalml.utils.graph_utils import (
    confusion_matrix,
    normalize_confusion_matrix
)


@pytest.mark.parametrize("data_type", ['np', 'pd'])
def test_confusion_matrix(data_type):
    y_true = [2, 0, 2, 2, 0, 1]
    y_predicted = [0, 0, 2, 2, 0, 2]
    if data_type == 'pd':
        y_true = pd.Series(y_true)
        y_predicted = pd.Series(y_predicted)
    conf_mat = confusion_matrix(y_true, y_predicted, normalize_method=None)
    conf_mat_expected = np.array([[2, 0, 0], [0, 0, 1], [1, 0, 2]])
    assert np.array_equal(conf_mat_expected, conf_mat)
    assert isinstance(conf_mat, pd.DataFrame)
    conf_mat = confusion_matrix(y_true, y_predicted, normalize_method='true')
    conf_mat_expected = np.array([[1, 0, 0], [0, 0, 1], [1 / 3.0, 0, 2 / 3.0]])
    assert np.array_equal(conf_mat_expected, conf_mat)
    assert isinstance(conf_mat, pd.DataFrame)
    conf_mat = confusion_matrix(y_true, y_predicted, normalize_method='pred')
    conf_mat_expected = np.array([[2 / 3.0, np.nan, 0], [0, np.nan, 1 / 3.0], [1 / 3.0, np.nan, 2 / 3.0]])
    assert np.allclose(conf_mat_expected, conf_mat, equal_nan=True)
    assert isinstance(conf_mat, pd.DataFrame)
    conf_mat = confusion_matrix(y_true, y_predicted, normalize_method='all')
    conf_mat_expected = np.array([[1 / 3.0, 0, 0], [0, 0, 1 / 6.0], [1 / 6.0, 0, 1 / 3.0]])
    assert np.array_equal(conf_mat_expected, conf_mat)
    assert isinstance(conf_mat, pd.DataFrame)
    with pytest.raises(ValueError, match='Invalid value provided'):
        conf_mat = confusion_matrix(y_true, y_predicted, normalize_method='Invalid Option')


@pytest.mark.parametrize("data_type", ['np', 'pd'])
def test_normalize_confusion_matrix(data_type):
    conf_mat = np.array([[2, 3, 0], [0, 1, 1], [1, 0, 2]])
    if data_type == 'pd':
        conf_mat = pd.DataFrame(conf_mat)
    conf_mat_normalized = normalize_confusion_matrix(conf_mat)
    assert all(conf_mat_normalized.sum(axis=1) == 1.0)
    assert isinstance(conf_mat_normalized, type(conf_mat))

    conf_mat_normalized = normalize_confusion_matrix(conf_mat, 'pred')
    for col_sum in conf_mat_normalized.sum(axis=0):
        assert col_sum == 1.0 or col_sum == 0.0

    conf_mat_normalized = normalize_confusion_matrix(conf_mat, 'all')
    assert conf_mat_normalized.sum().sum() == 1.0

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
    warnings.simplefilter('default', category=RuntimeWarning)

    with pytest.raises(ValueError, match='Invalid value provided'):
        normalize_confusion_matrix(conf_mat, normalize_method='invalid option')
    with pytest.raises(ValueError, match='Invalid value provided'):
        normalize_confusion_matrix(conf_mat, normalize_method=None)

    with pytest.raises(ValueError, match="Sum of given axis is 0"):
        normalize_confusion_matrix(conf_mat, 'true')
    with pytest.raises(ValueError, match="Sum of given axis is 0"):
        normalize_confusion_matrix(conf_mat, 'pred')
    with pytest.raises(ValueError, match="Sum of given axis is 0"):
        normalize_confusion_matrix(conf_mat, 'all')
