import warnings

import numpy as np
import pandas as pd
import pytest
from skopt.space import Real

from evalml.model_understanding.graphs import (
    confusion_matrix,
    normalize_confusion_matrix
)
from evalml.pipelines import BinaryClassificationPipeline


@pytest.fixture
def test_pipeline():
    class TestPipeline(BinaryClassificationPipeline):
        component_graph = ['Simple Imputer', 'One Hot Encoder', 'Standard Scaler', 'Logistic Regression Classifier']

        hyperparameters = {
            "penalty": ["l2"],
            "C": Real(.01, 10),
            "impute_strategy": ["mean", "median", "most_frequent"],
        }

        def __init__(self, parameters):
            super().__init__(parameters=parameters)

    return TestPipeline(parameters={"Logistic Regression Classifier": {"n_jobs": 1}})


@pytest.mark.parametrize("data_type", ['np', 'pd', 'ww'])
def test_confusion_matrix(data_type, make_data_type):
    y_true = np.array([2, 0, 2, 2, 0, 1, 1, 0, 2])
    y_predicted = np.array([0, 0, 2, 2, 0, 2, 1, 1, 1])
    y_true = make_data_type(data_type, y_true)
    y_predicted = make_data_type(data_type, y_predicted)

    conf_mat = confusion_matrix(y_true, y_predicted, normalize_method=None)
    conf_mat_expected = np.array([[2, 1, 0], [0, 1, 1], [1, 1, 2]])
    assert np.array_equal(conf_mat_expected, conf_mat.to_numpy())
    assert isinstance(conf_mat, pd.DataFrame)

    conf_mat = confusion_matrix(y_true, y_predicted, normalize_method='all')
    conf_mat_expected = conf_mat_expected / 9.0
    assert np.array_equal(conf_mat_expected, conf_mat.to_numpy())
    assert isinstance(conf_mat, pd.DataFrame)

    conf_mat = confusion_matrix(y_true, y_predicted, normalize_method='true')
    conf_mat_expected = np.array([[2 / 3.0, 1 / 3.0, 0], [0, 0.5, 0.5], [0.25, 0.25, 0.5]])
    assert np.array_equal(conf_mat_expected, conf_mat.to_numpy())
    assert isinstance(conf_mat, pd.DataFrame)

    conf_mat = confusion_matrix(y_true, y_predicted, normalize_method='pred')
    conf_mat_expected = np.array([[2 / 3.0, 1 / 3.0, 0], [0, 1 / 3.0, 1 / 3.0], [1 / 3.0, 1 / 3.0, 2 / 3.0]])
    assert np.allclose(conf_mat_expected, conf_mat.to_numpy(), equal_nan=True)
    assert isinstance(conf_mat, pd.DataFrame)

    with pytest.raises(ValueError, match='Invalid value provided'):
        conf_mat = confusion_matrix(y_true, y_predicted, normalize_method='Invalid Option')


@pytest.mark.parametrize("data_type", ['ww', 'np', 'pd'])
def test_normalize_confusion_matrix(data_type, make_data_type):
    conf_mat = np.array([[2, 3, 0], [0, 1, 1], [1, 0, 2]])
    conf_mat = make_data_type(data_type, conf_mat)

    conf_mat_normalized = normalize_confusion_matrix(conf_mat)
    assert all(conf_mat_normalized.sum(axis=1) == 1.0)
    assert isinstance(conf_mat_normalized, pd.DataFrame)

    conf_mat_normalized = normalize_confusion_matrix(conf_mat, 'pred')
    for col_sum in conf_mat_normalized.sum(axis=0):
        assert col_sum == 1.0 or col_sum == 0.0

    conf_mat_normalized = normalize_confusion_matrix(conf_mat, 'all')
    assert conf_mat_normalized.sum().sum() == 1.0

    # testing with named pd.DataFrames
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


@pytest.mark.parametrize("data_type", ['ww', 'np', 'pd'])
def test_normalize_confusion_matrix_error(data_type, make_data_type):
    conf_mat = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    conf_mat = make_data_type(data_type, conf_mat)

    warnings.simplefilter('default', category=RuntimeWarning)

    with pytest.raises(ValueError, match='Invalid value provided for "normalize_method": invalid option'):
        normalize_confusion_matrix(conf_mat, normalize_method='invalid option')
    with pytest.raises(ValueError, match='Invalid value provided'):
        normalize_confusion_matrix(conf_mat, normalize_method=None)

    with pytest.raises(ValueError, match="Sum of given axis is 0"):
        normalize_confusion_matrix(conf_mat, 'true')
    with pytest.raises(ValueError, match="Sum of given axis is 0"):
        normalize_confusion_matrix(conf_mat, 'pred')
    with pytest.raises(ValueError, match="Sum of given axis is 0"):
        normalize_confusion_matrix(conf_mat, 'all')


@pytest.mark.parametrize("data_type", ['ww', 'pd', 'np'])
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

    y_true = np.array(['blue', 'red', 'blue', 'red'])
    y_pred = np.array(['blue', 'red', 'red', 'red'])
    y_true = make_data_type(data_type, y_true)
    y_pred = make_data_type(data_type, y_pred)
    conf_mat = confusion_matrix(y_true=y_true, y_predicted=y_pred)
    labels = ['blue', 'red']
    assert np.array_equal(conf_mat.index, labels)
    assert np.array_equal(conf_mat.columns, labels)

    y_true = np.array(['blue', 'red', 'red', 'red', 'orange', 'orange'])
    y_pred = np.array(['red', 'blue', 'blue', 'red', 'orange', 'orange'])
    y_true = make_data_type(data_type, y_true)
    y_pred = make_data_type(data_type, y_pred)
    conf_mat = confusion_matrix(y_true=y_true, y_predicted=y_pred)
    labels = ['blue', 'orange', 'red']
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
