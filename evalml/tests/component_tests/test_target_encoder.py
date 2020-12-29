from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pytest import importorskip

from evalml.exceptions import ComponentNotYetFittedError
from evalml.pipelines import RegressionPipeline
from evalml.pipelines.components import TargetEncoder
from evalml.preprocessing import split_data

importorskip('category_encoders', reason='Skipping test because category_encoders not installed')


def test_init():
    parameters = {"cols": None,
                  "smoothing": 1.0,
                  "handle_unknown": "value",
                  "handle_missing": "value"}
    encoder = TargetEncoder()
    assert encoder.parameters == parameters


def test_parameters():
    encoder = TargetEncoder(cols=['a'])
    expected_parameters = {"cols": ['a'],
                           "smoothing": 1.0,
                           "handle_unknown": "value",
                           "handle_missing": "value"}
    assert encoder.parameters == expected_parameters


def test_categories():
    encoder = TargetEncoder()
    with pytest.raises(AttributeError, match="'TargetEncoder' object has no attribute"):
        encoder.categories


def test_invalid_inputs():
    with pytest.raises(ValueError, match="Invalid input 'test' for handle_unknown"):
        TargetEncoder(handle_unknown='test')
    with pytest.raises(ValueError, match="Invalid input 'test2' for handle_missing"):
        TargetEncoder(handle_missing='test2')
    with pytest.raises(ValueError, match="Smoothing value needs to be strictly larger than 0"):
        TargetEncoder(smoothing=0)


def test_null_values_in_dataframe():
    X = pd.DataFrame({'col_1': ["a", "b", "c", "d", np.nan],
                      'col_2': ["a", "b", "a", "c", "b"],
                      'col_3': ["a", "a", "a", "a", "a"]})
    y = pd.Series([0, 1, 1, 1, 0])
    encoder = TargetEncoder(handle_missing='value')
    encoder.fit(X, y)
    X_t = encoder.transform(X)
    X_expected = pd.DataFrame({'col_1': [0.6, 0.6, 0.6, 0.6, 0.6],
                               'col_2': [0.526894, 0.526894, 0.526894, 0.6, 0.526894],
                               'col_3': [0.6, 0.6, 0.6, 0.6, 0.6, ]})

    pd.testing.assert_frame_equal(X_t, X_expected)

    encoder = TargetEncoder(handle_missing='return_nan')
    encoder.fit(X, y)
    X_t = encoder.transform(X)
    X_expected = pd.DataFrame({'col_1': [0.6, 0.6, 0.6, 0.6, np.nan],
                               'col_2': [0.526894, 0.526894, 0.526894, 0.6, 0.526894],
                               'col_3': [0.6, 0.6, 0.6, 0.6, 0.6, ]})
    pd.testing.assert_frame_equal(X_t, X_expected)

    encoder = TargetEncoder(handle_missing='error')
    with pytest.raises(ValueError, match='Columns to be encoded can not contain null'):
        encoder.fit(X, y)


def test_cols():
    X = pd.DataFrame({'col_1': [1, 2, 1, 1, 2],
                      'col_2': ['2', '1', '1', '1', '1'],
                      'col_3': ["a", "a", "a", "a", "a"]})
    y = pd.Series([0, 1, 1, 1, 0])
    encoder = TargetEncoder(cols=[])
    encoder.fit(X, y)
    X_t = encoder.transform(X)
    pd.testing.assert_frame_equal(X, X_t)

    encoder = TargetEncoder(cols=['col_2'])
    encoder.fit(X, y)
    X_t = encoder.transform(X)
    X_expected = pd.DataFrame({'col_1': [1, 2, 1, 1, 2],
                               'col_2': [0.60000, 0.742886, 0.742886, 0.742886, 0.742886],
                               'col_3': ["a", "a", "a", "a", "a"]})
    pd.testing.assert_frame_equal(X_t, X_expected, check_less_precise=True)

    encoder = TargetEncoder(cols=['col_2', 'col_3'])
    encoder.fit(X, y)
    X_t = encoder.transform(X)
    encoder2 = TargetEncoder()
    encoder2.fit(X, y)
    X_t2 = encoder2.transform(X)
    pd.testing.assert_frame_equal(X_t, X_t2)


def test_transform():
    X = pd.DataFrame({'col_1': [1, 2, 1, 1, 2],
                      'col_2': ["r", "t", "s", "t", "t"],
                      'col_3': ["a", "a", "a", "b", "a"]})
    y = pd.Series([0, 1, 1, 1, 0])
    encoder = TargetEncoder()
    encoder.fit(X, y)
    X_t = encoder.transform(X)
    X_expected = pd.DataFrame({'col_1': [1, 2, 1, 1, 2],
                               'col_2': [0.6, 0.65872, 0.6, 0.65872, 0.65872],
                               'col_3': [0.504743, 0.504743, 0.504743, 0.6, 0.504743]})
    pd.testing.assert_frame_equal(X_t, X_expected)


def test_smoothing():
    # larger smoothing values should bring the values closer to the global mean
    X = pd.DataFrame({'col_1': [1, 2, 1, 1, 2],
                      'col_2': [2, 1, 1, 1, 1],
                      'col_3': ["a", "a", "a", "a", "b"]})
    y = pd.Series([0, 1, 1, 1, 0])
    encoder = TargetEncoder(smoothing=1)
    encoder.fit(X, y)
    X_t = encoder.transform(X)
    X_expected = pd.DataFrame({'col_1': [1, 2, 1, 1, 2],
                               'col_2': [2, 1, 1, 1, 1],
                               'col_3': [0.742886, 0.742886, 0.742886, 0.742886, 0.6]})
    pd.testing.assert_frame_equal(X_t, X_expected)

    encoder = TargetEncoder(smoothing=10)
    encoder.fit(X, y)
    X_t = encoder.transform(X)
    X_expected = pd.DataFrame({'col_1': [1, 2, 1, 1, 2],
                               'col_2': [2, 1, 1, 1, 1],
                               'col_3': [0.686166, 0.686166, 0.686166, 0.686166, 0.6]})
    pd.testing.assert_frame_equal(X_t, X_expected)

    encoder = TargetEncoder(smoothing=100)
    encoder.fit(X, y)
    X_t = encoder.transform(X)
    X_expected = pd.DataFrame({'col_1': [1, 2, 1, 1, 2],
                               'col_2': [2, 1, 1, 1, 1],
                               'col_3': [0.676125, 0.676125, 0.676125, 0.676125, 0.6]})
    pd.testing.assert_frame_equal(X_t, X_expected)


def test_get_feature_names():
    X = pd.DataFrame({'col_1': [1, 2, 1, 1, 2],
                      'col_2': ["r", "t", "s", "t", "t"],
                      'col_3': ["a", "a", "a", "b", "a"]})
    y = pd.Series([0, 1, 1, 1, 0])
    encoder = TargetEncoder()
    with pytest.raises(ComponentNotYetFittedError, match='This TargetEncoder is not fitted yet. You must fit'):
        encoder.get_feature_names()
    encoder.fit(X, y)
    np.testing.assert_array_equal(encoder.get_feature_names(), np.array(['col_1', 'col_2', 'col_3']))


def test_custom_indices():
    # custom regression pipeline
    class MyTargetPipeline(RegressionPipeline):
        component_graph = ['Imputer', 'Target Encoder', 'Linear Regressor']
        custom_name = "Target Pipeline"

    X = pd.DataFrame({"a": ["a", "b", "a", "a", "a", "c", "c", "c"], "b": [0, 1, 1, 1, 1, 1, 0, 1]})
    y = pd.Series([0, 0, 0, 1, 0, 1, 0, 0], index=[7, 2, 1, 4, 5, 3, 6, 8])

    x1, x2, y1, y2 = split_data(X, y, problem_type='binary')
    tp = MyTargetPipeline({})
    tp.fit(x2, y2)


@patch('evalml.pipelines.components.transformers.transformer.Transformer.fit')
def test_pandas_numpy(mock_fit, X_y_binary):
    X, y = X_y_binary
    X = pd.DataFrame(X).sample(frac=1)

    encoder = TargetEncoder()
    X_t = pd.DataFrame(X).reset_index(drop=True, inplace=False)

    encoder.fit(X, y)
    pd.testing.assert_frame_equal(mock_fit.call_args[0][0], X_t)

    X_numpy = X.to_numpy()
    encoder.fit(X_numpy, y)
