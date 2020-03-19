import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from evalml.pipelines.components import SimpleImputer


def test_median():
    X = pd.DataFrame([[np.nan, 0, 1, np.nan],
                      [1, 2, 3, 2],
                      [10, 2, np.nan, 2],
                      [10, 2, 5, np.nan],
                      [6, 2, 7, 0]])
    transformer = SimpleImputer(impute_strategy='median')
    X_expected_arr = pd.DataFrame([[8, 0, 1, 2],
                                   [1, 2, 3, 2],
                                   [10, 2, 4, 2],
                                   [10, 2, 5, 2],
                                   [6, 2, 7, 0]])
    X_t = transformer.fit_transform(X)
    assert_frame_equal(X_expected_arr, X_t, check_dtype=False)

    X_t = transformer.transform(X)
    assert_frame_equal(X_expected_arr, X_t, check_dtype=False)


def test_mean():
    X = pd.DataFrame([[np.nan, 0, 1, np.nan],
                      [1, 2, 3, 2],
                      [1, 2, 3, 0]])
    # test impute_strategy
    transformer = SimpleImputer(impute_strategy='mean')
    X_expected_arr = pd.DataFrame([[1.0, 0, 1, 1.0],
                                   [1.0, 2, 3, 2.0],
                                   [1.0, 2, 3, 0.0]])
    X_t = transformer.fit_transform(X)
    assert_frame_equal(X_expected_arr, X_t, check_dtype=False)

    X_t = transformer.transform(X)
    assert_frame_equal(X_expected_arr, X_t, check_dtype=False)


def test_constant():
    # test impute strategy is constant and fill value is not specified
    X = pd.DataFrame([[np.nan, 0, 1, np.nan],
                      ["a", 2, np.nan, 3],
                      ["b", 2, 3, 0]])

    transformer = SimpleImputer(impute_strategy='constant', fill_value=3)
    X_expected_arr = pd.DataFrame([[3, 0, 1.0, 3.0],
                                   ["a", 2, 3.0, 3.0],
                                   ["b", 2, 3.0, 0.0]])
    X_t = transformer.fit_transform(X)
    assert_frame_equal(X_expected_arr, X_t, check_dtype=False)

    X_t = transformer.transform(X)
    assert_frame_equal(X_expected_arr, X_t, check_dtype=False)


def test_most_frequent():
    X = pd.DataFrame([[np.nan, 0, 1, np.nan],
                      ["a", 2, np.nan, 3],
                      ["b", 2, 3, 0]])

    transformer = SimpleImputer(impute_strategy='most_frequent')
    X_expected_arr = pd.DataFrame([["a", 0, 1.0, 0.0],
                                   ["a", 2, 3.0, 3.0],
                                   ["b", 2, 3.0, 0.0]])
    X_t = transformer.fit_transform(X)
    assert_frame_equal(X_expected_arr, X_t, check_dtype=False)

    X_t = transformer.transform(X)
    assert_frame_equal(X_expected_arr, X_t, check_dtype=False)


def test_col_with_all_nan():
    # test that col with all NaN is removed
    X = pd.DataFrame([[np.nan, 0, 1, np.nan],
                      [np.nan, 2, 3, 3],
                      [np.nan, 2, 3, 0],
                      [np.nan, 2, 3, 0]])
    # test impute_strategy
    transformer = SimpleImputer(impute_strategy='mean')
    X_expected_arr = pd.DataFrame([[0, 1, 1],
                                   [2, 3, 3],
                                   [2, 3, 0],
                                   [2, 3, 0]])
    X_t = transformer.fit_transform(X)
    assert_frame_equal(X_expected_arr, X_t, check_dtype=False)

    X_t = transformer.transform(X)
    assert_frame_equal(X_expected_arr, X_t, check_dtype=False)
