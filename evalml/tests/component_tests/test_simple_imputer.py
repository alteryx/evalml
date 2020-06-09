import numpy as np
import pandas as pd
import pytest
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
    X_expected_arr = pd.DataFrame([[1, 0, 1, 1],
                                   [1, 2, 3, 2],
                                   [1, 2, 3, 0]])
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
    X_expected_arr = pd.DataFrame([[3, 0, 1, 3],
                                   ["a", 2, 3, 3],
                                   ["b", 2, 3, 0]])
    X_t = transformer.fit_transform(X)
    assert_frame_equal(X_expected_arr, X_t, check_dtype=False)

    X_t = transformer.transform(X)
    assert_frame_equal(X_expected_arr, X_t, check_dtype=False)


def test_most_frequent():
    X = pd.DataFrame([[np.nan, 0, 1, np.nan],
                      ["a", 2, np.nan, 3],
                      ["b", 2, 1, 0]])

    transformer = SimpleImputer(impute_strategy='most_frequent')
    X_expected_arr = pd.DataFrame([["a", 0, 1, 0],
                                   ["a", 2, 1, 3],
                                   ["b", 2, 1, 0]])
    X_t = transformer.fit_transform(X)
    assert_frame_equal(X_expected_arr, X_t, check_dtype=False)

    X_t = transformer.transform(X)
    assert_frame_equal(X_expected_arr, X_t, check_dtype=False)


def test_col_with_non_numeric():
    # test col with all strings
    X = pd.DataFrame([["a", 0, 1, np.nan],
                      ["b", 2, 3, 3],
                      ["a", 2, 3, 1],
                      [np.nan, 2, 3, 0]])

    transformer = SimpleImputer(impute_strategy='mean')
    with pytest.raises(ValueError, match="Cannot use mean strategy with non-numeric data"):
        transformer.fit_transform(X)
    with pytest.raises(ValueError, match="Cannot use mean strategy with non-numeric data"):
        transformer.fit(X)

    transformer = SimpleImputer(impute_strategy='median')
    with pytest.raises(ValueError, match="Cannot use median strategy with non-numeric data"):
        transformer.fit_transform(X)
    with pytest.raises(ValueError, match="Cannot use median strategy with non-numeric data"):
        transformer.fit(X)

    transformer = SimpleImputer(impute_strategy='most_frequent')
    X_expected_arr = pd.DataFrame([["a", 0, 1, 0],
                                   ["b", 2, 3, 3],
                                   ["a", 2, 3, 1],
                                   ["a", 2, 3, 0]])
    X_t = transformer.fit_transform(X)
    assert_frame_equal(X_expected_arr, X_t, check_dtype=False)

    X_t = transformer.transform(X)
    assert_frame_equal(X_expected_arr, X_t, check_dtype=False)

    transformer = SimpleImputer(impute_strategy='constant', fill_value=2)
    X_expected_arr = pd.DataFrame([["a", 0, 1, 2],
                                   ["b", 2, 3, 3],
                                   ["a", 2, 3, 1],
                                   [2, 2, 3, 0]])
    X_t = transformer.fit_transform(X)
    assert_frame_equal(X_expected_arr, X_t, check_dtype=False)

    X_t = transformer.transform(X)
    assert_frame_equal(X_expected_arr, X_t, check_dtype=False)


def test_clone():
    X = pd.DataFrame()
    X["col_1"] = [2, 0, 1, 0, 0]
    X["col_2"] = [3, 2, 5, 1, 3]
    X["col_3"] = [0, 0, 1, 3, 2]
    X["col_4"] = [2, 4, 1, 4, 0]

    imputer = SimpleImputer()

    imputer.fit(X)
    transformed_imputer = imputer.transform(X)
    assert isinstance(transformed_imputer, pd.DataFrame)

    # Test unlearned clone
    imputer_clone = imputer.clone(learned=False)
    assert imputer.random_state == imputer_clone.random_state
    with pytest.raises(ValueError):
        imputer_clone.transform(X)

    imputer_clone.fit(X)
    transformed_imputer_clone = imputer_clone.transform(X)
    np.testing.assert_almost_equal(transformed_imputer.values, transformed_imputer_clone.values)

    # Test learned clone
    imputer_clone = imputer.clone()
    transformed_imputer_clone = imputer_clone.transform(X)
    np.testing.assert_almost_equal(transformed_imputer.values, transformed_imputer_clone.values)
