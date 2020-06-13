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

    transformer = SimpleImputer(impute_strategy='constant', fill_value=2)
    X_expected_arr = pd.DataFrame([["a", 0, 1, 2],
                                   ["b", 2, 3, 3],
                                   ["a", 2, 3, 1],
                                   [2, 2, 3, 0]])
    X_t = transformer.fit_transform(X)
    assert_frame_equal(X_expected_arr, X_t, check_dtype=False)


def test_fit_transform_drop_all_nan_columns():
    X = pd.DataFrame({"all_nan": [np.nan, np.nan, np.nan],
                      "some_nan": [np.nan, 1, 0],
                      "another_col": [0, 1, 2]})

    transformer = SimpleImputer(impute_strategy='most_frequent')
    X_expected_arr = pd.DataFrame({"some_nan": [0, 1, 0], "another_col": [0, 1, 2]})
    X_t = transformer.fit_transform(X)
    assert_frame_equal(X_expected_arr, X_t, check_dtype=False)
    assert_frame_equal(X, pd.DataFrame({"all_nan": [np.nan, np.nan, np.nan],
                                        "some_nan": [np.nan, 1, 0],
                                        "another_col": [0, 1, 2]}))


def test_transform_drop_all_nan_columns():
    X = pd.DataFrame({"all_nan": [np.nan, np.nan, np.nan],
                      "some_nan": [np.nan, 1, 0],
                      "another_col": [0, 1, 2]})
    transformer = SimpleImputer(impute_strategy='most_frequent')
    transformer.fit(X)
    X_expected_arr = pd.DataFrame({"some_nan": [0, 1, 0], "another_col": [0, 1, 2]})
    assert_frame_equal(X_expected_arr, transformer.transform(X), check_dtype=False)
    assert_frame_equal(X, pd.DataFrame({"all_nan": [np.nan, np.nan, np.nan],
                                        "some_nan": [np.nan, 1, 0],
                                        "another_col": [0, 1, 2]}))


def test_transform_drop_all_nan_columns_empty():
    X = pd.DataFrame([[np.nan, np.nan, np.nan]])
    transformer = SimpleImputer(impute_strategy='most_frequent')
    assert transformer.fit_transform(X).empty
    assert_frame_equal(X, pd.DataFrame([[np.nan, np.nan, np.nan]]))

    transformer = SimpleImputer(impute_strategy='most_frequent')
    transformer.fit(X)
    assert transformer.transform(X).empty
    assert_frame_equal(X, pd.DataFrame([[np.nan, np.nan, np.nan]]))


def test_transform_before_fit():
    with pytest.raises(RuntimeError, match="Must fit transformer before calling transform!"):
        SimpleImputer(impute_strategy='most_frequent').transform(pd.DataFrame())


def test_numpy_input():
    X = np.array([[np.nan, 0, 1, np.nan],
                  [np.nan, 2, 3, 2],
                  [np.nan, 2, 3, 0]])
    transformer = SimpleImputer(impute_strategy='mean')
    X_expected_arr = np.array([[0, 1, 1],
                               [2, 3, 2],
                               [2, 3, 0]])
    assert np.allclose(X_expected_arr, transformer.fit_transform(X))
    np.testing.assert_almost_equal(X, np.array([[np.nan, 0, 1, np.nan],
                                                [np.nan, 2, 3, 2],
                                                [np.nan, 2, 3, 0]]))
