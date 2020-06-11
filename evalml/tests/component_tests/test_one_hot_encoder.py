import numpy as np
import pandas as pd
import pytest

from evalml.pipelines.components import OneHotEncoder
from evalml.utils import get_random_state


def test_init():
    parameters = {"top_n": 10,
                  "categories": None,
                  "drop": None,
                  "handle_unknown": "ignore",
                  "handle_missing": "error"}
    encoder = OneHotEncoder()
    assert encoder.parameters == parameters


def test_fit_first():
    encoder = OneHotEncoder()
    with pytest.raises(RuntimeError, match="You must fit one hot encoder before calling transform!"):
        encoder.transform(pd.DataFrame())


def test_null_values_in_dataframe():
    error_msg = "Invalid input {} for handle_missing".format("peanut butter")
    with pytest.raises(ValueError, match=error_msg):
        encoder = OneHotEncoder(handle_missing="peanut butter")

    X = pd.DataFrame({'col_1': ["a", "b", "c", "d", np.nan],
                      'col_2': ["a", "b", "a", "c", "b"],
                      'col_3': ["a", "a", "a", "a", "a"]})

    # Test NaN will be counted as a category if within the top_n
    encoder = OneHotEncoder(handle_missing='as_category')
    encoder.fit(X)
    X_t = encoder.transform(X)

    expected_col_names = set(["col_1_a", "col_1_b", "col_1_c", "col_1_d", "col_1_nan",
                              "col_2_a", "col_2_b", "col_2_c", "col_3_a"])
    col_names = set(X_t.columns)
    assert (col_names == expected_col_names)
    assert X_t.shape == (5, 9)

    # Test NaN will not be counted as a category if not in the top_n
    X = pd.DataFrame({'col_1': ["a", "a", "c", "c", np.nan],
                      'col_2': ["a", "b", "a", "c", "b"],
                      'col_3': ["a", "a", "a", "a", "a"],
                      'col_4': [2, 0, 1, np.nan, 0]})

    encoder = OneHotEncoder(top_n=2, handle_missing='as_category')
    encoder.fit(X)
    X_t = encoder.transform(X)

    expected_col_names = set(["col_1_a", "col_1_c", "col_2_a", "col_2_b", "col_3_a", "col_4"])
    col_names = set(X_t.columns)
    assert (col_names == expected_col_names)
    assert X_t.shape == (5, 6)

    # Test handle_missing='error' throws an error
    encoder = OneHotEncoder(handle_missing='error')

    X = pd.DataFrame({"col_1": [np.nan, "b", "c", "d", "e", "f", "g"],
                      "col_2": ["a", "c", "d", "b", "e", "e", "f"],
                      "col_3": ["a", "a", "a", "a", "a", "a", "b"],
                      "col_4": [2, 0, 1, 3, 0, 1, 2]})

    with pytest.raises(ValueError, match="Input contains NaN"):
        encoder.fit(X)

    # Test NaN values in transformed data
    X = pd.DataFrame({'col_1': ["a", "b", "c", "d", "d"],
                      'col_2': ["a", "b", "a", "c", "b"],
                      'col_3': ["a", "a", "a", "a", "a"]})
    encoder = OneHotEncoder(handle_missing='error')
    encoder.fit(X)
    X_missing = pd.DataFrame({'col_1': ["a", "b", "c", "d", "d"],
                              'col_2': ["a", "b", np.nan, "c", "b"],
                              'col_3': ["a", "a", "a", "a", "a"]})
    with pytest.raises(ValueError, match="Input contains NaN"):
        encoder.transform(X_missing)


def test_drop():
    X = pd.DataFrame({'col_1': ["a", "b", "c", "d", "d"],
                      'col_2': ["a", "b", "a", "c", "b"],
                      'col_3': ["a", "a", "a", "a", "a"]})
    encoder = OneHotEncoder(top_n=None, drop='first', handle_unknown='error')
    encoder.fit(X)
    X_t = encoder.transform(X)
    col_names = set(X_t.columns)
    expected_col_names = set(["col_1_b", "col_1_c", "col_1_d",
                              "col_2_b", "col_2_c"])
    assert col_names == expected_col_names


def test_handle_unknown():
    error_msg = "Invalid input {} for handle_unknown".format("bananas")
    with pytest.raises(ValueError, match=error_msg):
        encoder = OneHotEncoder(handle_unknown="bananas")

    X = pd.DataFrame({"col_1": ["a", "b", "c", "d", "e", "f", "g"],
                      "col_2": ["a", "c", "d", "b", "e", "e", "f"],
                      "col_3": ["a", "a", "a", "a", "a", "a", "b"],
                      "col_4": [2, 0, 1, 3, 0, 1, 2]})

    encoder = OneHotEncoder(handle_unknown='error')
    encoder.fit(X)
    assert isinstance(encoder.transform(X), pd.DataFrame)

    X = pd.DataFrame({"col_1": ["x", "b", "c", "d", "e", "f", "g"],
                      "col_2": ["a", "c", "d", "b", "e", "e", "f"],
                      "col_3": ["a", "a", "a", "a", "a", "a", "b"],
                      "col_4": [2, 0, 1, 3, 0, 1, 2]})
    with pytest.raises(ValueError):
        encoder.transform(X)


def test_no_top_n():
    # test all categories in all columns are encoded when top_n is None
    X = pd.DataFrame({"col_1": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"],
                      "col_2": ["a", "c", "d", "b", "e", "e", "f", "a", "b", "c", "d"],
                      "col_3": ["a", "a", "a", "a", "a", "a", "b", "a", "a", "b", "b"],
                      "col_4": [2, 0, 1, 3, 0, 1, 2, 0, 2, 1, 2]})

    encoder = OneHotEncoder(top_n=None, handle_unknown="error", random_state=2)
    encoder.fit(X)
    X_t = encoder.transform(X)

    expected_col_names = set(["col_3_a", "col_3_b", "col_4"])
    for val in X["col_1"]:
        expected_col_names.add("col_1_" + val)
    for val in X["col_2"]:
        expected_col_names.add("col_2_" + val)
    col_names = set(X_t.columns)

    assert (X_t.shape == (11, 20))
    assert (col_names == expected_col_names)

    # Make sure unknown values cause an error
    X_new = pd.DataFrame({"col_1": ["a", "b", "c", "x"],
                          "col_2": ["a", "c", "d", "b"],
                          "col_3": ["a", "a", "a", "a"],
                          "col_4": [2, 0, 1, 3]})

    with pytest.raises(ValueError):
        encoder.transform(X_new)


def test_categories():
    X = pd.DataFrame({"col_1": ["a", "b", "c", "d", "e", "f", "g"],
                      "col_2": ["a", "c", "d", "b", "e", "e", "f"],
                      "col_3": ["a", "a", "a", "a", "a", "a", "b"],
                      "col_4": [2, 0, 1, 3, 0, 1, 2]})

    categories = [["a", "b", "c", "d"],
                  ["a", "b", "c"],
                  ["a", "b"]]

    # test categories value works
    encoder = OneHotEncoder(top_n=None, categories=categories, random_state=2)
    encoder.fit(X)
    X_t = encoder.transform(X)

    col_names = set(X_t.columns)
    expected_col_names = set(["col_1_a", "col_1_b", "col_1_c", "col_1_d",
                              "col_2_a", "col_2_b", "col_2_c", "col_3_a",
                              "col_3_b", "col_4"])
    assert (X_t.shape == (7, 10))
    assert (col_names == expected_col_names)

    # test categories with top_n errors
    with pytest.raises(ValueError, match="Cannot use categories and top_n arguments simultaneously"):
        encoder = OneHotEncoder(top_n=10, categories=categories, random_state=2)


def test_less_than_top_n_unique_values():
    # test that columns with less than n unique values encodes properly
    X = pd.DataFrame({"col_1": ["a", "b", "c", "d", "a"],
                      "col_2": ["a", "b", "a", "c", "b"],
                      "col_3": ["a", "a", "a", "a", "a"],
                      "col_4": [2, 0, 1, 0, 0]})

    encoder = OneHotEncoder()
    encoder.parameters['top_n'] = 5
    encoder.fit(X)
    X_t = encoder.transform(X)
    expected_col_names = set(["col_1_a", "col_1_b", "col_1_c", "col_1_d",
                              "col_2_a", "col_2_b", "col_2_c", "col_3_a", "col_4"])
    col_names = set(X_t.columns)
    assert (col_names == expected_col_names)


def test_more_top_n_unique_values():
    # test that columns with >= n unique values encodes properly
    X = pd.DataFrame({"col_1": ["a", "b", "c", "d", "e", "f", "g"],
                      "col_2": ["a", "c", "d", "b", "e", "e", "f"],
                      "col_3": ["a", "a", "a", "a", "a", "a", "b"],
                      "col_4": [2, 0, 1, 3, 0, 1, 2]})

    random_seed = 2
    encoder = OneHotEncoder(random_state=random_seed)
    test_random_state = get_random_state(random_seed)
    encoder.parameters['top_n'] = 5
    encoder.fit(X)
    X_t = encoder.transform(X)
    col_1_counts = X["col_1"].value_counts(dropna=False).to_frame()
    col_1_counts = col_1_counts.sample(frac=1, random_state=test_random_state)
    col_1_counts = col_1_counts.sort_values(["col_1"], ascending=False, kind='mergesort')
    col_1_samples = col_1_counts.head(encoder.parameters['top_n']).index.tolist()

    col_2_counts = X["col_2"].value_counts(dropna=False).to_frame()
    col_2_counts = col_2_counts.sample(frac=1, random_state=test_random_state)
    col_2_counts = col_2_counts.sort_values(["col_2"], ascending=False, kind='mergesort')
    col_2_samples = col_2_counts.head(encoder.parameters['top_n']).index.tolist()

    expected_col_names = set(["col_2_e", "col_3_a", "col_3_b", "col_4"])
    for val in col_1_samples:
        expected_col_names.add("col_1_" + val)
    for val in col_2_samples:
        expected_col_names.add("col_2_" + val)

    col_names = set(X_t.columns)
    assert (col_names == expected_col_names)


def test_more_top_n_unique_values_large():
    X = pd.DataFrame({"col_1": ["a", "b", "c", "d", "e", "f", "g", "h", "i"],
                      "col_2": ["a", "a", "a", "b", "b", "c", "c", "d", "e"],
                      "col_3": ["a", "a", "a", "b", "b", "b", "c", "c", "d"],
                      "col_4": [2, 0, 1, 3, 0, 1, 2, 4, 1]})

    random_seed = 2
    encoder = OneHotEncoder(random_state=random_seed)
    test_random_state = get_random_state(random_seed)
    encoder.parameters['top_n'] = 3
    encoder.fit(X)
    X_t = encoder.transform(X)
    col_1_counts = X["col_1"].value_counts(dropna=False).to_frame()
    col_1_counts = col_1_counts.sample(frac=1, random_state=test_random_state)
    col_1_counts = col_1_counts.sort_values(["col_1"], ascending=False, kind='mergesort')
    col_1_samples = col_1_counts.head(encoder.parameters['top_n']).index.tolist()
    expected_col_names = set(["col_2_a", "col_2_b", "col_2_c", "col_3_a", "col_3_b", "col_3_c", "col_4"])
    for val in col_1_samples:
        expected_col_names.add("col_1_" + val)

    col_names = set(X_t.columns)
    assert (col_names == expected_col_names)


def test_categorical_dtype():
    # test that columns with the categorical type are encoded properly
    X = pd.DataFrame({"col_1": ["f", "b", "c", "d", "e"],
                      "col_2": ["a", "e", "d", "d", "e"],
                      "col_3": ["a", "a", "a", "a", "a"],
                      "col_4": [3, 3, 2, 2, 1]})
    X["col_4"] = X["col_4"].astype('category')

    encoder = OneHotEncoder()
    encoder.parameters['top_n'] = 5
    encoder.fit(X)
    X_t = encoder.transform(X)

    expected_col_names = set(["col_1_f", "col_1_b", "col_1_c", "col_1_d", "col_1_e",
                              "col_2_d", "col_2_e", "col_2_a", "col_3_a",
                              "col_4_1", "col_4_2", "col_4_3"])
    col_names = set(X_t.columns)
    assert (col_names == expected_col_names)
    assert ([X_t[col].dtype == "uint8" for col in X_t])


def test_all_numerical_dtype():
    # test that columns with the numerical type are preserved
    X = pd.DataFrame({"col_1": [2, 0, 1, 0, 0],
                      "col_2": [3, 2, 5, 1, 3],
                      "col_3": [0, 0, 1, 3, 2],
                      "col_4": [2, 4, 1, 4, 0]})

    encoder = OneHotEncoder()
    encoder.parameters['top_n'] = 5
    encoder.fit(X)
    X_t = encoder.transform(X)
    assert X.equals(X_t)


def test_numpy_input():
    X = np.array([[2, 0, 1, 0, 0], [3, 2, 5, 1, 3]])
    encoder = OneHotEncoder()
    encoder.fit(X)
    X_t = encoder.transform(X)
    assert pd.DataFrame(X).equals(X_t)


def test_large_number_of_categories():
    n_categories = 200000
    frequency_per_category = 5
    X = np.repeat(np.arange(n_categories), frequency_per_category).reshape((-1, 1))
    X_extra = np.repeat(np.arange(10) + n_categories, 10).reshape((-1, 1))
    X = np.array(np.concatenate([X, X_extra]))
    X = pd.DataFrame(X, columns=['cat'])
    X['cat'] = X['cat'].astype('category')
    encoder = OneHotEncoder(top_n=10)
    encoder.fit(X)
    X_t = encoder.transform(X)
    expected_col_names = ['cat_' + str(200000 + i) for i in range(10)]
    assert X_t.shape == (1000100, 10)
    assert set(expected_col_names) == set(list(X_t.columns))
