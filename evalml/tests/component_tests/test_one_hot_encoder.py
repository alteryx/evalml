import numpy as np
import pandas as pd
import pytest

from evalml.pipelines.components import OneHotEncoder


def test_fit_first():
    encoder = OneHotEncoder()
    with pytest.raises(RuntimeError, match="You must fit one hot encoder before calling transform!"):
        encoder.transform(pd.DataFrame())


def test_null_values_in_dataframe():
    X = pd.DataFrame()
    X["col_1"] = ["a", "b", "c", "d", np.nan]
    X["col_2"] = ["a", "b", "a", "c", "b"]
    X["col_3"] = ["a", "a", "a", "a", "a"]
    X["col_4"] = [2, 0, 1, 0, 0]
    encoder = OneHotEncoder()
    encoder.fit(X)
    X_t = encoder.transform(X)
    expected_col_names = set(["col_1_a", "col_1_b", "col_1_c", "col_1_d", "col_1_nan",
                              "col_2_a", "col_2_b", "col_2_c", "col_3_a", "col_4"])
    col_names = set(X_t.columns)
    assert (col_names == expected_col_names)


def test_less_than_top_n_unique_values():
    # test that columns with less than n unique values encodes properly
    X = pd.DataFrame()
    X["col_1"] = ["a", "b", "c", "d", "a"]
    X["col_2"] = ["a", "b", "a", "c", "b"]
    X["col_3"] = ["a", "a", "a", "a", "a"]
    X["col_4"] = [2, 0, 1, 0, 0]

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
    X = pd.DataFrame()
    X["col_1"] = ["a", "b", "c", "d", "e", "f", "g"]
    X["col_2"] = ["a", "c", "d", "b", "e", "e", "f"]
    X["col_3"] = ["a", "a", "a", "a", "a", "a", "b"]
    X["col_4"] = [2, 0, 1, 3, 0, 1, 2]

    encoder = OneHotEncoder()
    encoder.parameters['top_n'] = 5
    encoder.fit(X)
    X_t = encoder.transform(X)
    col_1_counts = X["col_1"].value_counts(dropna=False).to_frame()
    col_1_counts = col_1_counts.sample(frac=1, random_state=encoder.random_state)
    col_1_counts = col_1_counts.sort_values(["col_1"], ascending=False, kind='mergesort')
    col_1_samples = col_1_counts.head(encoder.parameters['top_n']).index.tolist()

    col_2_counts = X["col_2"].value_counts(dropna=False).to_frame()
    col_2_counts = col_2_counts.sample(frac=1, random_state=encoder.random_state)
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
    X = pd.DataFrame()
    X["col_1"] = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
    X["col_2"] = ["a", "a", "a", "b", "b", "c", "c", "d", "e"]
    X["col_3"] = ["a", "a", "a", "b", "b", "b", "c", "c", "d"]
    X["col_4"] = [2, 0, 1, 3, 0, 1, 2, 4, 1]

    encoder = OneHotEncoder()
    encoder.parameters['top_n'] = 3
    encoder.fit(X)
    X_t = encoder.transform(X)
    col_1_counts = X["col_1"].value_counts(dropna=False).to_frame()
    col_1_counts = col_1_counts.sample(frac=1, random_state=encoder.random_state)
    col_1_counts = col_1_counts.sort_values(["col_1"], ascending=False, kind='mergesort')
    col_1_samples = col_1_counts.head(encoder.parameters['top_n']).index.tolist()
    expected_col_names = set(["col_2_a", "col_2_b", "col_2_c", "col_3_a", "col_3_b", "col_3_c", "col_4"])
    for val in col_1_samples:
        expected_col_names.add("col_1_" + val)

    col_names = set(X_t.columns)
    assert (col_names == expected_col_names)


def test_categorical_dtype():
    # test that columns with the categorical type are encoded properly
    X = pd.DataFrame()
    X["col_1"] = ["f", "b", "c", "d", "e"]
    X["col_2"] = ["a", "e", "d", "d", "e"]
    X["col_3"] = ["a", "a", "a", "a", "a"]
    X["col_4"] = [3, 3, 2, 2, 1]
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


def test_all_numerical_dtype():
    # test that columns with the numerical type are preserved
    X = pd.DataFrame()
    X["col_1"] = [2, 0, 1, 0, 0]
    X["col_2"] = [3, 2, 5, 1, 3]
    X["col_3"] = [0, 0, 1, 3, 2]
    X["col_4"] = [2, 4, 1, 4, 0]

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
