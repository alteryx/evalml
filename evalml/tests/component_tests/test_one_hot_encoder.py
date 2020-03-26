import numpy as np
import pandas as pd

from evalml.pipelines.components import OneHotEncoder


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
    encoder.top_n = 5
    encoder.fit(X)
    X_t = encoder.transform(X)
    expected_col_names = set(["col_1_a", "col_1_b", "col_1_c", "col_1_d",
                              "col_2_a", "col_2_b", "col_2_c", "col_3_a", "col_4"])
    col_names = set(X_t.columns)
    assert (col_names == expected_col_names)


def test_more_top_n_unique_values():
    # test that columns with >= n unique values encodes properly
    X = pd.DataFrame()
    X["col_1"] = ["a", "b", "c", "d", "e", "f"]
    X["col_2"] = ["a", "c", "d", "b", "e", "e"]
    X["col_3"] = ["a", "a", "a", "a", "a", "a"]
    X["col_4"] = [2, 0, 1, 3, 0, 1]

    encoder = OneHotEncoder()
    encoder.top_n = 5
    encoder.fit(X)
    X_t = encoder.transform(X)
    expected_col_names = set(["col_1_a", "col_1_b", "col_1_c", "col_1_d", "col_1_e",
                              "col_2_e", "col_2_a", "col_2_b", "col_2_c", "col_2_d",
                              "col_3_a", "col_4"])
    col_names = set(X_t.columns)
    assert (col_names == expected_col_names)


def test_categorical_dtype():
    # test that columns with the categorical type are encoded properly
    X = pd.DataFrame()
    X["col_1"] = ["f", "b", "c", "d", "e", "a"]
    X["col_2"] = ["a", "e", "d", "d", "e", "f"]
    X["col_3"] = ["a", "a", "a", "a", "a", "a"]
    X["col_4"] = [3, 3, 2, 2, 1, 1]
    X["col_4"] = X["col_4"].astype('category')

    encoder = OneHotEncoder()
    encoder.top_n = 5
    encoder.fit(X)
    X_t = encoder.transform(X)
    expected_col_names = set(["col_1_a", "col_1_b", "col_1_c", "col_1_d", "col_1_e",
                              "col_2_d", "col_2_e", "col_2_a", "col_2_f", "col_3_a",
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
    encoder.top_n = 5
    encoder.fit(X)
    X_t = encoder.transform(X)
    assert X.equals(X_t)
