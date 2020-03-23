import numpy as np
import pandas as pd
import pytest
from evalml.pipelines.components import OneHotEncoder

def test_null_values():
    X = pd.DataFrame([[2, 0, 1, 0], [np.nan,"b","a","c"]])
    encoder = OneHotEncoder()
    with pytest.raises(ValueError, match="Dataframe to be encoded can not contain null values."):
        encoder.transform(X)

def test_less_than_top_n():
    # test that columns with less than n unique values encodes properly
    X = pd.DataFrame()
    X["col_1"] = ["a","b","c","d","d"]
    X["col_2"] = ["a","b","a","c","b"]
    X["col_3"] = ["a","a","a","a","a"]
    X["col_4"] = [2, 0, 1, 0, 0]
    
    encoder = OneHotEncoder()
    encoder.top_n = 5
    X_t = encoder.transform(X)
    expected_col_names = ["col_1_a","col_1_b","col_1_c","col_1_d",
                          "col_2_a","col_2_b","col_2_c", "col_3_a", "col_4"]
    col_names = list(X_t.columns)
    assert (col_names == expected_col_names)
    
def test_more_top_n():
    # test that columns with >= n unique values encodes properly
    X = pd.DataFrame()
    X["col_1"] = ["a","b","c","d","e","f"]
    X["col_2"] = ["a","b","c","d","e","e"]
    X["col_3"] = ["a","a","a","a","a","a"]
    X["col_4"] = [2, 0, 1, 3, 0, 1]

    encoder = OneHotEncoder()
    encoder.top_n = 5
    X_t = encoder.transform(X)
    expected_col_names = ["col_1_a","col_1_b","col_1_c","col_1_d", "col_1_e", "col_1_f",
                          "col_2_e","col_2_a","col_2_b", "col_2_c", "col_4"]
    X_t = encoder.transform(X)
    # print (X_t)

def test_categorical():
    # test that columns with the categorical type are encoded properly
    X = pd.DataFrame()
    X["col_1"] = ["a","b","c","d","e","f","g","h","i","k","l"]
    X["col_2"] = ["a","b","c","d","e","f","g","h","i","k","a"]
    X["col_3"] = ["a","a","a","a","a","a","a","a","a","a","a"]
    X["col_4"] = [2, 0, 1, 0, 0, 1, 3, 2, 5, 1, 3]
    X["col_4"] = X["col_4"].astype('category')

    encoder = OneHotEncoder()
    X_t = encoder.transform(X)
    # print (X_t)
