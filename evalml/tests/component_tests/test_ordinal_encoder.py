import re

import numpy as np
import pandas as pd
import pytest
from woodwork.logical_types import Ordinal

from evalml.pipelines.components import OrdinalEncoder


def set_first_three_columns_to_ordinal_with_categories(X, categories):
    X.ww.init(
        logical_types={
            "col_1": Ordinal(order=categories[0]),
            "col_2": Ordinal(order=categories[1]),
            "col_3": Ordinal(order=categories[2]),
        },
    )
    return X


def test_init():
    parameters = {
        "top_n": 10,
        "features_to_encode": None,
        "categories": None,
        "handle_unknown": "error",
        "unknown_value": None,
        "encoded_missing_value": np.nan,
    }
    encoder = OrdinalEncoder()
    assert encoder.parameters == parameters


def test_parameters():
    encoder = OrdinalEncoder(top_n=123)
    expected_parameters = {
        "top_n": 123,
        "features_to_encode": None,
        "categories": None,
        "handle_unknown": "error",
        "unknown_value": None,
        "encoded_missing_value": np.nan,
    }
    assert encoder.parameters == expected_parameters


def test_invalid_inputs():
    error_msg = "Invalid input {} for handle_unknown".format("bananas")
    with pytest.raises(ValueError, match=error_msg):
        encoder = OrdinalEncoder(handle_unknown="bananas")

    error_msg = (
        "To use encoded value for unknown categories, unknown_value must"
        "be specified as either np.nan or as an int that is distinct from"
        "the other encoded categories "
    )
    with pytest.raises(ValueError, match=error_msg):
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value")

    X = pd.DataFrame(
        {
            "col_1": ["a", "b", "c", "d", "a"],
            "col_2": ["a", "b", "a", "c", "b"],
            "col_3": ["a", "a", "a", "a", "a"],
        },
    )
    X = set_first_three_columns_to_ordinal_with_categories(
        X,
        [["a", "b", "c", "d"], ["a", "b", "c"], ["a"]],
    )
    encoder = OrdinalEncoder(top_n=None, categories=[["a", "b"], ["a", "c"]])
    error_msg = "Categories argument must contain a list of categories for each categorical feature"
    with pytest.raises(ValueError, match=error_msg):
        encoder.fit(X)

    encoder = OrdinalEncoder(top_n=None, categories=["a", "b", "c"])
    error_msg = "Categories argument must contain a list of categories for each categorical feature"
    with pytest.raises(ValueError, match=error_msg):
        encoder.fit(X)

    categories = [["a", "b", "c", "d"], ["a", "b", "c"], ["a", "b"]]
    with pytest.raises(
        ValueError,
        match="Cannot use categories and top_n arguments simultaneously",
    ):
        OrdinalEncoder(top_n=11, categories=categories, random_seed=2)


# --> test no transformation when ordinal type not set and hyes when explicitly set
# --> test feats to encode includes non ordinals


def test_ohe_features_to_encode_col_missing():
    X = pd.DataFrame({"col_1": [2, 0, 1, 0, 0], "col_2": ["a", "b", "a", "c", "d"]})

    encoder = OrdinalEncoder(top_n=5, features_to_encode=["col_3", "col_4"])

    with pytest.raises(ValueError, match="Could not find and encode"):
        encoder.fit(X)


def test_ordinal_encoder_is_no_op_for_not_ordinal_features():

    encoder = OrdinalEncoder(handle_missing="error")
    X = pd.DataFrame(
        {
            "col_1": [1.2, 3.2, None, 4.7],
            "col_2": [4.5, 8.9, 11.2, 23.4],
            "col_3": [True, False, True, True],
            "col_4": [
                "a",
                "b",
                "a",
                "c",
            ],
        },
    )
    X.ww.init(
        logical_types={
            "col_1": "Double",
            "col_2": "Integer",
            "col_3": "Boolean",
            "col_4": "Categorical",
        },
    )
    X_t = encoder.fit_transform(X)
    # --> need transform implemented for this to mean anything
    pd.testing.assert_frame_equal(X_t, X)


def test_ordinal_encoder_recognizes_ordinal_columns():
    X = pd.DataFrame(
        {
            "col_1": ["a", "b", "c", "d", "a"],
            "col_2": ["a", "b", "b", "c", "b"],
            "col_3": ["a", "a", "a", "a", "a"],
            "col_4": [1, 2, 3, 4, 5],
        },
    )
    encoder = OrdinalEncoder()
    encoder.fit(X)
    assert not encoder.features_to_encode

    categories = [["a", "b", "c", "d"], ["a", "b", "c"], ["a"]]
    X = set_first_three_columns_to_ordinal_with_categories(X, categories=categories)

    encoder = OrdinalEncoder()
    encoder.fit(X)
    assert encoder.features_to_encode == ["col_1", "col_2", "col_3"]
    assert encoder.features_to_encode == list(encoder._encoder.feature_names_in_)

    encoder = OrdinalEncoder(features_to_encode=["col_1"])
    encoder.fit(X)
    assert encoder.features_to_encode == ["col_1"]
    assert encoder.features_to_encode == list(encoder._encoder.feature_names_in_)
    expected_categories = [categories[0]]
    for i, category_list in enumerate(encoder._encoder.categories_):
        assert list(category_list) == expected_categories[i]


def test_ordinal_encoder_categories_set_correctly():
    X = pd.DataFrame(
        {
            "col_1": ["a", "b", "c", "d", "a"],
            "col_2": ["a", "b", "b", "c", "b"],
            "col_3": ["a", "a", "a", "a", "a"],
            "col_4": [1, 2, 3, 4, 5],
        },
    )
    categories = [["a", "b", "c", "d"], ["a", "b", "c"], ["a"]]
    X = set_first_three_columns_to_ordinal_with_categories(X, categories=categories)

    encoder = OrdinalEncoder()
    encoder.fit(X)
    for i, category_list in enumerate(encoder._encoder.categories_):
        assert list(category_list) == categories[i]

    subset_categories = [["a"], ["a"], ["a"]]
    encoder = OrdinalEncoder(top_n=None, categories=subset_categories)
    with pytest.raises(ValueError) as exec_info:
        encoder.fit(X)
    assert "Found unknown categories" in exec_info.value.args[0]

    encoder = OrdinalEncoder(
        top_n=None,
        categories=subset_categories,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    encoder.fit(X)
    for i, category_list in enumerate(encoder._encoder.categories_):
        assert list(category_list) == subset_categories[i]

    # --> feels weird that you have to supply these values  when just topn is set
    # --> do we need to mention tie behavior for top_n?
    encoder = OrdinalEncoder(
        top_n=1,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    encoder.fit(X)
    expected_categories = [["a"], ["b"], ["a"]]
    for i, category_list in enumerate(encoder._encoder.categories_):
        assert list(category_list) == expected_categories[i]


# --> test feature names
# --> test encoded feature values
# --> null values


def test_ordinal_encoder_transform():
    X = pd.DataFrame(
        {
            "col_1": ["a", "b", "c", "d", "d"],
            "col_2": ["a", "b", "a", "c", "b"],
            "col_3": ["a", "a", "a", "a", "a"],
        },
    )
    X.ww.init(
        logical_types={
            "col_1": Ordinal(order=["a", "b", "c", "d"]),
            "col_2": Ordinal(order=["a", "b", "c"]),
            "col_3": "categorical",
        },
    )
    # Test NaN will be counted as a category if within the top_n
    encoder = OrdinalEncoder(handle_missing="as_category")
    encoder.fit(X)
    X_t = encoder.transform(X)
    assert False
