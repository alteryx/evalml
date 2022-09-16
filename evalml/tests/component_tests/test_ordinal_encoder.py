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


def test_ordinal_encoder_col_missing():
    X = pd.DataFrame({"col_1": [2, 0, 1, 0, 0], "col_2": ["a", "b", "a", "c", "d"]})

    encoder = OrdinalEncoder(top_n=5, features_to_encode=["col_3", "col_4"])

    with pytest.raises(ValueError, match="Could not find and encode"):
        encoder.fit(X)


def test_ordinal_encoder_is_no_op_for_df_of_non_ordinal_features():
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

    # --> this isn't really testing its ability to recognize ordinals - its testing features to encode
    encoder = OrdinalEncoder(features_to_encode=["col_1"])
    encoder.fit(X)
    assert encoder.features_to_encode == ["col_1"]
    assert encoder.features_to_encode == list(encoder._encoder.feature_names_in_)
    expected_categories = [categories[0]]
    for i, category_list in enumerate(encoder._encoder.categories_):
        assert list(category_list) == expected_categories[i]


# --> test setting non ordinal col in features to encode


def test_ordinal_encoder_categories_set_correctly_from_fit():
    # The SKOrdinalEncoder.categories_ attribute is what determines what gets encoded
    # So we're checking how that gets set during fit
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

    # No parameters specified
    encoder = OrdinalEncoder()
    encoder.fit(X)
    for i, category_list in enumerate(encoder._encoder.categories_):
        assert list(category_list) == categories[i]

    # Categories set explicitly - means top_n must be set to None
    # --> this behavior should be tested elsewhere??
    # encoder = OrdinalEncoder(top_n=None, categories=subset_categories)
    # with pytest.raises(ValueError) as exec_info:
    #     encoder.fit(X)
    # assert "Found unknown categories" in exec_info.value.args[0]

    # Categories set at init explicitly - means we have to set top_n to None
    # and handle the unknown case
    subset_categories = [["a"], ["a"], ["a"]]
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
    # Categories not specified, but top_n specified to limit categories
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
            # Order is not alphabetical
            "col_2": Ordinal(order=["c", "b", "a"]),
            "col_3": "categorical",
        },
    )
    # Test NaN will be counted as a category if within the top_n
    encoder = OrdinalEncoder(handle_missing="as_category")
    encoder.fit(X)
    X_t = encoder.transform(X)
    assert set(X_t.columns) == {
        "col_1_ordinally_encoded",
        "col_2_ordinally_encoded",
        "col_3",
    }
    pd.testing.assert_series_equal(
        X_t["col_1_ordinally_encoded"],
        pd.Series([0, 1, 2, 3, 3], name="col_1_ordinally_encoded", dtype="float64"),
    )
    pd.testing.assert_series_equal(
        X_t["col_2_ordinally_encoded"],
        pd.Series([2, 1, 2, 0, 1], name="col_2_ordinally_encoded", dtype="float64"),
    )


def test_null_values_in_dataframe():
    X = pd.DataFrame(
        {
            "col_1": ["a", "b", "c", "d", np.nan],
            "col_2": [
                "a",
                "b",
                "a",
                "c",
                "c",
            ],  # --> add test where one is none and the other is nan and another is pd.na
            "col_3": ["a", "a", "a", "a", "a"],
        },
    )
    # Note - we cant include the null value in the categories used by Woodwork
    # because it sets the pandas dtypes' categories and they can't include a null value
    categories = [["a", "b", "c", "d"], ["a", "b", "c"], ["a"]]
    X.ww.init(
        logical_types={
            "col_1": Ordinal(order=categories[0]),
            "col_2": Ordinal(order=categories[1]),
        },
    )

    # With no args set, nan doesn't get encoded into any value
    encoder = OrdinalEncoder()
    encoder.fit(X)
    X_t = encoder.transform(X)
    assert pd.isna(X_t["col_1_ordinally_encoded"].iloc[-1])

    # If we handle unknowns with an encoded value, the nan will be set to that value
    encoder = OrdinalEncoder(encoded_missing_value=-1)
    encoder.fit(X)
    X_t = encoder.transform(X)
    assert X_t["col_1_ordinally_encoded"].iloc[-1] == -1

    # --> not sure that this is the desired behavior - in ohe it gets treated as its own category
    # Test NaN doesn't get counted as a category to encode, so it still gets
    # encoded as missing and not unknown even if it's not in the top n
    X = pd.DataFrame(
        {
            "col_1": ["a", "a", "c", "c", np.nan],
            "col_2": ["a", "b", "a", "c", "b"],
            "col_3": ["a", "a", "a", "a", "a"],
            "col_4": [2, 0, 1, np.nan, 0],
        },
    )
    categories = [["a", "b", "c"], ["a", "b", "c"], ["a"]]
    X.ww.init(
        logical_types={
            "col_1": Ordinal(order=categories[0]),
            "col_2": Ordinal(order=categories[1]),
        },
    )
    encoder = OrdinalEncoder(
        top_n=2,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    encoder.fit(X)
    X_t = encoder.transform(X)
    assert pd.isna(X_t["col_1_ordinally_encoded"].iloc[-1])
    assert X_t["col_2_ordinally_encoded"].iloc[3] == -1

    # Test handle_missing='error' throws an error
    # --> not currently an option - should we add?
    # encoder_error = OrdinalEncoder(handle_missing="error")

    # X = pd.DataFrame(
    #     {
    #         "col_1": [np.nan, "b", "c", "d", "e", "f", "g"],
    #         "col_2": ["a", "c", "d", "b", "e", "e", "f"],
    #         "col_3": ["a", "a", "a", "a", "a", "a", "b"],
    #         "col_4": [2, 0, 1, 3, 0, 1, 2],
    #     },
    # )
    # X.ww.init(logical_types={"col_1": "categorical"})
    # with pytest.raises(ValueError, match="Input contains NaN"):
    #     encoder_error.fit(X)


def test_ordinal_encoder_diff_na_types():
    X = pd.DataFrame(
        {
            "col_1": ["a", "b", "c", "d", np.nan],
            "col_2": ["a", "b", "a", "c", None],
            "col_3": ["a", "a", "a", "a", pd.NA],
        },
    )
    categories = [["a", "b", "c", "d"], ["a", "b", "c"], ["a"]]
    X.ww.init(
        logical_types={
            "col_1": Ordinal(order=categories[0]),
            "col_2": Ordinal(order=categories[1]),
            "col_3": Ordinal(order=categories[2]),
        },
    )
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    encoder.fit(X)
    X_t = encoder.transform(X)
    # Confirm all are still null
    assert pd.isna(X_t["col_1_ordinally_encoded"].iloc[-1])
    assert pd.isna(X_t["col_2_ordinally_encoded"].iloc[-1])
    assert pd.isna(X_t["col_3_ordinally_encoded"].iloc[-1])


# --> diff combinations of parameters


def test_handle_unknown():
    X = pd.DataFrame(
        {
            "col_1": ["a", "b", "c", "d", "e", "f", "g"],
            "col_2": ["a", "c", "d", "b", "e", "e", "f"],
            "col_3": ["a", "a", "a", "a", "a", "a", "b"],
            "col_4": [2, 0, 1, 3, 0, 1, 2],
        },
    )
    categories = [
        ["a", "b", "c", "d", "e", "f", "g"],
        ["a", "b", "c", "d", "e", "f"],
        ["a", "b"],
    ]
    X = set_first_three_columns_to_ordinal_with_categories(X, categories=categories)

    encoder = OrdinalEncoder(handle_unknown="error")
    encoder.fit(X)
    assert isinstance(encoder.transform(X), pd.DataFrame)

    X = pd.DataFrame(
        {
            "col_1": ["x", "b", "c", "d", "e", "f", "g"],
            "col_2": ["a", "c", "d", "b", "e", "e", "f"],
            "col_3": ["a", "a", "a", "a", "a", "a", "b"],
            "col_4": [2, 0, 1, 3, 0, 1, 2],
        },
    )
    categories = [
        ["x", "b", "c", "d", "e", "f", "g"],
        ["a", "b", "c", "d", "e", "f"],
        ["a", "b"],
    ]
    X = set_first_three_columns_to_ordinal_with_categories(X, categories=categories)
    with pytest.raises(ValueError) as exec_info:
        # Using the encoder that was fit on data without x
        encoder.transform(X)
    assert "Found unknown categories" in exec_info.value.args[0]


# --> passed in categories have a different sorted order than that of the data itsef - use ordinal order as sourceo f truth and just inpput param as ways to specify what subset


def test_no_top_n():
    # test all categories in all columns are encoded when top_n is None
    X = pd.DataFrame(
        {
            "col_1": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"],
            "col_2": ["a", "c", "d", "b", "e", "e", "f", "a", "b", "c", "d"],
            "col_3": ["a", "a", "a", "a", "a", "a", "b", "a", "a", "b", "b"],
            "col_4": [2, 0, 1, 3, 0, 1, 2, 0, 2, 1, 2],
        },
    )
    X.ww.init(
        logical_types={
            "col_1": Ordinal(
                order=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"],
            ),
            "col_2": Ordinal(order=["a", "b", "c", "d", "e", "f"]),
        },
    )
    expected_col_names = set(
        ["col_3", "col_4", "col_1_ordinally_encoded", "col_2_ordinally_encoded"],
    )

    encoder = OrdinalEncoder(top_n=None, handle_unknown="error", random_seed=2)
    encoder.fit(X)
    X_t = encoder.transform(X)

    col_names = set(X_t.columns)
    assert col_names == expected_col_names

    # Make sure unknown values cause an error
    X_new = pd.DataFrame(
        {
            "col_1": ["a", "b", "c", "x"],
            "col_2": ["a", "c", "d", "b"],
            "col_3": ["a", "a", "a", "a"],
            "col_4": [2, 0, 1, 3],
        },
    )
    X_new.ww.init(
        logical_types={
            "col_1": Ordinal(order=["a", "b", "c", "x"]),
            "col_2": Ordinal(order=["a", "b", "c", "d"]),
        },
    )
    with pytest.raises(ValueError) as exec_info:
        encoder.transform(X_new)
    assert "Found unknown categories" in exec_info.value.args[0]


# def test_categories():
#     X = pd.DataFrame(
#         {
#             "col_1": ["a", "b", "c", "d", "e", "f", "g"],
#             "col_2": ["a", "c", "d", "b", "e", "e", "f"],
#             "col_3": ["a", "a", "a", "a", "a", "a", "b"],
#             "col_4": [2, 0, 1, 3, 0, 1, 2],
#         },
#     )
#     X = set_first_three_columns_to_categorical(X)

#     categories = [["a", "b", "c", "d"], ["a", "b", "c"], ["a", "b"]]

#     # test categories value works
#     encoder = OrdinalEncoder(top_n=None, categories=categories, random_seed=2)
#     encoder.fit(X)
#     X_t = encoder.transform(X)

#     col_names = set(X_t.columns)
#     expected_col_names = set(
#         [
#             "col_1_a",
#             "col_1_b",
#             "col_1_c",
#             "col_1_d",
#             "col_2_a",
#             "col_2_b",
#             "col_2_c",
#             "col_3_a",
#             "col_3_b",
#             "col_4",
#         ],
#     )
#     assert X_t.shape == (7, 10)
#     assert col_names == expected_col_names

#     # test categories with top_n errors
#     with pytest.raises(
#         ValueError,
#         match="Cannot use categories and top_n arguments simultaneously",
#     ):
#         encoder = OrdinalEncoder(top_n=10, categories=categories, random_seed=2)


# def test_less_than_top_n_unique_values():
#     # test that columns with less than n unique values encodes properly
#     X = pd.DataFrame(
#         {
#             "col_1": ["a", "b", "c", "d", "a"],
#             "col_2": ["a", "b", "a", "c", "b"],
#             "col_3": ["a", "a", "a", "a", "a"],
#             "col_4": [2, 0, 1, 0, 0],
#         },
#     )
#     X.ww.init(logical_types={"col_1": "categorical", "col_2": "categorical"})
#     encoder = OrdinalEncoder(top_n=5)
#     encoder.fit(X)
#     X_t = encoder.transform(X)
#     expected_col_names = set(
#         [
#             "col_1_a",
#             "col_1_b",
#             "col_1_c",
#             "col_1_d",
#             "col_2_a",
#             "col_2_b",
#             "col_2_c",
#             "col_3_a",
#             "col_4",
#         ],
#     )
#     col_names = set(X_t.columns)
#     assert col_names == expected_col_names


# def test_more_top_n_unique_values():
#     # test that columns with >= n unique values encodes properly
#     X = pd.DataFrame(
#         {
#             "col_1": ["a", "b", "c", "d", "e", "f", "g"],
#             "col_2": ["a", "c", "d", "b", "e", "e", "f"],
#             "col_3": ["a", "a", "a", "a", "a", "a", "b"],
#             "col_4": [2, 0, 1, 3, 0, 1, 2],
#         },
#     )
#     X = set_first_three_columns_to_categorical(X)

#     random_seed = 2

#     encoder = OrdinalEncoder(top_n=5, random_seed=random_seed)
#     encoder.fit(X)
#     X_t = encoder.transform(X)

#     # Conversion changes the resulting dataframe dtype, resulting in a different random state, so we need make the conversion here too
#     X = infer_feature_types(X)
#     col_1_counts = X["col_1"].value_counts(dropna=False).to_frame()
#     col_1_counts = col_1_counts.sample(frac=1, random_state=random_seed)
#     col_1_counts = col_1_counts.sort_values(
#         ["col_1"],
#         ascending=False,
#         kind="mergesort",
#     )
#     col_1_samples = col_1_counts.head(encoder.parameters["top_n"]).index.tolist()

#     col_2_counts = X["col_2"].value_counts(dropna=False).to_frame()
#     col_2_counts = col_2_counts.sample(frac=1, random_state=random_seed)
#     col_2_counts = col_2_counts.sort_values(
#         ["col_2"],
#         ascending=False,
#         kind="mergesort",
#     )
#     col_2_samples = col_2_counts.head(encoder.parameters["top_n"]).index.tolist()

#     expected_col_names = set(["col_2_e", "col_3_b", "col_4"])
#     for val in col_1_samples:
#         expected_col_names.add("col_1_" + val)
#     for val in col_2_samples:
#         expected_col_names.add("col_2_" + val)

#     col_names = set(X_t.columns)
#     assert col_names == expected_col_names


# def test_more_top_n_unique_values_large():
#     X = pd.DataFrame(
#         {
#             "col_1": ["a", "b", "c", "d", "e", "f", "g", "h", "i"],
#             "col_2": ["a", "a", "a", "b", "b", "c", "c", "d", "e"],
#             "col_3": ["a", "a", "a", "b", "b", "b", "c", "c", "d"],
#             "col_4": [2, 0, 1, 3, 0, 1, 2, 4, 1],
#         },
#     )
#     X = set_first_three_columns_to_categorical(X)
#     random_seed = 2

#     encoder = OrdinalEncoder(top_n=3, random_seed=random_seed)
#     encoder.fit(X)
#     X_t = encoder.transform(X)

#     # Conversion changes the resulting dataframe dtype, resulting in a different random state, so we need make the conversion here too
#     X = infer_feature_types(X)
#     col_1_counts = X["col_1"].value_counts(dropna=False).to_frame()
#     col_1_counts = col_1_counts.sample(frac=1, random_state=random_seed)
#     col_1_counts = col_1_counts.sort_values(
#         ["col_1"],
#         ascending=False,
#         kind="mergesort",
#     )
#     col_1_samples = col_1_counts.head(encoder.parameters["top_n"]).index.tolist()
#     expected_col_names = set(
#         ["col_2_a", "col_2_b", "col_2_c", "col_3_a", "col_3_b", "col_3_c", "col_4"],
#     )
#     for val in col_1_samples:
#         expected_col_names.add("col_1_" + val)

#     col_names = set(X_t.columns)
#     assert col_names == expected_col_names
