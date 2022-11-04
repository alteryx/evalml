import numpy as np
import pandas as pd
import pytest
from woodwork.logical_types import Ordinal

from evalml.exceptions import ComponentNotYetFittedError
from evalml.pipelines.components import OrdinalEncoder


def set_first_three_columns_to_ordinal_with_categories(X, categories):
    X.ww.init(
        logical_types={
            "col_1": Ordinal(order=categories["col_1"]),
            "col_2": Ordinal(order=categories["col_2"]),
            "col_3": Ordinal(order=categories["col_3"]),
        },
    )
    return X


def test_init():
    parameters = {
        "features_to_encode": None,
        "categories": None,
        "handle_unknown": "error",
        "unknown_value": None,
        "encoded_missing_value": None,
    }
    encoder = OrdinalEncoder()
    assert encoder.parameters == parameters


def test_parameters():
    encoder = OrdinalEncoder(encoded_missing_value=-1)
    expected_parameters = {
        "features_to_encode": None,
        "categories": None,
        "handle_unknown": "error",
        "unknown_value": None,
        "encoded_missing_value": -1,
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
        {"col_1": ["a", "b", "c", "d"], "col_2": ["a", "b", "c"], "col_3": ["a"]},
    )
    encoder = OrdinalEncoder(categories={"col_1": ["a", "b"], "col_2": ["a", "c"]})
    error_msg = "Categories argument must contain as many elements as there are features to encode."
    with pytest.raises(ValueError, match=error_msg):
        encoder.fit(X)

    encoder = OrdinalEncoder(
        categories={"col_1": 1, "col_2": ["a", "c"], "col_3": ["a", "b"]},
    )
    error_msg = "Each of the values in the categories argument must be a list."
    with pytest.raises(ValueError, match=error_msg):
        encoder.fit(X)


def test_categories_list_not_passed_in_for_non_ordinal_column():
    """We indicate that the categories argument must contain categories only for each ordinal
    feature, so test the case where we pass in a categories list when not every column is ordinal.
    """
    X = pd.DataFrame(
        {
            "col_1": [2, 0, 1, 0, 0],
            "col_2": ["a", "b", "a", "c", "d"],
            "col_3": ["x", "x", "x", "y", "y"],
        },
    )
    X.ww.init(logical_types={"col_2": Ordinal(order=["a", "b", "c", "d"])})

    encoder = OrdinalEncoder(categories={"col_2": ["a", "b", "c", "d"]})
    encoder.fit(X)

    assert len(encoder._component_obj.categories_) == len(encoder.features_to_encode)

    error = 'Feature "col_1" was not provided to ordinal encoder as a training feature'
    with pytest.raises(ValueError, match=error):
        encoder.categories("col_1")

    # When features_to_encode is passed in, confirm the order there doesn't matter
    # in indexing into categories_
    X.ww.init(
        logical_types={
            "col_2": Ordinal(order=["a", "b", "c", "d"]),
            "col_3": Ordinal(order=["x", "y"]),
        },
    )
    encoder = OrdinalEncoder(
        # features_to_encode passed in different order than the dataframe's cols
        features_to_encode=["col_3", "col_2"],
        categories={"col_2": ["a", "b", "c", "d"], "col_3": ["x", "y"]},
    )
    encoder.fit(X)

    assert len(encoder._component_obj.categories_) == len(encoder.features_to_encode)
    set(encoder.categories("col_2")) == {"a", "b", "c", "d"}
    set(encoder.categories("col_3")) == {"x", "y"}

    X_t = encoder.transform(X)
    assert list(X_t["col_2_ordinal_encoding"]) == [0, 1, 0, 2, 3]
    assert list(X_t["col_3_ordinal_encoding"]) == [0, 0, 0, 1, 1]


def test_features_to_encode_non_ordinal_cols():
    X = pd.DataFrame({"col_1": [2, 0, 1, 0, 0], "col_2": ["a", "b", "a", "c", "d"]})
    X.ww.init(logical_types={"col_2": Ordinal(order=["a", "b", "c", "d"])})

    encoder = OrdinalEncoder(features_to_encode=["col_1"])
    error = "Column col_1 specified in features_to_encode is not Ordinal in nature"
    with pytest.raises(TypeError, match=error):
        encoder.fit(X)


def test_categories_specified_not_present_in_data():
    """Make sure that we can handle categories during fit that aren't present in
    the data so that they can be seen during transform. Note that because we fit on the
    Ordinal.order passed in at fit, that order is the source of truth for
    potential categories available at transform. In this test, that means that "x",
    though not in the data at fit, must be in the order in order to not be viewed
    as an unknown value at transform.
    """
    X = pd.DataFrame({"col_1": ["a", "b", "a", "c", "d"]})
    X.ww.init(logical_types={"col_1": Ordinal(order=["a", "b", "c", "d", "x"])})

    encoder = OrdinalEncoder(
        categories={"col_1": ["a", "x"]},
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    encoder.fit(X)
    assert set(encoder.categories("col_1")) == {"a", "x"}

    X_2 = pd.DataFrame({"col_1": ["a", "b", "a", "c", "x"]})
    X_2.ww.init(logical_types={"col_1": Ordinal(order=["a", "b", "c", "d", "x"])})
    X_t = encoder.transform(X_2)
    assert list(X_t["col_1_ordinal_encoding"]) == [0, -1, 0, -1, 1]


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

    categories = {
        "col_1": ["a", "b", "c", "d"],
        "col_2": ["a", "b", "c"],
        "col_3": ["a"],
    }
    X = set_first_three_columns_to_ordinal_with_categories(X, categories=categories)

    encoder = OrdinalEncoder()
    encoder.fit(X)
    assert encoder.features_to_encode == ["col_1", "col_2", "col_3"]
    assert encoder.features_to_encode == list(encoder._component_obj.feature_names_in_)

    encoder = OrdinalEncoder(features_to_encode=["col_1"])
    encoder.fit(X)
    assert encoder.features_to_encode == ["col_1"]
    assert encoder.features_to_encode == list(encoder._component_obj.feature_names_in_)
    expected_categories = {"col_1": categories["col_1"]}
    for i, category_list in enumerate(encoder._component_obj.categories_):
        assert list(category_list) == expected_categories[f"col_{i + 1}"]


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
    categories = {
        "col_1": ["a", "b", "c", "d"],
        "col_2": ["a", "b", "c"],
        "col_3": ["a"],
    }
    X = set_first_three_columns_to_ordinal_with_categories(X, categories=categories)

    # No parameters specified
    encoder = OrdinalEncoder()
    encoder.fit(X)
    for i, category_list in enumerate(encoder._component_obj.categories_):
        assert list(category_list) == categories[f"col_{i + 1}"]

    # Categories set at init explicitly - means we have to handle the unknown case
    subset_categories = {"col_1": ["a"], "col_2": ["a"], "col_3": ["a"]}
    encoder = OrdinalEncoder(
        categories=subset_categories,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    encoder.fit(X)
    for i, category_list in enumerate(encoder._component_obj.categories_):
        assert list(category_list) == subset_categories[f"col_{i + 1}"]


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
    encoder = OrdinalEncoder(handle_missing="as_category")
    encoder.fit(X)
    X_t = encoder.transform(X)
    assert set(X_t.columns) == {
        "col_1_ordinal_encoding",
        "col_2_ordinal_encoding",
        "col_3",
    }
    pd.testing.assert_series_equal(
        X_t["col_1_ordinal_encoding"],
        pd.Series([0, 1, 2, 3, 3], name="col_1_ordinal_encoding", dtype="float64"),
    )
    pd.testing.assert_series_equal(
        X_t["col_2_ordinal_encoding"],
        pd.Series([2, 1, 2, 0, 1], name="col_2_ordinal_encoding", dtype="float64"),
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
            ],
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
    assert pd.isna(X_t["col_1_ordinal_encoding"].iloc[-1])

    # If we handle unknowns with an encoded value, the nan will be set to that value
    encoder = OrdinalEncoder(encoded_missing_value=-1)
    encoder.fit(X)
    X_t = encoder.transform(X)
    assert X_t["col_1_ordinal_encoding"].iloc[-1] == -1


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
    encoder = OrdinalEncoder(encoded_missing_value=-1)
    encoder.fit(X)
    X_t = encoder.transform(X)
    # Confirm were recognized as null and encoded
    assert X_t["col_1_ordinal_encoding"].iloc[-1] == -1
    assert X_t["col_2_ordinal_encoding"].iloc[-1] == -1
    assert X_t["col_3_ordinal_encoding"].iloc[-1] == -1


def test_null_values_with_categories_specified():
    """Nans aren't treated by Woodwork as categories in ordinal cols, so they shouldn't
    have an impact on the categories parameter and be handled entirely independently."""
    X = pd.DataFrame(
        {
            "col_1": ["a", "b", "c", np.nan, np.nan],
            "col_2": [
                "a",
                "b",
                "a",
                "c",
                np.nan,
            ],
            "col_3": ["a", "a", "a", "a", "a"],
        },
    )
    # Note - we cant include the null value in the categories used by Woodwork
    # because it sets the pandas dtypes' categories and they can't include a null value
    categories = [["a", "b", "c", "d"], ["a", "b", "c"]]
    X.ww.init(
        logical_types={
            "col_1": Ordinal(order=categories[0]),
            "col_2": Ordinal(order=categories[1]),
        },
    )

    # Try putting a nan in the categories list in one of the columns but not the other
    encoder = OrdinalEncoder(
        categories={"col_1": ["a"], "col_2": ["a", np.nan]},
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    encoder.fit(X)
    X_t = encoder.transform(X)
    # Check that the null values were handled as missing even when they're present in categories
    assert pd.isna(X_t["col_1_ordinal_encoding"].iloc[-1])
    assert pd.isna(X_t["col_1_ordinal_encoding"].iloc[-2])
    assert pd.isna(X_t["col_2_ordinal_encoding"].iloc[-1])


def test_handle_unknown():
    X = pd.DataFrame(
        {
            "col_1": ["a", "b", "c", "d", "e", "f", "g"],
            "col_2": ["a", "c", "d", "b", "e", "e", "f"],
            "col_3": ["a", "a", "a", "a", "a", "a", "b"],
            "col_4": [2, 0, 1, 3, 0, 1, 2],
        },
    )
    categories = {
        "col_1": ["a", "b", "c", "d", "e", "f", "g"],
        "col_2": ["a", "b", "c", "d", "e", "f"],
        "col_3": ["a", "b"],
    }
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
    categories = {
        "col_1": ["x", "b", "c", "d", "e", "f", "g"],
        "col_2": ["a", "b", "c", "d", "e", "f"],
        "col_3": ["a", "b"],
    }
    X = set_first_three_columns_to_ordinal_with_categories(X, categories=categories)
    with pytest.raises(ValueError) as exec_info:
        # Using the encoder that was fit on data without x
        encoder.transform(X)
    assert "Found unknown categories" in exec_info.value.args[0]


def test_categories_set_at_init():
    X = pd.DataFrame(
        {
            "col_1": ["a", "b", "c", "d", "e", "f", "g"],
            "col_2": ["a", "c", "d", "b", "e", "e", "f"],
            "col_3": ["a", "a", "a", "a", "a", "a", "b"],
            "col_4": [2, 0, 1, 3, 0, 1, 2],
        },
    )

    full_categories = {
        "col_1": ["a", "b", "c", "d", "e", "f", "g"],
        "col_2": ["a", "b", "c", "d", "e", "f"],
        "col_3": ["a", "b"],
    }
    X = set_first_three_columns_to_ordinal_with_categories(
        X,
        categories=full_categories,
    )

    categories = {
        "col_1": ["a", "b", "c", "d"],
        "col_2": ["a", "b", "c"],
        "col_3": ["a", "b"],
    }

    # test categories value works when transforming
    encoder = OrdinalEncoder(
        categories=categories,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        random_seed=2,
    )
    encoder.fit(X)
    X_t = encoder.transform(X)

    assert list(X_t["col_1_ordinal_encoding"]) == [0, 1, 2, 3, -1, -1, -1]
    assert list(X_t["col_2_ordinal_encoding"]) == [0, 2, -1, 1, -1, -1, -1]
    assert list(X_t["col_3_ordinal_encoding"]) == [0, 0, 0, 0, 0, 0, 1]


def test_categories_includes_not_present_value():
    """This tests the case where the categories we pass into the encoder include
    values that aren't in the data or even the Ordinal.order."""
    X = pd.DataFrame(
        {
            "col_1": ["a", "b", "c", "d", "e", "f", "g"],
            "col_2": ["a", "c", "d", "b", "e", "e", "f"],
            "col_3": ["a", "a", "a", "a", "a", "a", "b"],
            "col_4": [2, 0, 1, 3, 0, 1, 2],
        },
    )
    full_categories = {
        "col_1": ["a", "b", "c", "d", "e", "f", "g"],
        "col_2": ["a", "b", "c", "d", "e", "f"],
        "col_3": ["a", "b"],
    }
    X = set_first_three_columns_to_ordinal_with_categories(
        X,
        categories=full_categories,
    )

    # Categories passed in has value "x" that's not in the data
    categories = {"col_1": ["a", "x"], "col_2": ["a", "x"], "col_3": ["a", "x"]}

    # test categories value works when transforming
    encoder = OrdinalEncoder(
        categories=categories,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        random_seed=2,
    )
    encoder.fit(X)
    assert set(encoder.categories("col_1")) == {"a"}
    assert set(encoder.categories("col_2")) == {"a"}
    assert set(encoder.categories("col_3")) == {"a"}


def test_categories_different_order_from_ltype():
    # The order of categories comes from the Ordinal.order property of the data.
    # Categories passed in as input to the encoder just determine what subset should
    # be used.
    X = pd.DataFrame(
        {
            "col_1": ["a", "b", "c", "d", "e", "f", "g"],
            "col_2": ["a", "c", "d", "b", "e", "e", "f"],
            "col_3": ["a", "a", "a", "a", "a", "a", "b"],
            "col_4": [2, 0, 1, 3, 0, 1, 2],
        },
    )
    full_categories = {
        "col_1": ["a", "b", "c", "d", "e", "f", "g"],
        "col_2": ["a", "b", "c", "d", "e", "f"],
        "col_3": ["a", "b"],
    }
    X = set_first_three_columns_to_ordinal_with_categories(
        X,
        categories=full_categories,
    )

    # The order doesn't match the full categories above but outputted data will still match above
    categories = {
        "col_1": ["d", "a", "c", "b"],
        "col_2": ["c", "b", "a"],
        "col_3": ["b", "a"],
    }

    # test categories value works when transforming
    encoder = OrdinalEncoder(
        categories=categories,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        random_seed=2,
    )
    encoder.fit(X)
    X_t = encoder.transform(X)

    assert list(X_t["col_1_ordinal_encoding"]) == [0, 1, 2, 3, -1, -1, -1]
    assert list(X_t["col_2_ordinal_encoding"]) == [0, 2, -1, 1, -1, -1, -1]
    assert list(X_t["col_3_ordinal_encoding"]) == [0, 0, 0, 0, 0, 0, 1]


@pytest.mark.parametrize(
    "index",
    [
        list(range(-5, 0)),
        list(range(100, 105)),
        [f"row_{i}" for i in range(5)],
        pd.date_range("2020-09-08", periods=5),
    ],
)
def test_ordinal_encoder_preserves_custom_index(index):
    df = pd.DataFrame(
        {"categories": [f"cat_{i}" for i in range(5)], "numbers": np.arange(5)},
        index=index,
    )
    encoder = OrdinalEncoder()
    new_df = encoder.fit_transform(df)
    pd.testing.assert_index_equal(new_df.index, df.index)
    assert not new_df.isna().any(axis=None)


def test_ordinal_encoder_categories():
    X = pd.DataFrame(
        {"col_1": ["a"] * 10, "col_2": ["a"] * 3 + ["b"] * 3 + ["c"] * 2 + ["d"] * 2},
    )
    X.ww.init(
        logical_types={
            "col_1": Ordinal(order=["a"]),
            "col_2": Ordinal(order=["a", "b", "c", "d"]),
        },
    )
    encoder = OrdinalEncoder(
        categories={"col_1": ["a"], "col_2": ["a", "b"]},
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    with pytest.raises(
        ComponentNotYetFittedError,
        match="This OrdinalEncoder is not fitted yet. You must fit OrdinalEncoder before calling categories.",
    ):
        encoder.categories("col_1")

    encoder.fit(X)
    np.testing.assert_array_equal(encoder.categories("col_1"), np.array(["a"]))
    np.testing.assert_array_equal(encoder.categories("col_2"), np.array(["a", "b"]))
    with pytest.raises(
        ValueError,
        match='Feature "col_12345" was not provided to ordinal encoder as a training feature',
    ):
        encoder.categories("col_12345")


def test_ordinal_encoder_get_feature_names():
    X = pd.DataFrame(
        {"col_1": ["a"] * 10, "col_2": ["a"] * 3 + ["b"] * 3 + ["c"] * 2 + ["d"] * 2},
    )
    X.ww.init(
        logical_types={
            "col_1": Ordinal(order=["a"]),
            "col_2": Ordinal(order=["a", "b", "c", "d"]),
        },
    )
    ordinal_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    with pytest.raises(
        ComponentNotYetFittedError,
        match="This OrdinalEncoder is not fitted yet. You must fit OrdinalEncoder before calling get_feature_names.",
    ):
        ordinal_encoder.get_feature_names()
    ordinal_encoder.fit(X)
    np.testing.assert_array_equal(
        ordinal_encoder.get_feature_names(),
        np.array(["col_1_ordinal_encoding", "col_2_ordinal_encoding"]),
    )

    ordinal_encoder = OrdinalEncoder(features_to_encode=["col_2"])
    ordinal_encoder.fit(X)
    np.testing.assert_array_equal(
        ordinal_encoder.get_feature_names(),
        np.array(["col_2_ordinal_encoding"]),
    )


def test_ordinal_encoder_features_to_encode():
    # Test feature that doesn't need encoding and
    # feature that needs encoding but is not specified remain untouched
    X = pd.DataFrame({"col_1": [2, 0, 1, 0, 0], "col_2": ["a", "b", "a", "c", "d"]})
    X.ww.init(
        logical_types={
            "col_1": Ordinal(order=[0, 1, 2]),
            "col_2": Ordinal(order=["a", "b", "c", "d"]),
        },
    )
    encoder = OrdinalEncoder(features_to_encode=["col_1"])
    encoder.fit(X)
    X_t = encoder.transform(X)
    expected_col_names = set(["col_2", "col_1_ordinal_encoding"])
    col_names = set(X_t.columns)
    assert col_names == expected_col_names
    assert [X_t[col].dtype == "uint8" for col in X_t]

    encoder = OrdinalEncoder(features_to_encode=["col_1", "col_2"])
    encoder.fit(X)
    X_t = encoder.transform(X)
    expected_col_names = set(
        ["col_1_ordinal_encoding", "col_2_ordinal_encoding"],
    )
    col_names = set(X_t.columns)
    assert col_names == expected_col_names
    assert [X_t[col].dtype == "uint8" for col in X_t]


def test_ordinal_encoder_features_to_encode_col_missing():
    X = pd.DataFrame({"col_1": [2, 0, 1, 0, 0], "col_2": ["a", "b", "a", "c", "d"]})
    X.ww.init(
        logical_types={
            "col_1": Ordinal(order=[0, 1, 2]),
            "col_2": Ordinal(order=["a", "b", "c", "d"]),
        },
    )

    encoder = OrdinalEncoder(features_to_encode=["col_3", "col_4"])

    with pytest.raises(ValueError, match="Could not find and encode"):
        encoder.fit(X)


def test_ordinal_encoder_features_to_encode_no_col_names():
    X = pd.DataFrame([["b", 0], ["a", 1], ["b", 1]])
    X.ww.init(
        logical_types={
            0: Ordinal(order=["b", "a"]),
            1: Ordinal(order=[0, 1]),
        },
    )
    encoder = OrdinalEncoder(features_to_encode=[0])
    encoder.fit(X)
    X_t = encoder.transform(X)
    expected_col_names = set([1, "0_ordinal_encoding"])
    col_names = set(X_t.columns)
    assert col_names == expected_col_names
    assert [X_t[col].dtype == "uint8" for col in X_t]


def test_ordinal_encoder_output_doubles():
    X = pd.DataFrame(
        {
            "bool": [bool(i % 2) for i in range(100)],
            "categorical": ["dog"] * 20 + ["cat"] * 40 + ["fish"] * 40,
            "integers": [i for i in range(100)],
            "doubles": [i * 1.0 for i in range(100)],
        },
    )
    X.ww.init(
        logical_types={
            "categorical": Ordinal(order=["dog", "cat", "fish"]),
        },
    )
    y = pd.Series([i % 2 for i in range(100)])
    y.ww.init()
    ordinal_encoder = OrdinalEncoder()
    output = ordinal_encoder.fit_transform(X, y)
    for name, types in output.ww.types["Logical Type"].items():
        if name == "integers":
            assert str(types) == "Integer"
        elif name == "bool":
            assert str(types) == "Boolean"
        else:
            assert str(types) == "Double"
    assert len(output.columns) == len(X.columns)


@pytest.mark.parametrize("data_type", ["list", "np", "pd", "ww"])
def test_data_types(data_type):
    if data_type == "list":
        X = [["a"], ["b"], ["c"]] * 5
    elif data_type == "np":
        X = np.array([["a"], ["b"], ["c"]] * 5)
    elif data_type == "pd":
        X = pd.DataFrame(["a", "b", "c"] * 5, columns=[0])
    elif data_type == "ww":
        X = pd.DataFrame(["a", "b", "c"] * 5)
        X.ww.init(
            logical_types={
                0: Ordinal(order=["a", "b", "c"]),
            },
        )
    encoder = OrdinalEncoder()
    encoder.fit(X)
    X_t = encoder.transform(X)

    if data_type != "ww":
        # Woodwork wont infer Ordinal, so none of the other types will encode features
        assert not encoder.features_to_encode
        expected_df = pd.DataFrame(
            [["a"], ["b"], ["c"]] * 5,
            columns=[0],
            dtype="category",
        )
        pd.testing.assert_frame_equal(X_t, expected_df)
    else:
        assert list(X_t.columns) == ["0_ordinal_encoding"]
        expected_df = pd.DataFrame(
            [[0], [1], [2]] * 5,
            columns=["0_ordinal_encoding"],
            dtype="float64",
        )
        pd.testing.assert_frame_equal(X_t, expected_df)
