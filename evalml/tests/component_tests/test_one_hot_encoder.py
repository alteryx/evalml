import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from woodwork.exceptions import TypeConversionError
from woodwork.logical_types import (
    Boolean,
    Categorical,
    Datetime,
    Double,
    Integer,
    NaturalLanguage,
)

from evalml.exceptions import ComponentNotYetFittedError
from evalml.pipelines.components import OneHotEncoder
from evalml.utils import get_random_seed, infer_feature_types


def set_first_three_columns_to_categorical(X):
    X.ww.init(
        logical_types={
            "col_1": "categorical",
            "col_2": "categorical",
            "col_3": "categorical",
        }
    )
    return X


def test_init():
    parameters = {
        "top_n": 10,
        "features_to_encode": None,
        "categories": None,
        "drop": "if_binary",
        "handle_unknown": "ignore",
        "handle_missing": "error",
    }
    encoder = OneHotEncoder()
    assert encoder.parameters == parameters


def test_parameters():
    encoder = OneHotEncoder(top_n=123)
    expected_parameters = {
        "top_n": 123,
        "features_to_encode": None,
        "categories": None,
        "drop": "if_binary",
        "handle_unknown": "ignore",
        "handle_missing": "error",
    }
    assert encoder.parameters == expected_parameters


def test_invalid_inputs():
    error_msg = "Invalid input {} for handle_missing".format("peanut butter")
    with pytest.raises(ValueError, match=error_msg):
        encoder = OneHotEncoder(handle_missing="peanut butter")

    error_msg = "Invalid input {} for handle_unknown".format("bananas")
    with pytest.raises(ValueError, match=error_msg):
        encoder = OneHotEncoder(handle_unknown="bananas")

    X = pd.DataFrame(
        {
            "col_1": ["a", "b", "c", "d", "a"],
            "col_2": ["a", "b", "a", "c", "b"],
            "col_3": ["a", "a", "a", "a", "a"],
        }
    )
    encoder = OneHotEncoder(top_n=None, categories=[["a", "b"], ["a", "c"]])
    error_msg = "Categories argument must contain a list of categories for each categorical feature"
    with pytest.raises(ValueError, match=error_msg):
        encoder.fit(X)

    encoder = OneHotEncoder(top_n=None, categories=["a", "b", "c"])
    error_msg = "Categories argument must contain a list of categories for each categorical feature"
    with pytest.raises(ValueError, match=error_msg):
        encoder.fit(X)


def test_null_values_in_dataframe():
    X = pd.DataFrame(
        {
            "col_1": ["a", "b", "c", "d", np.nan],
            "col_2": ["a", "b", "a", "c", "b"],
            "col_3": ["a", "a", "a", "a", "a"],
        }
    )
    X.ww.init(logical_types={"col_1": "categorical", "col_2": "categorical"})
    # Test NaN will be counted as a category if within the top_n
    encoder = OneHotEncoder(handle_missing="as_category")
    encoder.fit(X)
    X_t = encoder.transform(X)

    expected_col_names = set(
        [
            "col_1_a",
            "col_1_b",
            "col_1_c",
            "col_1_d",
            "col_1_nan",
            "col_2_a",
            "col_2_b",
            "col_2_c",
            "col_3_a",
        ]
    )
    col_names = set(X_t.columns)
    assert col_names == expected_col_names
    assert X_t.shape == (5, 9)

    # Test NaN will not be counted as a category if not in the top_n
    X = pd.DataFrame(
        {
            "col_1": ["a", "a", "c", "c", np.nan],
            "col_2": ["a", "b", "a", "c", "b"],
            "col_3": ["a", "a", "a", "a", "a"],
            "col_4": [2, 0, 1, np.nan, 0],
        }
    )
    X.ww.init(logical_types={"col_1": "categorical", "col_2": "categorical"})
    encoder = OneHotEncoder(top_n=2, handle_missing="as_category")
    encoder.fit(X)
    X_t = encoder.transform(X)

    expected_col_names = set(
        ["col_1_a", "col_1_c", "col_2_a", "col_2_b", "col_3_a", "col_4"]
    )
    col_names = set(X_t.columns)
    assert col_names == expected_col_names
    assert X_t.shape == (5, 6)

    # Test handle_missing='error' throws an error
    encoder = OneHotEncoder(handle_missing="error")

    X = pd.DataFrame(
        {
            "col_1": [np.nan, "b", "c", "d", "e", "f", "g"],
            "col_2": ["a", "c", "d", "b", "e", "e", "f"],
            "col_3": ["a", "a", "a", "a", "a", "a", "b"],
            "col_4": [2, 0, 1, 3, 0, 1, 2],
        }
    )

    with pytest.raises(ValueError, match="Input contains NaN"):
        encoder.fit(X)

    # Test NaN values in transformed data
    X = pd.DataFrame(
        {
            "col_1": ["a", "b", "c", "d", "d"],
            "col_2": ["a", "b", "a", "c", "b"],
            "col_3": ["a", "a", "a", "a", "a"],
        }
    )
    encoder = OneHotEncoder(handle_missing="error")
    encoder.fit(X)
    X_missing = pd.DataFrame(
        {
            "col_1": ["a", "b", "c", "d", "d"],
            "col_2": ["a", "b", np.nan, "c", "b"],
            "col_3": ["a", "a", "a", "a", "a"],
        }
    )
    with pytest.raises(ValueError, match="Input contains NaN"):
        encoder.transform(X_missing)


def test_drop_first():
    X = pd.DataFrame(
        {
            "col_1": ["a", "b", "c", "d", "d"],
            "col_2": ["a", "b", "a", "c", "b"],
            "col_3": ["a", "a", "a", "a", "a"],
        }
    )
    X = set_first_three_columns_to_categorical(X)
    encoder = OneHotEncoder(top_n=None, drop="first", handle_unknown="error")
    encoder.fit(X)
    X_t = encoder.transform(X)
    col_names = set(X_t.columns)
    expected_col_names = set(["col_1_b", "col_1_c", "col_1_d", "col_2_b", "col_2_c"])
    assert col_names == expected_col_names


def test_drop_binary():
    X = pd.DataFrame(
        {
            "col_1": ["a", "b", "b", "a", "b"],
            "col_2": ["a", "b", "a", "c", "b"],
            "col_3": ["a", "a", "a", "a", "a"],
        }
    )
    X = set_first_three_columns_to_categorical(X)
    encoder = OneHotEncoder(top_n=None, drop="if_binary", handle_unknown="error")
    encoder.fit(X)
    X_t = encoder.transform(X)
    col_names = set(X_t.columns)
    expected_col_names = set(["col_1_a", "col_2_a", "col_2_b", "col_2_c", "col_3_a"])
    assert col_names == expected_col_names


def test_drop_parameter_is_array():
    X = pd.DataFrame(
        {
            "col_1": ["a", "b", "b", "a", "b"],
            "col_2": ["a", "b", "a", "c", "b"],
            "col_3": ["a", "a", "a", "a", "a"],
        }
    )
    X = set_first_three_columns_to_categorical(X)
    encoder = OneHotEncoder(top_n=None, drop=["b", "c", "a"], handle_unknown="error")
    encoder.fit(X)
    X_t = encoder.transform(X)
    col_names = set(X_t.columns)
    expected_col_names = {"col_1_a", "col_2_a", "col_2_b"}
    assert col_names == expected_col_names


def test_drop_binary_and_top_n_2():
    # Test that columns that originally had two values have one column dropped,
    # but columns that end up with two values keep both values
    X = pd.DataFrame(
        {
            "col_1": ["a", "b", "b", "a", "b"],
            "col_2": ["a", "b", "a", "c", "b"],
            "col_3": ["a", "a", "a", "a", "a"],
        }
    )
    X = set_first_three_columns_to_categorical(X)
    encoder = OneHotEncoder(top_n=2, drop="if_binary")
    encoder.fit(X)
    X_t = encoder.transform(X)
    col_names = set(X_t.columns)
    expected_col_names = set(["col_1_a", "col_2_a", "col_2_b", "col_3_a"])
    assert col_names == expected_col_names


def test_handle_unknown():
    X = pd.DataFrame(
        {
            "col_1": ["a", "b", "c", "d", "e", "f", "g"],
            "col_2": ["a", "c", "d", "b", "e", "e", "f"],
            "col_3": ["a", "a", "a", "a", "a", "a", "b"],
            "col_4": [2, 0, 1, 3, 0, 1, 2],
        }
    )
    X = set_first_three_columns_to_categorical(X)
    encoder = OneHotEncoder(handle_unknown="error")
    encoder.fit(X)
    assert isinstance(encoder.transform(X), pd.DataFrame)

    X = pd.DataFrame(
        {
            "col_1": ["x", "b", "c", "d", "e", "f", "g"],
            "col_2": ["a", "c", "d", "b", "e", "e", "f"],
            "col_3": ["a", "a", "a", "a", "a", "a", "b"],
            "col_4": [2, 0, 1, 3, 0, 1, 2],
        }
    )
    with pytest.raises(ValueError) as exec_info:
        encoder.transform(X)
    assert "Found unknown categories" in exec_info.value.args[0]


def test_no_top_n():
    # test all categories in all columns are encoded when top_n is None
    X = pd.DataFrame(
        {
            "col_1": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"],
            "col_2": ["a", "c", "d", "b", "e", "e", "f", "a", "b", "c", "d"],
            "col_3": ["a", "a", "a", "a", "a", "a", "b", "a", "a", "b", "b"],
            "col_4": [2, 0, 1, 3, 0, 1, 2, 0, 2, 1, 2],
        }
    )
    X.ww.init(logical_types={"col_1": "categorical", "col_2": "categorical"})
    expected_col_names = set(["col_3_b", "col_4"])
    for val in X["col_1"]:
        expected_col_names.add("col_1_" + val)
    for val in X["col_2"]:
        expected_col_names.add("col_2_" + val)

    encoder = OneHotEncoder(top_n=None, handle_unknown="error", random_seed=2)
    encoder.fit(X)
    X_t = encoder.transform(X)

    col_names = set(X_t.columns)
    assert X_t.shape == (11, 19)
    assert col_names == expected_col_names

    # Make sure unknown values cause an error
    X_new = pd.DataFrame(
        {
            "col_1": ["a", "b", "c", "x"],
            "col_2": ["a", "c", "d", "b"],
            "col_3": ["a", "a", "a", "a"],
            "col_4": [2, 0, 1, 3],
        }
    )

    with pytest.raises(ValueError) as exec_info:
        encoder.transform(X_new)
    assert "Found unknown categories" in exec_info.value.args[0]


def test_categories():
    X = pd.DataFrame(
        {
            "col_1": ["a", "b", "c", "d", "e", "f", "g"],
            "col_2": ["a", "c", "d", "b", "e", "e", "f"],
            "col_3": ["a", "a", "a", "a", "a", "a", "b"],
            "col_4": [2, 0, 1, 3, 0, 1, 2],
        }
    )
    X = set_first_three_columns_to_categorical(X)

    categories = [["a", "b", "c", "d"], ["a", "b", "c"], ["a", "b"]]

    # test categories value works
    encoder = OneHotEncoder(top_n=None, categories=categories, random_seed=2)
    encoder.fit(X)
    X_t = encoder.transform(X)

    col_names = set(X_t.columns)
    expected_col_names = set(
        [
            "col_1_a",
            "col_1_b",
            "col_1_c",
            "col_1_d",
            "col_2_a",
            "col_2_b",
            "col_2_c",
            "col_3_a",
            "col_3_b",
            "col_4",
        ]
    )
    assert X_t.shape == (7, 10)
    assert col_names == expected_col_names

    # test categories with top_n errors
    with pytest.raises(
        ValueError, match="Cannot use categories and top_n arguments simultaneously"
    ):
        encoder = OneHotEncoder(top_n=10, categories=categories, random_seed=2)


def test_less_than_top_n_unique_values():
    # test that columns with less than n unique values encodes properly
    X = pd.DataFrame(
        {
            "col_1": ["a", "b", "c", "d", "a"],
            "col_2": ["a", "b", "a", "c", "b"],
            "col_3": ["a", "a", "a", "a", "a"],
            "col_4": [2, 0, 1, 0, 0],
        }
    )
    X.ww.init(logical_types={"col_1": "categorical", "col_2": "categorical"})
    encoder = OneHotEncoder(top_n=5)
    encoder.fit(X)
    X_t = encoder.transform(X)
    expected_col_names = set(
        [
            "col_1_a",
            "col_1_b",
            "col_1_c",
            "col_1_d",
            "col_2_a",
            "col_2_b",
            "col_2_c",
            "col_3_a",
            "col_4",
        ]
    )
    col_names = set(X_t.columns)
    assert col_names == expected_col_names


def test_more_top_n_unique_values():
    # test that columns with >= n unique values encodes properly
    X = pd.DataFrame(
        {
            "col_1": ["a", "b", "c", "d", "e", "f", "g"],
            "col_2": ["a", "c", "d", "b", "e", "e", "f"],
            "col_3": ["a", "a", "a", "a", "a", "a", "b"],
            "col_4": [2, 0, 1, 3, 0, 1, 2],
        }
    )
    X = set_first_three_columns_to_categorical(X)

    random_seed = 2

    encoder = OneHotEncoder(top_n=5, random_seed=random_seed)
    encoder.fit(X)
    X_t = encoder.transform(X)

    # Conversion changes the resulting dataframe dtype, resulting in a different random state, so we need make the conversion here too
    X = infer_feature_types(X)
    col_1_counts = X["col_1"].value_counts(dropna=False).to_frame()
    col_1_counts = col_1_counts.sample(frac=1, random_state=random_seed)
    col_1_counts = col_1_counts.sort_values(
        ["col_1"], ascending=False, kind="mergesort"
    )
    col_1_samples = col_1_counts.head(encoder.parameters["top_n"]).index.tolist()

    col_2_counts = X["col_2"].value_counts(dropna=False).to_frame()
    col_2_counts = col_2_counts.sample(frac=1, random_state=random_seed)
    col_2_counts = col_2_counts.sort_values(
        ["col_2"], ascending=False, kind="mergesort"
    )
    col_2_samples = col_2_counts.head(encoder.parameters["top_n"]).index.tolist()

    expected_col_names = set(["col_2_e", "col_3_b", "col_4"])
    for val in col_1_samples:
        expected_col_names.add("col_1_" + val)
    for val in col_2_samples:
        expected_col_names.add("col_2_" + val)

    col_names = set(X_t.columns)
    assert col_names == expected_col_names


def test_more_top_n_unique_values_large():
    X = pd.DataFrame(
        {
            "col_1": ["a", "b", "c", "d", "e", "f", "g", "h", "i"],
            "col_2": ["a", "a", "a", "b", "b", "c", "c", "d", "e"],
            "col_3": ["a", "a", "a", "b", "b", "b", "c", "c", "d"],
            "col_4": [2, 0, 1, 3, 0, 1, 2, 4, 1],
        }
    )
    X = set_first_three_columns_to_categorical(X)
    random_seed = 2

    encoder = OneHotEncoder(top_n=3, random_seed=random_seed)
    encoder.fit(X)
    X_t = encoder.transform(X)

    # Conversion changes the resulting dataframe dtype, resulting in a different random state, so we need make the conversion here too
    X = infer_feature_types(X)
    col_1_counts = X["col_1"].value_counts(dropna=False).to_frame()
    col_1_counts = col_1_counts.sample(frac=1, random_state=random_seed)
    col_1_counts = col_1_counts.sort_values(
        ["col_1"], ascending=False, kind="mergesort"
    )
    col_1_samples = col_1_counts.head(encoder.parameters["top_n"]).index.tolist()
    expected_col_names = set(
        ["col_2_a", "col_2_b", "col_2_c", "col_3_a", "col_3_b", "col_3_c", "col_4"]
    )
    for val in col_1_samples:
        expected_col_names.add("col_1_" + val)

    col_names = set(X_t.columns)
    assert col_names == expected_col_names


def test_categorical_dtype():
    # test that columns with the categorical type are encoded properly
    X = pd.DataFrame(
        {
            "col_1": ["f", "b", "c", "d", "e"],
            "col_2": ["a", "e", "d", "d", "e"],
            "col_3": ["a", "a", "a", "a", "a"],
            "col_4": [3, 3, 2, 2, 1],
        }
    )
    X["col_4"] = X["col_4"].astype("category")
    X.ww.init(logical_types={"col_1": "categorical", "col_2": "categorical"})

    encoder = OneHotEncoder(top_n=5)
    encoder.fit(X)
    X_t = encoder.transform(X)

    expected_col_names = set(
        [
            "col_1_f",
            "col_1_b",
            "col_1_c",
            "col_1_d",
            "col_1_e",
            "col_2_d",
            "col_2_e",
            "col_2_a",
            "col_3_a",
            "col_4_1",
            "col_4_2",
            "col_4_3",
        ]
    )
    col_names = set(X_t.columns)
    assert col_names == expected_col_names
    assert [X_t[col].dtype == "uint8" for col in X_t]


def test_all_numerical_dtype():
    # test that columns with the numerical type are preserved
    X = pd.DataFrame(
        {
            "col_1": [2, 0, 1, 0, 0],
            "col_2": [3, 2, 5, 1, 3],
            "col_3": [0, 0, 1, 3, 2],
            "col_4": [2, 4, 1, 4, 0],
        }
    )
    X_expected = X.copy()
    encoder = OneHotEncoder(top_n=5)
    encoder.fit(X)
    X_t = encoder.transform(X)
    assert_frame_equal(X_expected, X_t)


def test_numpy_input():
    X = np.array([[2, 0, 1, 0, 0], [3, 2, 5, 1, 3]])
    encoder = OneHotEncoder()
    encoder.fit(X)
    X_t = encoder.transform(X)
    assert_frame_equal(pd.DataFrame(X), X_t, check_dtype=False)


def test_large_number_of_categories():
    n_categories = 200000
    frequency_per_category = 5
    X = np.repeat(np.arange(n_categories), frequency_per_category).reshape((-1, 1))
    X_extra = np.repeat(np.arange(10) + n_categories, 10).reshape((-1, 1))
    X = np.array(np.concatenate([X, X_extra]))
    X = pd.DataFrame(X, columns=["cat"])
    X["cat"] = X["cat"].astype("category")
    encoder = OneHotEncoder(top_n=10)
    encoder.fit(X)
    X_t = encoder.transform(X)
    expected_col_names = ["cat_" + str(200000 + i) for i in range(10)]
    assert X_t.shape == (1000100, 10)
    assert set(expected_col_names) == set(list(X_t.columns))


@pytest.mark.parametrize("data_type", ["list", "np", "pd_no_index", "pd_index", "ww"])
def test_data_types(data_type):
    if data_type == "list":
        X = [["a"], ["b"], ["c"]] * 5
    elif data_type == "np":
        X = np.array([["a"], ["b"], ["c"]] * 5)
    elif data_type == "pd_no_index":
        X = pd.DataFrame(["a", "b", "c"] * 5)
    elif data_type == "pd_index":
        X = pd.DataFrame(["a", "b", "c"] * 5, columns=["0"])
    elif data_type == "ww":
        X = pd.DataFrame(["a", "b", "c"] * 5)
        X.ww.init()
    encoder = OneHotEncoder()
    encoder.fit(X)
    X_t = encoder.transform(X)
    assert list(X_t.columns) == ["0_a", "0_b", "0_c"]
    mask = np.identity(3)
    for _ in range(4):
        mask = np.vstack((mask, np.identity(3)))
    np.testing.assert_array_equal(X_t.to_numpy(), mask)


@pytest.mark.parametrize(
    "index",
    [
        list(range(-5, 0)),
        list(range(100, 105)),
        [f"row_{i}" for i in range(5)],
        pd.date_range("2020-09-08", periods=5),
    ],
)
def test_ohe_preserves_custom_index(index):

    df = pd.DataFrame(
        {"categories": [f"cat_{i}" for i in range(5)], "numbers": np.arange(5)},
        index=index,
    )
    ohe = OneHotEncoder()
    new_df = ohe.fit_transform(df)
    pd.testing.assert_index_equal(new_df.index, df.index)
    assert not new_df.isna().any(axis=None)


def test_ohe_categories():
    X = pd.DataFrame(
        {"col_1": ["a"] * 10, "col_2": ["a"] * 3 + ["b"] * 3 + ["c"] * 2 + ["d"] * 2}
    )
    X.ww.init(logical_types={"col_2": "categorical"})
    ohe = OneHotEncoder(top_n=2)
    with pytest.raises(
        ComponentNotYetFittedError,
        match="This OneHotEncoder is not fitted yet. You must fit OneHotEncoder before calling categories.",
    ):
        ohe.categories("col_1")

    ohe.fit(X)
    np.testing.assert_array_equal(ohe.categories("col_1"), np.array(["a"]))
    np.testing.assert_array_equal(ohe.categories("col_2"), np.array(["a", "b"]))
    with pytest.raises(
        ValueError,
        match='Feature "col_12345" was not provided to one-hot encoder as a training feature',
    ):
        ohe.categories("col_12345")


def test_ohe_get_feature_names():
    X = pd.DataFrame(
        {"col_1": ["a"] * 10, "col_2": ["a"] * 3 + ["b"] * 3 + ["c"] * 2 + ["d"] * 2}
    )
    X.ww.init(logical_types={"col_2": "categorical"})
    ohe = OneHotEncoder(top_n=2)
    with pytest.raises(
        ComponentNotYetFittedError,
        match="This OneHotEncoder is not fitted yet. You must fit OneHotEncoder before calling get_feature_names.",
    ):
        ohe.get_feature_names()
    ohe.fit(X)
    np.testing.assert_array_equal(
        ohe.get_feature_names(), np.array(["col_1_a", "col_2_a", "col_2_b"])
    )

    X = pd.DataFrame({"col_1": ["a"] * 4 + ["b"] * 6, "col_2": ["b"] * 3 + ["c"] * 7})
    ohe = OneHotEncoder(drop="if_binary")
    ohe.fit(X)
    np.testing.assert_array_equal(
        ohe.get_feature_names(), np.array(["col_1_a", "col_2_b"])
    )


def test_ohe_features_to_encode():
    # Test feature that doesn't need encoding and
    # feature that needs encoding but is not specified remain untouched
    X = pd.DataFrame({"col_1": [2, 0, 1, 0, 0], "col_2": ["a", "b", "a", "c", "d"]})

    encoder = OneHotEncoder(top_n=5, features_to_encode=["col_1"])
    encoder.fit(X)
    X_t = encoder.transform(X)
    expected_col_names = set(["col_1_0", "col_1_1", "col_1_2", "col_2"])
    col_names = set(X_t.columns)
    assert col_names == expected_col_names
    assert [X_t[col].dtype == "uint8" for col in X_t]

    encoder = OneHotEncoder(top_n=5, features_to_encode=["col_1", "col_2"])
    encoder.fit(X)
    X_t = encoder.transform(X)
    expected_col_names = set(
        ["col_1_0", "col_1_1", "col_1_2", "col_2_a", "col_2_b", "col_2_c", "col_2_d"]
    )
    col_names = set(X_t.columns)
    assert col_names == expected_col_names
    assert [X_t[col].dtype == "uint8" for col in X_t]


def test_ohe_features_to_encode_col_missing():
    X = pd.DataFrame({"col_1": [2, 0, 1, 0, 0], "col_2": ["a", "b", "a", "c", "d"]})

    encoder = OneHotEncoder(top_n=5, features_to_encode=["col_3", "col_4"])

    with pytest.raises(ValueError, match="Could not find and encode"):
        encoder.fit(X)


def test_ohe_features_to_encode_no_col_names():
    X = pd.DataFrame([["b", 0], ["a", 1], ["b", 1]])
    encoder = OneHotEncoder(top_n=5, features_to_encode=[0])
    encoder.fit(X)
    X_t = encoder.transform(X)
    expected_col_names = set([1, "0_a"])
    col_names = set(X_t.columns)
    assert col_names == expected_col_names
    assert [X_t[col].dtype == "uint8" for col in X_t]


def test_ohe_top_n_categories_always_the_same():
    df = pd.DataFrame(
        {
            "categories": ["cat_1"] * 5
            + ["cat_2"] * 4
            + ["cat_3"] * 3
            + ["cat_4"] * 3
            + ["cat_5"] * 3,
            "numbers": range(18),
        }
    )

    def check_df_equality(random_seed):
        ohe = OneHotEncoder(top_n=4, random_seed=random_seed)
        df1 = ohe.fit_transform(df)
        df2 = ohe.fit_transform(df)
        assert_frame_equal(df1, df2)

    check_df_equality(5)
    check_df_equality(get_random_seed(5))


def test_ohe_column_names_unique():
    df = pd.DataFrame({"A": ["x_y"], "A_x": ["y"]})
    df.ww.init(logical_types={"A": "categorical", "A_x": "categorical"})
    df_transformed = OneHotEncoder().fit_transform(df)
    assert set(df_transformed.columns) == {"A_x_y", "A_x_y_1"}

    df = pd.DataFrame(
        {
            "A": ["x_y", "z", "z"],
            "A_x": [
                "y",
                "a",
                "a",
            ],
            "A_x_y": ["1", "y", "y"],
        }
    )
    df.ww.init(
        logical_types={"A": "categorical", "A_x": "categorical", "A_x_y": "categorical"}
    )
    df_transformed = OneHotEncoder().fit_transform(df)
    # category y in A_x gets mapped to A_x_y_1 because A_x_y already exists
    # category 1 in A_x_y gets mapped to A_x_y_1_1 because A_x_y_1 already exists
    assert set(df_transformed.columns) == {"A_x_y", "A_x_y_1", "A_x_y_1_1"}

    df = pd.DataFrame(
        {"A": ["x_y", "z", "a"], "A_x": ["y_1", "y", "b"], "A_x_y": ["1", "y", "c"]}
    )
    df.ww.init(
        logical_types={"A": "categorical", "A_x": "categorical", "A_x_y": "categorical"}
    )
    df_transformed = OneHotEncoder().fit_transform(df)
    # category y in A_x gets mapped to A_x_y_1 because A_x_y already exists
    # category y_1 in A_x gets mapped to A_x_y_1_1 because A_x_y_1 already exists
    # category 1 in A_x_y gets mapped to A_x_y_1_2 because A_x_y_1_1 already exists
    assert set(df_transformed.columns) == {
        "A_x_y",
        "A_z",
        "A_a",
        "A_x_y_1",
        "A_x_y_1_1",
        "A_x_b",
        "A_x_y_1_2",
        "A_x_y_y",
        "A_x_y_c",
    }


@pytest.mark.parametrize(
    "X_df",
    [
        pd.DataFrame(
            pd.to_datetime(["20190902", "20200519", "20190607"], format="%Y%m%d")
        ),
        pd.DataFrame(pd.Series([1, 2, 3], dtype="Int64")),
        pd.DataFrame(pd.Series([1.0, 2.0, 3.0], dtype="float")),
        pd.DataFrame(pd.Series(["a", "b", "a"], dtype="category")),
        pd.DataFrame(pd.Series([True, False, True], dtype="boolean")),
        pd.DataFrame(
            pd.Series(
                ["this will be a natural language column because length", "yay", "hay"],
                dtype="string",
            )
        ),
    ],
)
def test_ohe_woodwork_custom_overrides_returned_by_components(X_df):
    y = pd.Series([1, 2, 1])
    override_types = [Integer, Double, Categorical, NaturalLanguage, Datetime, Boolean]
    for logical_type in override_types:
        try:
            X = X_df.copy()
            X.ww.init(logical_types={0: logical_type})
        except (TypeConversionError, ValueError, TypeError):
            continue

        ohe = OneHotEncoder()
        ohe.fit(X, y)
        transformed = ohe.transform(X, y)
        assert isinstance(transformed, pd.DataFrame)
        if logical_type != Categorical:
            assert {k: type(v) for k, v in transformed.ww.logical_types.items()} == {
                0: logical_type
            }


def test_ohe_output_bools():
    X = pd.DataFrame(
        {
            "bool": [bool(i % 2) for i in range(100)],
            "categorical": ["dog"] * 20 + ["cat"] * 40 + ["fish"] * 40,
            "integers": [i for i in range(100)],
        }
    )
    X.ww.init()
    y = pd.Series([i % 2 for i in range(100)])
    y.ww.init()
    ohe = OneHotEncoder()
    output = ohe.fit_transform(X, y)
    for name, types in output.ww.types["Logical Type"].items():
        if name == "integers":
            assert str(types) == "Integer"
        else:
            assert str(types) == "Boolean"
    assert len(output.columns) == 5
