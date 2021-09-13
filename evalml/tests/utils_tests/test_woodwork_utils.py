from itertools import product

import numpy as np
import pandas as pd
import pytest
import woodwork as ww
from woodwork.logical_types import Double, Unknown

from evalml.utils import _convert_numeric_dataset_pandas, infer_feature_types


def test_infer_feature_types_no_type_change():
    X_dt = pd.DataFrame([[1, 2], [3, 4]])
    X_dt.ww.init()
    pd.testing.assert_frame_equal(X_dt, infer_feature_types(X_dt))

    X_dc = ww.init_series(pd.Series([1, 2, 3, 4]))
    pd.testing.assert_series_equal(X_dc, infer_feature_types(X_dc))

    X_pd = pd.DataFrame(
        {0: pd.Series([1, 2], dtype="int64"), 1: pd.Series([3, 4], dtype="int64")}
    )
    pd.testing.assert_frame_equal(X_pd, infer_feature_types(X_pd))

    X_list = [1, 2, 3, 4]
    X_expected = ww.init_series(pd.Series(X_list))
    pd.testing.assert_series_equal(X_expected, infer_feature_types(X_list))
    assert X_list == [1, 2, 3, 4]

    X_np = np.array([1, 2, 3, 4])
    X_expected = ww.init_series(pd.Series(X_np))
    pd.testing.assert_series_equal(X_expected, infer_feature_types(X_np))
    assert np.array_equal(X_np, np.array([1, 2, 3, 4]))

    X_np = np.array([[1, 2], [3, 4]])
    X_expected = pd.DataFrame(X_np)
    X_expected.ww.init()
    pd.testing.assert_frame_equal(X_expected, infer_feature_types(X_np))
    assert np.array_equal(X_np, np.array([[1, 2], [3, 4]]))


def test_infer_feature_types_series_name():
    name = "column with name"
    X_pd = pd.Series([1, 2, 3, 4], dtype="int64", name=name)
    X_dc = infer_feature_types(X_pd)
    assert X_dc.name == name
    pd.testing.assert_series_equal(X_pd, X_dc)


def test_infer_feature_types_dataframe():
    X_pd = pd.DataFrame({0: pd.Series([1, 2]), 1: pd.Series([3, 4])})
    pd.testing.assert_frame_equal(X_pd, infer_feature_types(X_pd), check_dtype=False)

    X_pd = pd.DataFrame(
        {0: pd.Series([1, 2], dtype="int64"), 1: pd.Series([3, 4], dtype="int64")}
    )
    pd.testing.assert_frame_equal(X_pd, infer_feature_types(X_pd))

    X_expected = X_pd.copy()
    X_expected[0] = X_expected[0].astype("category")
    pd.testing.assert_frame_equal(
        X_expected, infer_feature_types(X_pd, {0: "categorical"})
    )
    pd.testing.assert_frame_equal(
        X_expected, infer_feature_types(X_pd, {0: ww.logical_types.Categorical})
    )


def test_infer_feature_types_series():
    X_pd = pd.Series([1, 2, 3, 4])
    X_expected = X_pd.astype("int64")
    pd.testing.assert_series_equal(X_expected, infer_feature_types(X_pd))

    X_pd = pd.Series([1, 2, 3, 4], dtype="int64")
    pd.testing.assert_series_equal(X_pd, infer_feature_types(X_pd))

    X_pd = pd.Series([1, 2, 3, 4], dtype="int64")
    X_expected = X_pd.astype("category")
    pd.testing.assert_series_equal(X_expected, infer_feature_types(X_pd, "categorical"))

    X_pd = pd.Series([1, 2, 3, 4], dtype="int64")
    X_expected = X_pd.astype("category")
    pd.testing.assert_series_equal(
        X_expected, infer_feature_types(X_pd, ww.logical_types.Categorical)
    )


@pytest.mark.parametrize(
    "value,error",
    [
        (1, False),
        (-1, False),
        (2.3, False),
        (None, True),
        (np.nan, True),
        ("hello", True),
    ],
)
@pytest.mark.parametrize("datatype", ["np", "pd", "ww"])
def test_convert_numeric_dataset_pandas(datatype, value, error, make_data_type):
    if datatype == "np" and value == "hello":
        pytest.skip("Unsupported configuration")

    X = pd.DataFrame([[1, 2, 3, 4], [2, value, 4, value]])
    y = pd.Series([0, 1])
    X = make_data_type(datatype, X)
    y = make_data_type(datatype, y)

    if error:
        with pytest.raises(
            ValueError, match="Values not all numeric or there are null"
        ):
            _convert_numeric_dataset_pandas(X, y)
    else:
        X_transformed, y_transformed = _convert_numeric_dataset_pandas(X, y)
        X_ww = infer_feature_types(X)
        y_ww = infer_feature_types(y)

        pd.testing.assert_frame_equal(X_ww, X_transformed)
        pd.testing.assert_series_equal(y_ww, y_transformed)


def test_infer_feature_types_value_error():

    df = pd.DataFrame(
        {
            "a": pd.Series([1, 2, 3]),
            "b": pd.Series([4, 5, 6]),
            "c": pd.Series([True, False, True]),
        }
    )
    df.ww.init(logical_types={"a": "IntegerNullable", "c": "BooleanNullable"})
    msg = "These are the columns with nullable types: \\[\\('a', 'Int64'\\), \\('c', 'boolean'\\)\\]"
    with pytest.raises(ValueError, match=msg):
        infer_feature_types(df)

    y = pd.Series([1, 2, 3], name="series")
    y = ww.init_series(y, logical_type="IntegerNullable")

    with pytest.raises(
        ValueError,
        match="These are the columns with nullable types: \\[\\('series', 'Int64'\\)]",
    ):
        infer_feature_types(y)

    df = pd.DataFrame({"A": pd.Series([4, 5, 6], dtype="Float64"), "b": [1, 2, 3]})
    with pytest.raises(
        ValueError,
        match="These are the columns with nullable types: \\[\\('A', 'Float64'\\)]",
    ):
        infer_feature_types(df)


def test_infer_feature_types_preserves_semantic_tags():
    df = pd.DataFrame(
        {
            "a": pd.Series([1, 2, 3]),
            "b": pd.Series([4, 5, 6]),
            "c": pd.Series([True, False, True]),
            "my_index": [1, 2, 3],
            "time_index": ["2020-01-01", "2020-01-02", "2020-01-03"],
        }
    )
    df.ww.init(
        logical_types={"a": "Integer", "c": "Categorical", "b": "Double"},
        semantic_tags={"a": "My Integer", "c": "My Categorical", "b": "My Double"},
        index="my_index",
        time_index="time_index",
    )
    new_df = infer_feature_types(df)
    assert new_df.ww.schema == df.ww.schema

    series = pd.Series([1, 2, 3], name="target")
    series.ww.init(
        logical_type="Integer", semantic_tags=["Cool Series"], description="Great data"
    )
    assert series.ww.schema == infer_feature_types(series).ww.schema


def test_infer_feature_types_raises_invalid_schema_error():

    df = pd.DataFrame(pd.Series([1, 2, None]))

    # Raise error when user requests incompatible type
    with pytest.raises(ww.exceptions.TypeConversionError):
        infer_feature_types(df, feature_types={0: "Integer"})

    # Raise error when user breaks the schema and then passes it to evalml
    with pytest.raises(
        ValueError,
        match=(
            "Dataframe types are not consistent with logical types. This usually happens "
            "when a data transformation does not go through the ww accessor."
        ),
    ):
        df.iloc[2, 0] = 3
        df.ww.init(logical_types={0: "Integer"})
        df.iloc[2, 0] = None
        infer_feature_types(df, feature_types={0: "Integer"})

    with pytest.raises(
        ValueError,
        match="Please initialize ww with df.ww.init()",
    ):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df.ww.init()
        df.drop(columns=["b"], inplace=True)
        infer_feature_types(df)


@pytest.mark.parametrize(
    "null_col,already_inited",
    product(
        [
            [None, None, None],
            [np.nan, np.nan, np.nan],
            [pd.NA, pd.NA, pd.NA],
            ["ax23n9ck23l", "1,28&*_%*&&xejc", "xnmvz@@Dcmeods-0"],
        ],
        [True, False],
    ),
)
def test_infer_feature_types_NA_to_nan(null_col, already_inited):
    """A short test to make sure that columns with all null values
    get converted from woodwork Unknown logical type with string
    physical type back to the original Double logical type with
    float physical type.  Other Unknown columns should remain unchanged."""

    df = pd.DataFrame(
        {
            "unknown": null_col,
        },
    )

    # Check that all null columns are inferred as Unknown type.
    df.ww.init()
    assert isinstance(df.ww.logical_types["unknown"], Unknown)
    if all(df["unknown"].isnull()):
        assert all([isinstance(x, type(pd.NA)) for x in df["unknown"]])
    else:
        assert all([isinstance(x, str) for x in df["unknown"]])

    # Use infer_feature_types() to init the WW accessor and verify that all
    # null columns are now Double logical types backed by np.nan and that other
    # columns inferred as Unknown remain untouched.
    del df.ww
    if already_inited:
        df.ww.init()
    inferred_df = infer_feature_types(df)
    if all(df["unknown"].isnull()):
        assert isinstance(inferred_df.ww.logical_types["unknown"], Double)
        assert all([isinstance(x, type(np.nan)) for x in inferred_df["unknown"]])
    else:
        assert all([isinstance(x, str) for x in df["unknown"]])
