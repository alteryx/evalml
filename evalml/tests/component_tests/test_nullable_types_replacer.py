import pandas as pd
import pytest
from woodwork import init_series
from woodwork.logical_types import (
    Age,
    AgeNullable,
    Boolean,
    BooleanNullable,
    Categorical,
    Double,
    Integer,
    IntegerNullable,
)

from evalml.pipelines.components import ReplaceNullableTypes


@pytest.fixture
def nullable_data():
    return pd.DataFrame(
        {
            "non_nullable_integer": [0, 1, 2, 3, 4],
            "nullable_integer_with_null": [0, 1, 2, 3, None],
            "nullable_integer_without_null": [0, 1, 2, 3, 4],
            "non_nullable_age": [20, 21, 22, 23, 24],
            "nullable_age_with_null": [20, None, 22, 23, None],
            "nullable_age_without_null": [20, 21, 22, 23, 24],
            "non_nullable_boolean": [True, False, True, False, True],
            "nullable_boolean_with_null": [True, False, True, False, None],
            "nullable_boolean_without_null": [True, True, False, True, False],
        },
    )


@pytest.mark.parametrize("methods_to_test", ["fit and transform", "fit_transform"])
@pytest.mark.parametrize("input_type", ["ww", "pandas"])
def test_replace_nullable_types(nullable_data, input_type, methods_to_test):
    X = nullable_data

    nullable_types_replacer = ReplaceNullableTypes()

    X = X.astype(
        {
            "nullable_integer_with_null": "Int64",
            "nullable_integer_without_null": "Int64",
            "nullable_age_with_null": "Int64",
            "nullable_age_without_null": "Int64",
            "nullable_boolean_with_null": "boolean",
            "nullable_boolean_without_null": "boolean",
        },
    )

    assert str(X.dtypes.loc["non_nullable_integer"]) == "int64"
    assert str(X.dtypes.loc["nullable_integer_with_null"]) == "Int64"
    assert str(X.dtypes.loc["nullable_integer_without_null"]) == "Int64"
    assert str(X.dtypes.loc["non_nullable_age"]) == "int64"
    assert str(X.dtypes.loc["nullable_age_with_null"]) == "Int64"
    assert str(X.dtypes.loc["nullable_age_without_null"]) == "Int64"
    assert str(X.dtypes.loc["non_nullable_boolean"]) == "bool"
    assert str(X.dtypes.loc["nullable_boolean_with_null"]) == "boolean"
    assert str(X.dtypes.loc["nullable_boolean_without_null"]) == "boolean"

    if input_type == "ww":
        X.ww.init(
            logical_types={
                "non_nullable_age": Age,
                "nullable_age_with_null": AgeNullable,
                "nullable_age_without_null": AgeNullable,
                "nullable_integer_without_null": IntegerNullable,
                "nullable_boolean_without_null": BooleanNullable,
            },
        )

        assert isinstance(
            X.ww.logical_types["nullable_integer_with_null"],
            IntegerNullable,
        )
        assert isinstance(
            X.ww.logical_types["nullable_integer_without_null"],
            IntegerNullable,
        )
        assert isinstance(X.ww.logical_types["nullable_age_with_null"], AgeNullable)
        assert isinstance(X.ww.logical_types["nullable_age_without_null"], AgeNullable)
        assert isinstance(
            X.ww.logical_types["nullable_boolean_with_null"],
            BooleanNullable,
        )
        assert isinstance(
            X.ww.logical_types["nullable_boolean_without_null"],
            BooleanNullable,
        )

    if methods_to_test == "fit and transform":
        nullable_types_replacer.fit(X)
        if input_type == "ww":
            assert set(nullable_types_replacer._nullable_int_cols) == {
                "nullable_integer_with_null",
                "nullable_integer_without_null",
                "nullable_age_with_null",
                "nullable_age_without_null",
            }
        else:
            assert set(nullable_types_replacer._nullable_int_cols) == {
                "nullable_integer_with_null",
                "nullable_age_with_null",
            }
        if input_type == "ww":
            assert nullable_types_replacer._nullable_bool_cols == [
                "nullable_boolean_with_null",
                "nullable_boolean_without_null",
            ]
        else:
            assert nullable_types_replacer._nullable_bool_cols == [
                "nullable_boolean_with_null",
            ]

        X_t, y_t = nullable_types_replacer.transform(X)
        assert set(X_t.columns) == set(X.columns)
        assert X_t.shape == X.shape
    elif methods_to_test == "fit_transform":
        X_t, y_t = nullable_types_replacer.fit_transform(X)
        if input_type == "ww":
            assert set(nullable_types_replacer._nullable_int_cols) == {
                "nullable_integer_with_null",
                "nullable_integer_without_null",
                "nullable_age_with_null",
                "nullable_age_without_null",
            }
        else:
            assert set(nullable_types_replacer._nullable_int_cols) == {
                "nullable_integer_with_null",
                "nullable_age_with_null",
            }
            assert nullable_types_replacer._nullable_bool_cols == [
                "nullable_boolean_with_null",
            ]
        assert set(X_t.columns) == set(X.columns)
        assert X_t.shape == X.shape

    # Check the pandas dtypes
    assert str(X_t.dtypes.loc["non_nullable_integer"]) == "int64"
    assert str(X_t.dtypes.loc["nullable_integer_with_null"]) == "Int64"
    assert (
        str(X_t.dtypes.loc["nullable_integer_without_null"]) == "float64"
        if input_type == "ww"
        else "Int64"
    )
    assert str(X_t.dtypes.loc["non_nullable_age"]) == "int64"
    assert str(X_t.dtypes.loc["nullable_age_with_null"]) == "Int64"
    assert (
        str(X_t.dtypes.loc["nullable_age_without_null"]) == "float64"
        if input_type == "ww"
        else "Int64"
    )
    assert str(X_t.dtypes.loc["non_nullable_boolean"]) == "bool"
    assert str(X_t.dtypes.loc["nullable_boolean_with_null"]) == "boolean"
    assert (
        str(X_t.dtypes.loc["nullable_boolean_without_null"]) == "bool"
        if input_type == "ww"
        else "boolean"
    )

    # Check the Woodwork dtypes
    assert isinstance(X_t.ww.logical_types["non_nullable_integer"], Integer)
    assert isinstance(
        X_t.ww.logical_types["nullable_integer_with_null"],
        IntegerNullable,
    )
    if input_type == "ww":
        assert isinstance(X_t.ww.logical_types["nullable_integer_without_null"], Double)
    else:
        assert isinstance(
            X_t.ww.logical_types["nullable_integer_without_null"],
            Integer,
        )

    if input_type == "ww":
        assert isinstance(X_t.ww.logical_types["non_nullable_age"], Age)
        assert isinstance(X_t.ww.logical_types["nullable_age_with_null"], AgeNullable)
        assert isinstance(X_t.ww.logical_types["nullable_age_without_null"], Double)
    else:
        assert isinstance(X_t.ww.logical_types["non_nullable_age"], Integer)
        assert isinstance(
            X_t.ww.logical_types["nullable_age_with_null"],
            IntegerNullable,
        )
        assert isinstance(X_t.ww.logical_types["nullable_age_without_null"], Integer)

    assert isinstance(X_t.ww.logical_types["non_nullable_boolean"], Boolean)
    assert isinstance(
        X_t.ww.logical_types["nullable_boolean_with_null"],
        BooleanNullable,
    )
    assert isinstance(X_t.ww.logical_types["nullable_boolean_without_null"], Boolean)


@pytest.mark.parametrize("input_type", ["ww", "pandas"])
@pytest.mark.parametrize("with_null", [True, False])
def test_replace_nullable_types_boolean_target(nullable_data, input_type, with_null):
    nullable_types_replacer = ReplaceNullableTypes()

    # Get input data
    X = nullable_data
    X = X.astype(
        {
            "nullable_integer_with_null": "Int64",
            "nullable_boolean_with_null": "boolean",
        },
    )

    if with_null:
        y = pd.Series([True, False, None, True, False])
    else:
        y = pd.Series([True, False, True, True, False])

    y = y.astype("boolean")
    if input_type == "ww":
        y = init_series(y, logical_type=BooleanNullable)
        assert isinstance(y.ww.logical_type, BooleanNullable)

    nullable_types_replacer.fit(X, y)

    if with_null or input_type == "ww":
        assert nullable_types_replacer._nullable_target == "nullable_bool"
    else:
        assert nullable_types_replacer._nullable_target is None

    X_t, y_t = nullable_types_replacer.transform(X, y)

    if with_null:
        assert str(y_t.dtypes) == "category"
        assert isinstance(y_t.ww.logical_type, Categorical)
    else:
        assert str(y_t.dtypes) == "bool"
        assert isinstance(y_t.ww.logical_type, Boolean)


@pytest.mark.parametrize("input_type", ["ww", "pandas"])
@pytest.mark.parametrize("with_null", [True, False])
def test_replace_nullable_types_integer_target(nullable_data, input_type, with_null):
    nullable_types_replacer = ReplaceNullableTypes()

    # Get input data
    X = nullable_data
    X = X.astype(
        {
            "nullable_integer_with_null": "Int64",
            "nullable_boolean_with_null": "boolean",
        },
    )

    if with_null:
        y = pd.Series([0, 1, None, 3, 4])
    else:
        y = pd.Series([0, 1, 2, 3, 4])

    y = y.astype("Int64")

    if input_type == "ww":
        y = init_series(y, logical_type=IntegerNullable)

    nullable_types_replacer.fit(X, y)

    if with_null or input_type == "ww":
        assert nullable_types_replacer._nullable_target == "nullable_int"
    else:
        assert nullable_types_replacer._nullable_target is None

    X_t, y_t = nullable_types_replacer.transform(X, y)

    if with_null:
        assert str(y_t.dtypes) == "float64"
        assert isinstance(y_t.ww.logical_type, Double)
    else:
        if input_type == "pandas":
            assert str(y_t.dtypes) == "int64"
            assert isinstance(y_t.ww.logical_type, Integer)
        else:
            assert str(y_t.dtypes) == "float64"
            assert isinstance(y_t.ww.logical_type, Double)
