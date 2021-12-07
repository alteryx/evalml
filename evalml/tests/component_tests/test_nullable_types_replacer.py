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
            "nullable_integer": [0, 1, 2, 3, None],
            "non_nullable_age": [20, 21, 22, 23, 24],
            "nullable_age": [20, None, 22, 23, None],
            "non_nullable_boolean": [True, False, True, False, True],
            "nullable_boolean": [None, True, False, True, False],
        }
    )


@pytest.mark.parametrize("methods_to_test", ["fit and transform", "fit_transform"])
@pytest.mark.parametrize("input_type", ["ww", "pandas"])
def test_replace_nullable_types(nullable_data, input_type, methods_to_test):
    X = nullable_data

    nullable_types_replacer = ReplaceNullableTypes()

    X = X.astype(
        {
            "nullable_integer": "Int64",
            "nullable_age": "Int64",
            "nullable_boolean": "boolean",
        }
    )

    assert str(X.dtypes.loc["non_nullable_integer"]) == "int64"
    assert str(X.dtypes.loc["nullable_integer"]) == "Int64"
    assert str(X.dtypes.loc["non_nullable_age"]) == "int64"
    assert str(X.dtypes.loc["nullable_age"]) == "Int64"
    assert str(X.dtypes.loc["non_nullable_boolean"]) == "bool"
    assert str(X.dtypes.loc["nullable_boolean"]) == "boolean"

    if input_type == "ww":
        X.ww.init(logical_types={"nullable_age": AgeNullable, "non_nullable_age": Age})
        assert isinstance(X.ww.logical_types["nullable_integer"], IntegerNullable)
        assert isinstance(X.ww.logical_types["nullable_age"], AgeNullable)
        assert isinstance(X.ww.logical_types["nullable_boolean"], BooleanNullable)

    if methods_to_test == "fit and transform":
        nullable_types_replacer.fit(X)

        assert set(nullable_types_replacer._nullable_int_cols) == {
            "nullable_integer",
            "nullable_age",
        }
        assert nullable_types_replacer._nullable_bool_cols == ["nullable_boolean"]

        X_t, y_t = nullable_types_replacer.transform(X)
        assert set(X_t.columns) == set(X.columns)
        assert X_t.shape == X.shape
    elif methods_to_test == "fit_transform":
        X_t, y_t = nullable_types_replacer.fit_transform(X)

        assert set(nullable_types_replacer._nullable_int_cols) == {
            "nullable_integer",
            "nullable_age",
        }
        assert nullable_types_replacer._nullable_bool_cols == ["nullable_boolean"]
        assert set(X_t.columns) == set(X.columns)
        assert X_t.shape == X.shape

    # Check the pandas dtypes
    assert str(X_t.dtypes.loc["non_nullable_integer"]) == "int64"
    assert str(X_t.dtypes.loc["nullable_integer"]) == "float64"
    assert str(X_t.dtypes.loc["non_nullable_age"]) == "int64"
    assert str(X_t.dtypes.loc["nullable_age"]) == "float64"
    assert str(X_t.dtypes.loc["non_nullable_boolean"]) == "bool"
    assert str(X_t.dtypes.loc["nullable_boolean"]) == "category"

    # Check the Woodwork dtypes
    assert isinstance(X_t.ww.logical_types["non_nullable_integer"], Integer)
    assert isinstance(X_t.ww.logical_types["nullable_integer"], Double)
    if input_type == "ww":
        assert isinstance(X_t.ww.logical_types["non_nullable_age"], Age)
    else:
        assert isinstance(X_t.ww.logical_types["non_nullable_age"], Integer)
    assert isinstance(X_t.ww.logical_types["nullable_age"], Double)
    assert isinstance(X_t.ww.logical_types["non_nullable_boolean"], Boolean)
    assert isinstance(X_t.ww.logical_types["nullable_boolean"], Categorical)


@pytest.mark.parametrize("input_type", ["ww", "pandas"])
def test_replace_nullable_types_boolean_target(nullable_data, input_type):
    nullable_types_replacer = ReplaceNullableTypes()

    # Get input data
    X = nullable_data
    X = X.astype({"nullable_integer": "Int64", "nullable_boolean": "boolean"})

    y = pd.Series([True, False, None, True, False])
    y = y.astype("boolean")

    if input_type == "ww":
        y = init_series(y)
        assert isinstance(y.ww.logical_type, BooleanNullable)

    nullable_types_replacer.fit(X, y)

    assert nullable_types_replacer._nullable_target == "nullable_bool"

    X_t, y_t = nullable_types_replacer.transform(X, y)

    assert str(y_t.dtypes) == "category"
    assert isinstance(y_t.ww.logical_type, Categorical)


@pytest.mark.parametrize("input_type", ["ww", "pandas"])
def test_replace_nullable_types_integer_target(nullable_data, input_type):
    nullable_types_replacer = ReplaceNullableTypes()

    # Get input data
    X = nullable_data
    X = X.astype({"nullable_integer": "Int64", "nullable_boolean": "boolean"})

    y = pd.Series([0, 1, None, 3, 4])
    y = y.astype("Int64")

    if input_type == "ww":
        y = init_series(y)
        assert isinstance(y.ww.logical_type, IntegerNullable)

    nullable_types_replacer.fit(X, y)

    assert nullable_types_replacer._nullable_target == "nullable_int"

    X_t, y_t = nullable_types_replacer.transform(X, y)

    assert str(y_t.dtypes) == "float64"
    assert isinstance(y_t.ww.logical_type, Double)
