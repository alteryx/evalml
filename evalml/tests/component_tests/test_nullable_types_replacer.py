import pandas as pd
import pytest
from woodwork.logical_types import (
    BooleanNullable,
    Double,
    IntegerNullable,
    Unknown,
)

from evalml.pipelines.components import ReplaceNullableTypes


@pytest.mark.parametrize("nullable_types_properly_set", [True, False])
@pytest.mark.parametrize("input_type", ["ww", "pandas"])
def test_replace_nullable_types(nullable_types_properly_set, input_type):
    nullable_types_replacer = ReplaceNullableTypes()
    X = pd.DataFrame(
        {
            "non_nullable_integer": [0, 1, 2, 3, 4],
            "nullable_integer": [0, 1, 2, 3, None],
            "non_nullable_boolean": [True, False, True, False, True],
            "nullable_boolean": [None, True, False, True, False],
        }
    )
    # Check that the underlying pandas data types are set properly
    if nullable_types_properly_set:
        X = X.astype({"nullable_integer": "Int64", "nullable_boolean": "boolean"})

        assert str(X.dtypes.loc["non_nullable_integer"]) == "int64"
        assert str(X.dtypes.loc["nullable_integer"]) == "Int64"
        assert str(X.dtypes.loc["non_nullable_boolean"]) == "bool"
        assert str(X.dtypes.loc["nullable_boolean"]) == "boolean"
    else:
        assert str(X.dtypes.loc["non_nullable_integer"]) == "int64"
        assert str(X.dtypes.loc["nullable_integer"]) == "float64"
        assert str(X.dtypes.loc["non_nullable_boolean"]) == "bool"
        assert str(X.dtypes.loc["nullable_boolean"]) == "object"

    if input_type == "ww":
        X.ww.init()
        if nullable_types_properly_set:
            assert isinstance(X.ww.logical_types["nullable_integer"], IntegerNullable)
            assert isinstance(X.ww.logical_types["nullable_boolean"], BooleanNullable)
        else:
            assert isinstance(X.ww.logical_types["nullable_integer"], Double)
            assert isinstance(X.ww.logical_types["nullable_boolean"], Unknown)

    nullable_types_replacer.fit(X)

    if nullable_types_properly_set:
        # The transformer finds the columns if the datatypes are set properly.
        assert nullable_types_replacer._nullable_int_cols == ["nullable_integer"]
        assert nullable_types_replacer._nullable_bool_cols == ["nullable_boolean"]
    else:
        # If the datatypes are not set properly, it does nothing.
        assert nullable_types_replacer._nullable_int_cols == []
        assert nullable_types_replacer._nullable_bool_cols == []

    X_t = nullable_types_replacer.transform(X)
    assert set(X_t.columns) == set(X.columns)
    assert X_t.shape == X.shape

    if nullable_types_properly_set:
        # The ReplaceNullableTypes transformer swaps 'float64' for NullableInt and
        # 'category' for NullableBool.
        assert str(X_t.dtypes.loc["non_nullable_integer"]) == "int64"
        assert str(X_t.dtypes.loc["nullable_integer"]) == "float64"
        assert str(X_t.dtypes.loc["non_nullable_boolean"]) == "bool"
        assert str(X_t.dtypes.loc["nullable_boolean"]) == "category"
    else:
        # The ReplaceNullableTypes transformer leaves the types unchanged.
        assert str(X_t.dtypes.loc["non_nullable_integer"]) == "int64"
        assert str(X_t.dtypes.loc["nullable_integer"]) == "float64"
        assert str(X_t.dtypes.loc["non_nullable_boolean"]) == "bool"
        if input_type == "ww":
            assert str(X_t.dtypes.loc["nullable_boolean"]) == "string"
        else:
            assert str(X_t.dtypes.loc["nullable_boolean"]) == "object"
