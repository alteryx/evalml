import pandas as pd
import pytest

from evalml.pipelines.components import ReplaceNullableTypes


@pytest.mark.parametrize("input_type", ["ww", "pandas"])
def test_replace_nullable_types(input_type):
    replace_nullable_types_transformer = ReplaceNullableTypes()
    X = pd.DataFrame(
        {
            "non_nullable_integer": [0, 1, 2, 3, 4],
            "nullable_integer": [0, 1, 2, 3, None],
            "non_nullable_boolean": [True, False, True, False, True],
            "nullable_boolean": [None, True, False, True, False],
        }
    )
    X = X.astype({"nullable_integer": "Int64", "nullable_boolean": "boolean"})

    # Check that the underlying pandas data types are set properly
    assert str(X.dtypes.loc["non_nullable_integer"]) == "int64"
    assert (
        str(X.dtypes.loc["nullable_integer"]) == "Int64" # <- the new nullable integer
    )
    assert str(X.dtypes.loc["non_nullable_boolean"]) == "bool"
    assert (
        str(X.dtypes.loc["nullable_boolean"]) == "boolean" # <- the new nullable boolean
    )

    if input_type == "ww":
        X.ww.init()

    replace_nullable_types_transformer.fit(X)
    # Check that the transformer found the right columns with the nullable types
    assert replace_nullable_types_transformer._nullable_int_cols == ["nullable_integer"]
    assert replace_nullable_types_transformer._nullable_bool_cols == [
        "nullable_boolean"
    ]

    X_t = replace_nullable_types_transformer.transform(X)
    assert set(X_t.columns) == set(X.columns)
    assert X_t.shape == X.shape

    # Check that the underlying pandas data types have changed properly
    assert str(X_t.dtypes.loc["non_nullable_integer"]) == "int64"
    assert str(X_t.dtypes.loc["nullable_integer"]) == "float64"
    assert str(X_t.dtypes.loc["non_nullable_boolean"]) == "bool"
    assert str(X_t.dtypes.loc["nullable_boolean"]) == "category"
