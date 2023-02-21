import pandas as pd
from woodwork.logical_types import (
    Age,
    AgeFractional,
    AgeNullable,
    Boolean,
    BooleanNullable,
    Categorical,
    Double,
    Integer,
    IntegerNullable,
)

from evalml.utils.nullable_type_utils import _downcast_nullable_X, _downcast_nullable_y


def test_downcast_utils_handle_woodwork_not_init(X_y_binary):
    X, y = X_y_binary
    # Remove woodwork types
    X = X.copy()
    y = y.copy()

    assert X.ww.schema is None
    assert y.ww.schema is None

    X_d = _downcast_nullable_X(X)
    y_d = _downcast_nullable_y(y)

    assert X_d.ww.schema is not None
    assert y_d.ww.schema is not None


# --> test that we're always initing y with init_series so that we never get a type conversion error
# --> confirm nan values are maintained


def test_downcast_nullable_X_noop_when_no_downcast_needed(imputer_test_data):
    X = imputer_test_data
    original_X = X.ww.copy()

    assert (
        len(X.ww.select(["IntegerNullable", "AgeNullable", "BooleanNullable"]).columns)
        > 0
    )
    X_d = _downcast_nullable_X(
        X,
        handle_boolean_nullable=False,
        handle_integer_nullable=False,
    )

    pd.testing.assert_frame_equal(X_d, original_X)


def test_downcast_nullable_X_noop_when_no_nullable_types_present(X_y_binary):
    X, _ = X_y_binary
    original_X = X.ww.copy()

    assert (
        len(X.ww.select(["IntegerNullable", "AgeNullable", "BooleanNullable"]).columns)
        == 0
    )

    X_d = _downcast_nullable_X(
        X,
        handle_boolean_nullable=True,
        handle_integer_nullable=True,
    )

    pd.testing.assert_frame_equal(X_d, original_X)


def test_downcast_nullable_X_replaces_nullable_types(nullable_type_test_data):
    X = nullable_type_test_data()
    # Set other typing info to confirm it's maintained
    X.ww.init(
        schema=X.ww.schema,
        column_origins={"int col nullable": "base", "float col": "engineered"},
    )
    original_X = X.ww.copy()

    assert len(original_X.ww.select(["IntegerNullable", "BooleanNullable"]).columns) > 0
    non_nullable_original_schema = original_X.ww.select(
        exclude=["IntegerNullable", "BooleanNullable"],
        return_schema=True,
    )

    X_d = _downcast_nullable_X(
        X,
        handle_boolean_nullable=True,
        handle_integer_nullable=True,
    )

    assert set(X_d.columns) == set(original_X.columns)
    assert len(X_d.ww.select(["IntegerNullable", "BooleanNullable"]).columns) == 0
    assert X_d.ww["int col nullable"].ww.origin == "base"

    # Check the correct logical types were used at downcast
    downcast_ltypes = X_d.ww.logical_types
    assert isinstance(downcast_ltypes["int col nullable"], Integer)
    assert isinstance(downcast_ltypes["int with nan"], Double)
    assert isinstance(downcast_ltypes["bool col nullable"], Boolean)
    assert isinstance(downcast_ltypes["bool with nan"], Categorical)
    assert isinstance(downcast_ltypes["age col nullable"], Age)
    assert isinstance(downcast_ltypes["age with nan"], AgeFractional)

    # Confirm the other columns' woodwork info is unchanged
    undowncasted_schema = original_X.ww.get_subset_schema(
        non_nullable_original_schema.columns.keys(),
    )
    assert non_nullable_original_schema == undowncasted_schema


def test_downcast_nullable_X_only_bools(nullable_type_test_data):
    # --> consider parameterizing this and test below
    X = nullable_type_test_data()

    X_d = _downcast_nullable_X(
        X,
        handle_boolean_nullable=True,
        handle_integer_nullable=False,
    )
    downcast_ltypes = X_d.ww.logical_types

    assert isinstance(downcast_ltypes["int col nullable"], IntegerNullable)
    assert isinstance(downcast_ltypes["int with nan"], IntegerNullable)
    assert isinstance(downcast_ltypes["bool col nullable"], Boolean)
    assert isinstance(downcast_ltypes["bool with nan"], Categorical)
    assert isinstance(downcast_ltypes["age col nullable"], AgeNullable)
    assert isinstance(downcast_ltypes["age with nan"], AgeNullable)


def test_downcast_nullable_X_only_ints(nullable_type_test_data):
    X = nullable_type_test_data()

    X_d = _downcast_nullable_X(
        X,
        handle_boolean_nullable=False,
        handle_integer_nullable=True,
    )
    downcast_ltypes = X_d.ww.logical_types

    assert isinstance(downcast_ltypes["int col nullable"], Integer)
    assert isinstance(downcast_ltypes["int with nan"], Double)
    assert isinstance(downcast_ltypes["bool col nullable"], BooleanNullable)
    assert isinstance(downcast_ltypes["bool with nan"], BooleanNullable)
    assert isinstance(downcast_ltypes["age col nullable"], Age)
    assert isinstance(downcast_ltypes["age with nan"], AgeFractional)
