import pytest
from woodwork.logical_types import AgeNullable, BooleanNullable, IntegerNullable

from evalml.utils.nullable_type_utils import (
    _determine_downcast_type,
    _downcast_nullable_X,
    _downcast_nullable_y,
    _get_incompatible_nullable_types,
)


def test_get_incompatible_nullable_types():
    all_nullable_types = _get_incompatible_nullable_types(True, True)
    assert all_nullable_types == [BooleanNullable, IntegerNullable, AgeNullable]

    only_bool_nullable = _get_incompatible_nullable_types(True, False)
    assert only_bool_nullable == [BooleanNullable]

    only_int_nullable = _get_incompatible_nullable_types(False, True)
    assert only_int_nullable == [IntegerNullable, AgeNullable]

    no_nullable_types = _get_incompatible_nullable_types(False, False)
    assert not no_nullable_types


@pytest.mark.parametrize("has_nans", [True, False])
@pytest.mark.parametrize(
    "nullable_ltype",
    ["BooleanNullable", "IntegerNullable", "AgeNullable"],
)
def test_determine_downcast_type(nullable_type_target, nullable_ltype, has_nans):
    col = nullable_type_target(ltype=nullable_ltype, has_nans=has_nans)

    downcast_ltype = _determine_downcast_type(col)

    # Confirm we're setting the downcast type correctly according to whether nans are present
    if has_nans:
        assert downcast_ltype in ("AgeFractional", "Double", "Categorical")
    else:
        assert downcast_ltype in ("Age", "Integer", "Boolean")

    # Confirm we're choosing the correct downcast types for the original nullable type
    if nullable_ltype == "BooleanNullable":
        assert downcast_ltype in ("Boolean", "Categorical")
    elif nullable_ltype == "IntegerNullable":
        assert downcast_ltype in ("Integer", "Double")
    elif nullable_ltype == "AgeNullable":
        assert downcast_ltype in ("Age", "AgeFractional")


@pytest.mark.parametrize(
    "downcast_util, data_type",
    [(_downcast_nullable_X, "X"), (_downcast_nullable_y, "y")],
)
def test_downcast_utils_handle_woodwork_not_init(X_y_binary, downcast_util, data_type):
    X, y = X_y_binary
    # Remove woodwork types
    if data_type == "X":
        data = X.copy()
    else:
        data = y.copy()
    assert data.ww.schema is None

    data_d = downcast_util(data)
    assert data_d.ww.schema is not None


@pytest.mark.parametrize(
    "downcast_util, data_type",
    [(_downcast_nullable_X, "X"), (_downcast_nullable_y, "y")],
)
def test_downcast_nullable_X_noop_when_no_downcast_needed(
    nullable_type_test_data,
    nullable_type_target,
    downcast_util,
    data_type,
):
    if data_type == "X":
        data = nullable_type_test_data()
    else:
        data = nullable_type_target(ltype="IntegerNullable")

    data_d = downcast_util(
        data,
        handle_boolean_nullable=False,
        handle_integer_nullable=False,
    )

    assert data_d is data


@pytest.mark.parametrize(
    "downcast_util, data_type",
    [(_downcast_nullable_X, "X"), (_downcast_nullable_y, "y")],
)
def test_downcast_nullable_X_noop_when_no_nullable_types_present(
    X_y_binary,
    downcast_util,
    data_type,
):
    X, y = X_y_binary
    if data_type == "X":
        data = X
        assert (
            len(
                data.ww.select(
                    ["IntegerNullable", "AgeNullable", "BooleanNullable"],
                ).columns,
            )
            == 0
        )
    else:
        data = y
        assert not isinstance(
            data.ww.logical_type,
            (IntegerNullable, BooleanNullable, AgeNullable),
        )

    data_d = downcast_util(
        data,
        handle_boolean_nullable=True,
        handle_integer_nullable=True,
    )

    assert data_d is data


def test_downcast_nullable_X_replaces_nullable_types(nullable_type_test_data):
    X = nullable_type_test_data()
    # Set other typing info to confirm it's maintained
    X.ww.init(
        schema=X.ww.schema,
        column_origins={"int with nan": "base", "float col": "engineered"},
    )
    original_X = X.ww.copy()

    assert (
        len(
            original_X.ww.select(
                ["IntegerNullable", "BooleanNullable", "AgeNullable"],
            ).columns,
        )
        > 0
    )
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
    assert X_d.ww["int with nan"].ww.origin == "base"

    # Check that no nullable types remain
    assert (
        len(
            X_d.ww.select(
                ["IntegerNullable", "BooleanNullable", "AgeNullable"],
            ).columns,
        )
        == 0
    )

    # Confirm the other columns' woodwork info is unchanged
    undowncasted_schema = original_X.ww.get_subset_schema(
        non_nullable_original_schema.columns.keys(),
    )
    assert non_nullable_original_schema == undowncasted_schema


def test_downcast_nullable_X_only_bools(nullable_type_test_data):
    X = nullable_type_test_data()

    X_d = _downcast_nullable_X(
        X,
        handle_boolean_nullable=True,
        handle_integer_nullable=False,
    )

    # Check that only the expected types remain
    assert len(X_d.ww.select(["BooleanNullable"]).columns) == 0
    assert len(X_d.ww.select(["IntegerNullable", "AgeNullable"]).columns) > 0


def test_downcast_nullable_X_only_ints(nullable_type_test_data):
    X = nullable_type_test_data()

    X_d = _downcast_nullable_X(
        X,
        handle_boolean_nullable=False,
        handle_integer_nullable=True,
    )

    # Check that only the expected types remain
    assert len(X_d.ww.select(["IntegerNullable", "AgeNullable"]).columns) == 0
    assert len(X_d.ww.select(["BooleanNullable"]).columns) > 0


@pytest.mark.parametrize("has_nans", [True, False])
@pytest.mark.parametrize(
    "nullable_ltype",
    ["BooleanNullable", "IntegerNullable", "AgeNullable"],
)
def test_downcast_nullable_y_replaces_nullable_types(
    nullable_type_target,
    nullable_ltype,
    has_nans,
):
    y = nullable_type_target(ltype=nullable_ltype, has_nans=has_nans)

    y_d = _downcast_nullable_y(
        y,
        handle_boolean_nullable=True,
        handle_integer_nullable=True,
    )

    assert not isinstance(
        y_d.ww.logical_type,
        (AgeNullable, IntegerNullable, BooleanNullable),
    )


@pytest.mark.parametrize("has_nans", [True, False])
@pytest.mark.parametrize(
    "nullable_ltype",
    ["BooleanNullable", "IntegerNullable", "AgeNullable"],
)
def test_downcast_nullable_y_only_bools(nullable_type_target, nullable_ltype, has_nans):
    y = nullable_type_target(ltype=nullable_ltype, has_nans=has_nans)
    original_ltype = y.ww.logical_type

    y_d = _downcast_nullable_y(
        y,
        handle_boolean_nullable=True,
        handle_integer_nullable=False,
    )

    if nullable_ltype in ["BooleanNullable"]:
        assert not isinstance(
            y_d.ww.logical_type,
            BooleanNullable,
        )
    else:
        assert y_d.ww.logical_type == original_ltype


@pytest.mark.parametrize("has_nans", [True, False])
@pytest.mark.parametrize(
    "nullable_ltype",
    ["BooleanNullable", "IntegerNullable", "AgeNullable"],
)
def test_downcast_nullable_y_only_ints(nullable_type_target, nullable_ltype, has_nans):
    y = nullable_type_target(ltype=nullable_ltype, has_nans=has_nans)
    original_ltype = y.ww.logical_type

    y_d = _downcast_nullable_y(
        y,
        handle_boolean_nullable=False,
        handle_integer_nullable=True,
    )

    if nullable_ltype in ["IntegerNullable", "AgeNullable"]:
        assert not isinstance(
            y_d.ww.logical_type,
            (AgeNullable, IntegerNullable),
        )
    else:
        assert y_d.ww.logical_type == original_ltype
