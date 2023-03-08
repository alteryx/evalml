import pytest
from woodwork.logical_types import (
    AgeNullable,
    BooleanNullable,
    IntegerNullable,
)

from evalml.utils import (
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


@pytest.mark.parametrize("has_nans", [True, False])
@pytest.mark.parametrize("bool_null_incompatible", [True, False])
@pytest.mark.parametrize("int_null_incompatible", [True, False])
def test_downcast_nullable_X_replaces_nullable_types(
    split_nullable_logical_types_by_compatibility,
    nullable_type_test_data,
    int_null_incompatible,
    bool_null_incompatible,
    has_nans,
):
    X = nullable_type_test_data(has_nans=has_nans)
    # Set other typing info to confirm it's maintained
    X.ww.init(
        schema=X.ww.schema,
        column_origins={"int col nullable": "base", "float col": "engineered"},
    )
    original_X = X.ww.copy()

    # Confirm nullable types are all present in original data
    assert (
        len(
            original_X.ww.select(
                ["IntegerNullable", "BooleanNullable", "AgeNullable"],
            ).columns,
        )
        > 0
    )

    _, incompatible_ltypes = split_nullable_logical_types_by_compatibility(
        int_null_incompatible,
        bool_null_incompatible,
    )

    compatible_original_schema = original_X.ww.select(
        exclude=incompatible_ltypes,
        return_schema=True,
    )

    X_d = _downcast_nullable_X(
        X,
        handle_boolean_nullable=bool_null_incompatible,
        handle_integer_nullable=int_null_incompatible,
    )

    assert set(X_d.columns) == set(original_X.columns)
    assert len(X_d.ww.select(incompatible_ltypes).columns) == 0
    assert X_d.ww["int col nullable"].ww.origin == "base"

    # Confirm the other columns' woodwork info is unchanged
    undowncasted_partial_schema = original_X.ww.get_subset_schema(
        compatible_original_schema.columns.keys(),
    )
    assert compatible_original_schema == undowncasted_partial_schema


@pytest.mark.parametrize("has_nans", [True, False])
@pytest.mark.parametrize(
    "nullable_ltype",
    ["BooleanNullable", "IntegerNullable", "AgeNullable"],
)
@pytest.mark.parametrize("bool_null_incompatible", [True, False])
@pytest.mark.parametrize("int_null_incompatible", [True, False])
def test_downcast_nullable_y_replaces_nullable_types(
    split_nullable_logical_types_by_compatibility,
    nullable_type_target,
    int_null_incompatible,
    bool_null_incompatible,
    nullable_ltype,
    has_nans,
):
    y = nullable_type_target(ltype=nullable_ltype, has_nans=has_nans)

    y_d = _downcast_nullable_y(
        y,
        handle_boolean_nullable=bool_null_incompatible,
        handle_integer_nullable=int_null_incompatible,
    )

    (
        y_compatible_ltypes,
        y_incompatible_ltypes,
    ) = split_nullable_logical_types_by_compatibility(
        int_null_incompatible,
        bool_null_incompatible,
    )

    if nullable_ltype in {str(ltype) for ltype in y_compatible_ltypes}:
        assert isinstance(
            y_d.ww.logical_type,
            tuple(y_compatible_ltypes),
        )
    else:
        assert not isinstance(
            y_d.ww.logical_type,
            tuple(y_incompatible_ltypes),
        )
