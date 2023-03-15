from woodwork.logical_types import (
    AgeNullable,
    IntegerNullable,
)

from evalml.utils.nullable_type_utils import _determine_fractional_type


def _get_new_logical_types_for_imputed_data(
    impute_strategy,
    original_schema,
):
    # --> needs docstring
    # Some impute strategies will always maintain integer values, so we can use the original logical types
    if impute_strategy in {"most_frequent", "forwards_fill", "backwards_fill"}:
        return original_schema.logical_types

    return {
        col: _determine_fractional_type(ltype)
        if isinstance(ltype, (AgeNullable, IntegerNullable))
        else ltype
        for col, ltype in original_schema.logical_types.items()
    }
