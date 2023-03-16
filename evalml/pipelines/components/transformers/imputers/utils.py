"""Utilities useful for imputers."""
from woodwork.logical_types import (
    AgeNullable,
    IntegerNullable,
)

from evalml.utils.nullable_type_utils import _determine_fractional_type


def _get_new_logical_types_for_imputed_data(
    impute_strategy,
    original_schema,
):
    """Determines what the logical types should be after imputing data. New logical types are only needed for integer data that may have had fractional values imputed.

    Args:
        impute_strategy (str): The strategy used to impute data. May be one of
            "most_frequent", "forwards_fill", "backwards_fill", "mean", "median", "constant", "interpolate, or "knn".
            Integer types will be converted to their corresponding fractional types if any but
            "most_frequent", "forwards_fill" or "backwards_fill" are used.
        original_schema (ww.TableSchema): The Woodwork table schema of the original data that was passed to the imputer.

    Returns:
        dict[str, ww.LogicalType]: Updated logical types to use for imputed data.
    """
    # Some impute strategies will always maintain integer values, so we can use the original logical types
    if impute_strategy in {"most_frequent", "forwards_fill", "backwards_fill"}:
        return original_schema.logical_types

    return {
        col: _determine_fractional_type(ltype)
        if isinstance(ltype, (AgeNullable, IntegerNullable))
        else ltype
        for col, ltype in original_schema.logical_types.items()
    }
