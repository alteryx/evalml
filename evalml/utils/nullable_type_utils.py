import woodwork as ww
from woodwork.logical_types import AgeNullable, BooleanNullable, IntegerNullable

DOWNCAST_TYPE_DICT = {
    "BooleanNullable": ("Boolean", "Categorical"),
    "IntegerNullable": ("Integer", "Double"),
    "AgeNullable": ("Age", "AgeFractional"),
}


def _downcast_nullable_X(X, handle_boolean_nullable=True, handle_integer_nullable=True):
    """Removes Pandas nullable integer and nullable boolean dtypes from data by transforming
        to other dtypes via Woodwork logical type transformations.

    Args:
        X (pd.DataFrame): Input data of shape [n_samples, n_features] whose nullable types will be changed.
        handle_boolean_nullable (bool, optional): Whether or not to downcast data with BooleanNullable logical types.
        handle_integer_nullable (bool, optional): Whether or not to downcast data with IntegerNullable or AgeNullable logical types.


    Returns:
        X with any incompatible nullable types downcasted to compatible equivalents.
    """
    if X.ww.schema is None:
        X.ww.init()

    incompatible_logical_types = _get_incompatible_nullable_types(
        handle_boolean_nullable,
        handle_integer_nullable,
    )

    data_to_downcast = X.ww.select(incompatible_logical_types)
    # If no incompatible types are present, no downcasting is needed
    if not len(data_to_downcast.columns):
        return X

    new_ltypes = {
        col: _determine_downcast_type(data_to_downcast.ww[col])
        for col in data_to_downcast.columns
    }

    X.ww.set_types(logical_types=new_ltypes)
    return X


def _downcast_nullable_y(y, handle_boolean_nullable=True, handle_integer_nullable=True):
    """Removes Pandas nullable integer and nullable boolean dtypes from data by transforming
        to other dtypes via Woodwork logical type transformations.

    Args:
        y (pd.Series): Target data of shape [n_samples] whose nullable types will be changed.
        handle_boolean_nullable (bool, optional): Whether or not to downcast data with BooleanNullable logical types.
        handle_integer_nullable (bool, optional): Whether or not to downcast data with IntegerNullable or AgeNullable logical types.


    Returns:
        y with any incompatible nullable types downcasted to compatible equivalents.
    """
    if y.ww.schema is None:
        y = ww.init_series(y)

    incompatible_logical_types = _get_incompatible_nullable_types(
        handle_boolean_nullable,
        handle_integer_nullable,
    )

    if isinstance(y.ww.logical_type, tuple(incompatible_logical_types)):
        new_ltype = _determine_downcast_type(y)
        return y.ww.set_logical_type(new_ltype)

    return y


def _get_incompatible_nullable_types(handle_boolean_nullable, handle_integer_nullable):
    """Determines which Woodwork logical types are incompatible.

    Args:
        handle_boolean_nullable (bool): Whether boolean nullable logical types are incompatible.
        handle_integer_nullable (bool): Whether integer nullable logical types are incompatible.

    Returns:
        list[ww.LogicalType] of logical types that are incompatible.
    """
    nullable_types_to_handle = []
    if handle_boolean_nullable:
        nullable_types_to_handle.append(BooleanNullable)
    if handle_integer_nullable:
        nullable_types_to_handle.append(IntegerNullable)
        nullable_types_to_handle.append(AgeNullable)

    return nullable_types_to_handle


def _determine_downcast_type(col):
    """Determines what logical type to downcast to based on whether nans were present or not.
        - BooleanNullable becomes Boolean if nans are not present and Categorical if they are
        - IntegerNullable becomes Integer if nans are not present and Double if they are.
        - AgeNullable becomes Age if nans are not present and AgeFractional if they are.

    Args:
        col (Woodwork Series): The data whose downcast logical type we are determining by inspecting
            its current logical type and whether nans are present.

    Returns:
        LogicalType string to be used to downcast incompatible nullable logical types.
    """
    no_nans_ltype, has_nans_ltype = DOWNCAST_TYPE_DICT[str(col.ww.logical_type)]
    if col.isnull().any():
        return has_nans_ltype

    return no_nans_ltype


def _determine_fractional_type(logical_type):
    """Determines what logical type to use for integer data that has fractional values imputed.
    - IntegerNullable becomes Double.
    - AgeNullable becomes AgeFractional.
    - All other logical types are returned unchanged.

    Args:
        logical_type (ww.LogicalType): The logical type whose fractional equivalent we are determining.
            Should be either IntegerNullable or AgeNullable.

    Returns:
        LogicalType to be used after fractional values have been added to a previously integer column.
    """
    fractional_ltype = None
    if isinstance(logical_type, (IntegerNullable, AgeNullable)):
        _, fractional_ltype = DOWNCAST_TYPE_DICT[str(logical_type)]

    return fractional_ltype or logical_type


def _determine_non_nullable_equivalent(logical_type):
    """Determines the non nullable equivalent logical type to use for nullable types. These types cannot support null values.
    - IntegerNullable becomes Integer.
    - AgeNullable becomes Age.
    - BooleanNullable becomes Boolean.
    - All other logical types are returned unchanged.

    Args:
        logical_type (ww.LogicalType): The logical type whose non nullable equivalent we are determining.

    Returns:
        LogicalType to be used instead of nullable type when nans aren't present.
    """
    non_nullable_ltype, _ = DOWNCAST_TYPE_DICT.get(str(logical_type), (None, None))

    return non_nullable_ltype or logical_type


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
