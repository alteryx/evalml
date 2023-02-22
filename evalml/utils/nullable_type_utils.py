import woodwork as ww


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

    select_param = []
    if handle_boolean_nullable:
        select_param.append("BooleanNullable")
    if handle_integer_nullable:
        select_param.append("IntegerNullable")
        select_param.append("AgeNullable")

    nullable_X_to_downcast = X.ww.select(select_param)
    if not len(nullable_X_to_downcast.columns):
        return X

    cols_with_nans = set(
        nullable_X_to_downcast.columns[nullable_X_to_downcast.isnull().any()],
    )

    new_ltypes = {
        col: _get_downcast_logical_type(ltype, col in cols_with_nans)
        for col, ltype in nullable_X_to_downcast.ww.logical_types.items()
    }

    if new_ltypes:
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
        ww.init_series(y)

    nullable_types_to_handle = []
    if handle_boolean_nullable:
        nullable_types_to_handle.append(ww.logical_types.BooleanNullable)
    if handle_integer_nullable:
        nullable_types_to_handle.append(ww.logical_types.IntegerNullable)
        nullable_types_to_handle.append(ww.logical_types.AgeNullable)

    if isinstance(y.ww.logical_type, tuple(nullable_types_to_handle)):
        new_ltype = _get_downcast_logical_type(y.ww.logical_type, y.isnull().any())
        return y.ww.set_logical_type(new_ltype)

    return y


def _get_downcast_logical_type(nullable_logical_type, data_has_nans):
    """Determines what logical type to downcast to based on whether nans were present or not.
        - BooleanNullable becomes Boolean if nans are not present and Categorical if they are
        - IntegerNullable becomes Integer if nans are not present and Double if they are.
        - AgeNullable becomes Age if nans are not present and AgeFractional if they are.

    Args:
        nullable_logical_type (str): String representation of the Woodwork LogicalType to downcast
        data_has_nans (bool): Whether or not nans were present in the data.
            Determines whether a non nullable LogicalType can be used for downcasting or not.

    Returns:
        LogicalType string to be used to downcast incompatible nullable logical types.
    """
    downcast_matches = {
        "BooleanNullable": ("Boolean", "Categorical"),
        "IntegerNullable": ("Integer", "Double"),
        "AgeNullable": ("Age", "AgeFractional"),
    }

    no_nans_ltype, has_nans_ltype = downcast_matches[str(nullable_logical_type)]

    if data_has_nans:
        return has_nans_ltype

    return no_nans_ltype
