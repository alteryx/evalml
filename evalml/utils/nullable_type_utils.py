import woodwork as ww


# --> add to __init__.py
def _downcast_nullable_X(X, handle_boolean_nullable=True, handle_integer_nullable=True):
    # --> initial look is that this is 3x faster - for fraud and contains_nulls and churn
    # --> breast cancer and wine are faster but not by too much - theyre all doubles
    # --> wine was

    """--> add full docstrings - make sure to note that the handle_integer_nullable refers to the dtype so itll catch both age and itneger ltypes"""
    # --> consider adding param for expecting there to not be any nans present so we're
    # notified if we're ever unknowingly converting to Double or Categorical when we shouldnt in automl search
    if X.ww.schema is None:
        X.ww.init()

    select_param = []
    if handle_boolean_nullable:
        select_param.append("BooleanNullable")
    if handle_integer_nullable:
        select_param.append("IntegerNullable")
        select_param.append("AgeNullable")

    if not select_param:
        return X

    nullable_X_to_downcast = X.ww.select(select_param)
    if not len(nullable_X_to_downcast.columns):
        return X

    # --> try to find a more elegant way to do this
    # --> consider using objects instead of str representations
    downcast_matches = {
        "BooleanNullable": ("Boolean", "Categorical"),
        "IntegerNullable": ("Integer", "Double"),
        # --> age fractional or double? I think AgeFractional to avoid losing info
        "AgeNullable": ("Age", "AgeFractional"),
    }
    cols_with_nans = set(
        nullable_X_to_downcast.columns[nullable_X_to_downcast.isnull().any()],
    )
    original_ltypes = nullable_X_to_downcast.ww.logical_types
    # --> this is probably overly confusing to read. Simplify!
    def get_ltype(type_tuple, col):
        return type_tuple[1] if col in cols_with_nans else type_tuple[0]

    new_ltypes = {
        col: get_ltype(downcast_matches[str(ltype)], col)
        for col, ltype in original_ltypes.items()
    }

    if new_ltypes:
        X.ww.set_types(logical_types=new_ltypes)
    return X


def _downcast_nullable_y(y, handle_boolean_nullable=True, handle_integer_nullable=True):
    """--> add full docstrings"""
    if y.ww.schema is None:
        ww.init_series(y)

    return y


def _get_downcast_type(col_name, logical_types, has_nans):
    # --> might be worth implementing this
    pass
