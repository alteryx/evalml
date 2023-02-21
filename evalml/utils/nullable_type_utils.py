import woodwork as ww


# --> add to __init__.py
def _downcast_nullable_X(X, handle_boolean_nullable=True, handle_integer_nullable=True):
    """--> add full docstrings"""
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
    downcast_matches = {
        "BooleanNullable": ("Boolean", "Categorical"),
        "IntegerNullable": ("Integer", "Double"),
        "AgeNullable": ("Integer", "Double"),
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
