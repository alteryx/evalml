import pandas as pd

from evalml.utils.nullable_type_utils import (
    _downcast_nullable_X,
    _downcast_nullable_y,
)


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
# --> add fixture like the imputer that has a mixture of nullalbe with and without nans
# --> test with age nullable


def test_downcast_nullable_X_noop_when_no_downcast_needed(imputer_test_data):
    X = imputer_test_data
    original_X = X.ww.copy()

    X_d = _downcast_nullable_X(
        X,
        handle_boolean_nullable=False,
        handle_integer_nullable=False,
    )

    pd.testing.assert_frame_equal(X_d, original_X)


def test_downcast_nullable_X_noop_when_no_nullable_types_present(X_y_binary):
    X, _ = X_y_binary
    original_X = X.ww.copy()

    X_d = _downcast_nullable_X(
        X,
        handle_boolean_nullable=True,
        handle_integer_nullable=True,
    )

    pd.testing.assert_frame_equal(X_d, original_X)


def test_downcast_nullable_X_replaces_nullable_types(imputer_test_data):
    X = imputer_test_data
    original_X = X.ww.copy()

    assert len(original_X.ww.select(["IntegerNullable", "BooleanNullable"]).columns) > 0

    X_d = _downcast_nullable_X(
        X,
        handle_boolean_nullable=True,
        handle_integer_nullable=True,
    )

    assert set(X_d.columns) == set(original_X.columns)
    assert len(X_d.ww.select(["IntegerNullable", "BooleanNullable"]).columns) == 0
