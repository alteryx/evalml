import inspect
import numpy as np
import pandas as pd
import pytest

from evalml.utils.gen_utils import (
    SEED_BOUNDS,
    classproperty,
    convert_to_seconds,
    get_random_seed,
    get_random_state,
    import_or_raise,
    normalize_confusion_matrix
)


def test_import_or_raise_errors():
    with pytest.raises(ImportError, match="Failed to import _evalml"):
        import_or_raise("_evalml")
    with pytest.raises(ImportError, match="error message"):
        import_or_raise("_evalml", "error message")


def test_import_or_raise_imports():
    math = import_or_raise("math", "error message")
    assert math.ceil(0.1) == 1


def test_convert_to_seconds():
    assert convert_to_seconds("10 s") == 10
    assert convert_to_seconds("10 sec") == 10
    assert convert_to_seconds("10 second") == 10
    assert convert_to_seconds("10 seconds") == 10

    assert convert_to_seconds("10 m") == 600
    assert convert_to_seconds("10 min") == 600
    assert convert_to_seconds("10 minute") == 600
    assert convert_to_seconds("10 minutes") == 600

    assert convert_to_seconds("10 h") == 36000
    assert convert_to_seconds("10 hr") == 36000
    assert convert_to_seconds("10 hour") == 36000
    assert convert_to_seconds("10 hours") == 36000


def test_get_random_state():
    assert abs(get_random_state(None).rand() - get_random_state(None).rand()) > 1e-6
    assert get_random_state(42).rand() == np.random.RandomState(42).rand()
    assert get_random_state(np.random.RandomState(42)).rand() == np.random.RandomState(42).rand()


def test_get_random_seed():
    default_min_bound = inspect.signature(get_random_seed).parameters['min_bound'].default
    default_max_bound = inspect.signature(get_random_seed).parameters['max_bound'].default
    assert default_min_bound == SEED_BOUNDS.min_bound
    assert default_max_bound == SEED_BOUNDS.max_bound

    # test default boundaries - value should modulate within the default range
    assert get_random_seed(SEED_BOUNDS.max_bound - 2) == SEED_BOUNDS.max_bound - 2
    assert get_random_seed(SEED_BOUNDS.max_bound - 1) == SEED_BOUNDS.max_bound - 1
    assert get_random_seed(SEED_BOUNDS.max_bound)     == SEED_BOUNDS.min_bound
    assert get_random_seed(SEED_BOUNDS.max_bound + 1) == SEED_BOUNDS.min_bound + 1
    assert get_random_seed(SEED_BOUNDS.max_bound + 2) == SEED_BOUNDS.min_bound + 2
    assert get_random_seed(SEED_BOUNDS.min_bound - 2) == SEED_BOUNDS.max_bound - 2
    assert get_random_seed(SEED_BOUNDS.min_bound - 1) == SEED_BOUNDS.max_bound - 1
    assert get_random_seed(SEED_BOUNDS.min_bound)     == SEED_BOUNDS.min_bound
    assert get_random_seed(SEED_BOUNDS.min_bound + 1) == SEED_BOUNDS.min_bound + 1
    assert get_random_seed(SEED_BOUNDS.min_bound + 2) == SEED_BOUNDS.min_bound + 2

    # test some simple cases
    def get_random_seed_vec(min_bound=None, max_bound=None):
        def get_random_seed_wrapper(random_seed):
            return get_random_seed(random_seed,
                                   min_bound = min_bound if min_bound is not None else default_min_bound,
                                   max_bound = max_bound if max_bound is not None else default_max_bound)
        return np.vectorize(get_random_seed_wrapper)
    vals = np.arange(10)
    np.testing.assert_equal(get_random_seed_vec()(vals), vals)
    np.testing.assert_equal(get_random_seed_vec(min_bound=0)(vals), vals)
    np.testing.assert_equal(get_random_seed_vec(min_bound=0, max_bound=11)(vals), vals % 11)
    np.testing.assert_equal(get_random_seed_vec(min_bound=0, max_bound=11)(vals), vals % 10)

    # test some more complex cases. the value should always stay between min (inclusive) and max (exclusive)
    # this particular implementation was designed to modulate evenly through the defined range as it exceeded the limits.
    vals = np.arange(-100, 100)
    def make_expected_values(vals, min_bound, max_bound):
        return np.array([i if (min_bound <= i and i < max_bound) else ((i - min_bound) % (max_bound - min_bound)) + min_bound
                         for i in vals])
    np.testing.assert_equal(get_random_seed_vec(       min_bound=   0, max_bound=  5)(vals),
                            make_expected_values(vals, min_bound=   0, max_bound=  5))
    np.testing.assert_equal(get_random_seed_vec(       min_bound=  -5, max_bound=  0)(vals),
                            make_expected_values(vals, min_bound=  -5, max_bound=  0))
    np.testing.assert_equal(get_random_seed_vec(       min_bound=  -5, max_bound=  5)(vals),
                            make_expected_values(vals, min_bound=  -5, max_bound=  5))
    np.testing.assert_equal(get_random_seed_vec(       min_bound=   5, max_bound= 10)(vals),
                            make_expected_values(vals, min_bound=   5, max_bound= 10))
    np.testing.assert_equal(get_random_seed_vec(       min_bound= -10, max_bound= -5)(vals),
                            make_expected_values(vals, min_bound= -10, max_bound= -5))

    assert get_random_seed(-2, min_bound=0, max_bound=43) == 41
    assert get_random_seed(-1, min_bound=0, max_bound=43) == 42
    assert get_random_seed(0,  min_bound=0, max_bound=43) == 0
    assert get_random_seed(42, min_bound=0, max_bound=43) == 42
    assert get_random_seed(43, min_bound=0, max_bound=43) == 0
    assert get_random_seed(44, min_bound=0, max_bound=43) == 1
    assert get_random_seed(42, min_bound=42) == 42
    assert get_random_seed(42, max_bound=43) == 42
    assert get_random_seed(42, min_bound=42, max_bound=43) == 42
    assert get_random_seed(-42, min_bound=-42, max_bound=0) == -42
    assert get_random_seed(420, min_bound=-500, max_bound=400) == ((420 - -500) % (400 - -500)) + -500
    assert get_random_seed(-420, min_bound=-400, max_bound=500) == ((-420 - -400) % (500 - -400)) + -400

    with pytest.raises(ValueError):
        get_random_seed(42, min_bound=42, max_bound=42)
    with pytest.raises(ValueError):
        get_random_seed(42, min_bound=420, max_bound=4)


def test_normalize_confusion_matrix():
    conf_mat = np.array([[2, 0, 0], [0, 0, 1], [1, 0, 2]])
    conf_mat_normalized = normalize_confusion_matrix(conf_mat)
    assert all(conf_mat_normalized.sum(axis=1) == 1.0)

    conf_mat_normalized = normalize_confusion_matrix(conf_mat, 'pred')
    for col_sum in conf_mat_normalized.sum(axis=0):
        assert col_sum == 1.0 or col_sum == 0.0

    conf_mat_normalized = normalize_confusion_matrix(conf_mat, 'all')
    assert conf_mat_normalized.sum() == 1.0

    # testing with pd.DataFrames
    conf_mat_df = pd.DataFrame()
    conf_mat_df["col_1"] = [0, 1, 2]
    conf_mat_df["col_2"] = [0, 0, 3]
    conf_mat_df["col_3"] = [2, 0, 0]
    conf_mat_normalized = normalize_confusion_matrix(conf_mat_df)
    assert all(conf_mat_normalized.sum(axis=1) == 1.0)
    assert list(conf_mat_normalized.columns) == ['col_1', 'col_2', 'col_3']

    conf_mat_normalized = normalize_confusion_matrix(conf_mat_df, 'pred')
    for col_sum in conf_mat_normalized.sum(axis=0):
        assert col_sum == 1.0 or col_sum == 0.0

    conf_mat_normalized = normalize_confusion_matrix(conf_mat_df, 'all')
    assert conf_mat_normalized.sum().sum() == 1.0


def test_class_property():
    class MockClass:
        name = "MockClass"

        @classproperty
        def caps_name(cls):
            return cls.name.upper()

    assert MockClass.caps_name == "MOCKCLASS"
