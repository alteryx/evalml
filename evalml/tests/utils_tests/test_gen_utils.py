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
    for i in range(10):
        print('Running get_random_seed test cases with i={}'.format(i))
        assert get_random_seed(i) == i
        assert get_random_seed(i, min_bound=0) == i
        assert get_random_seed(i, min_bound=0, max_bound=11) == i
        assert get_random_seed(i, min_bound=0, max_bound=10) == i % 10
        assert get_random_seed(i, min_bound=0, max_bound=5) == i % 5
        if 5 <= i and i < 11:
            assert get_random_seed(i, min_bound=5, max_bound=11) == i
        else:
            assert get_random_seed(i, min_bound=5, max_bound=11) == (i % 6) + 5
        if 2 <= i and i < 8:
            assert get_random_seed(i, min_bound=2, max_bound=8) == i
        else:
            assert get_random_seed(i, min_bound=2, max_bound=5) == (i % 3) + 2

    assert get_random_seed(0) == 0
    assert get_random_seed(1) == 1
    assert get_random_seed(42) == 42
    assert get_random_seed(-42) == -42
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
    assert get_random_seed(420, min_bound=-500, max_bound=400) == -500 + (420 % (400 - -500))
    assert get_random_seed(-420, min_bound=-400, max_bound=500) == -400 + (-420 % (500 - -400))

    assert get_random_seed(SEED_BOUNDS.max_bound - 2) == SEED_BOUNDS.max_bound - 2
    assert get_random_seed(SEED_BOUNDS.max_bound - 1) == SEED_BOUNDS.max_bound - 1
    assert get_random_seed(SEED_BOUNDS.max_bound)     == 0
    assert get_random_seed(SEED_BOUNDS.max_bound + 1) == 1
    assert get_random_seed(SEED_BOUNDS.max_bound + 2) == 2
    assert get_random_seed(SEED_BOUNDS.min_bound - 2) == -2
    assert get_random_seed(SEED_BOUNDS.min_bound - 1) == -1
    assert get_random_seed(SEED_BOUNDS.min_bound)     == SEED_BOUNDS.min_bound
    assert get_random_seed(SEED_BOUNDS.min_bound + 1) == SEED_BOUNDS.min_bound + 1
    assert get_random_seed(SEED_BOUNDS.min_bound + 2) == SEED_BOUNDS.min_bound + 2

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
