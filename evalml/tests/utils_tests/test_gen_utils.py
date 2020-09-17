import inspect
from unittest.mock import patch

import numpy as np
import pytest

from evalml.pipelines.components import ComponentBase
from evalml.utils.gen_utils import (
    SEED_BOUNDS,
    check_random_state_equality,
    classproperty,
    convert_to_seconds,
    get_importable_subclasses,
    get_random_seed,
    get_random_state,
    import_or_raise
)


@patch('importlib.import_module')
def test_import_or_raise_errors(dummy_importlib):
    def _mock_import_function(library_str):
        if library_str == "_evalml":
            raise ImportError("Mock ImportError executed!")
        if library_str == "attr_error_lib":
            raise Exception("Mock Exception executed!")

    dummy_importlib.side_effect = _mock_import_function

    with pytest.raises(ImportError, match="Missing optional dependency '_evalml'"):
        import_or_raise("_evalml")
    with pytest.raises(ImportError, match="Missing optional dependency '_evalml'. Please use pip to install _evalml. Additional error message"):
        import_or_raise("_evalml", "Additional error message")
    with pytest.raises(Exception, match="An exception occurred while trying to import `attr_error_lib`: Mock Exception executed!"):
        import_or_raise("attr_error_lib")


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

    with pytest.raises(AssertionError, match="Invalid unit."):
        convert_to_seconds("10 years")


def test_get_random_state_int():
    assert abs(get_random_state(None).rand() - get_random_state(None).rand()) > 1e-6
    assert get_random_state(42).rand() == np.random.RandomState(42).rand()
    assert get_random_state(np.random.RandomState(42)).rand() == np.random.RandomState(42).rand()
    assert get_random_state(SEED_BOUNDS.min_bound).rand() == np.random.RandomState(SEED_BOUNDS.min_bound).rand()
    assert get_random_state(SEED_BOUNDS.max_bound).rand() == np.random.RandomState(SEED_BOUNDS.max_bound).rand()
    with pytest.raises(ValueError, match=r'Seed "[-0-9]+" is not in the range \[{}, {}\], inclusive'.format(
            SEED_BOUNDS.min_bound, SEED_BOUNDS.max_bound)):
        get_random_state(SEED_BOUNDS.min_bound - 1)
    with pytest.raises(ValueError, match=r'Seed "[-0-9]+" is not in the range \[{}, {}\], inclusive'.format(
            SEED_BOUNDS.min_bound, SEED_BOUNDS.max_bound)):
        get_random_state(SEED_BOUNDS.max_bound + 1)


def test_get_random_seed_rng():

    def make_mock_random_state(return_value):

        class MockRandomState(np.random.RandomState):
            def __init__(self):
                self.min_bound = None
                self.max_bound = None
                super().__init__()

            def randint(self, min_bound, max_bound):
                self.min_bound = min_bound
                self.max_bound = max_bound
                return return_value
        return MockRandomState()

    rng = make_mock_random_state(42)
    assert get_random_seed(rng) == 42
    assert rng.min_bound == SEED_BOUNDS.min_bound
    assert rng.max_bound == SEED_BOUNDS.max_bound


def test_get_random_seed_int():
    # ensure the invariant "min_bound < max_bound" is enforced
    with pytest.raises(ValueError):
        get_random_seed(0, min_bound=0, max_bound=0)
    with pytest.raises(ValueError):
        get_random_seed(0, min_bound=0, max_bound=-1)

    # test default boundaries to show the provided value should modulate within the default range
    assert get_random_seed(SEED_BOUNDS.max_bound - 2) == SEED_BOUNDS.max_bound - 2
    assert get_random_seed(SEED_BOUNDS.max_bound - 1) == SEED_BOUNDS.max_bound - 1
    assert get_random_seed(SEED_BOUNDS.max_bound) == SEED_BOUNDS.min_bound
    assert get_random_seed(SEED_BOUNDS.max_bound + 1) == SEED_BOUNDS.min_bound + 1
    assert get_random_seed(SEED_BOUNDS.max_bound + 2) == SEED_BOUNDS.min_bound + 2
    assert get_random_seed(SEED_BOUNDS.min_bound - 2) == SEED_BOUNDS.max_bound - 2
    assert get_random_seed(SEED_BOUNDS.min_bound - 1) == SEED_BOUNDS.max_bound - 1
    assert get_random_seed(SEED_BOUNDS.min_bound) == SEED_BOUNDS.min_bound
    assert get_random_seed(SEED_BOUNDS.min_bound + 1) == SEED_BOUNDS.min_bound + 1
    assert get_random_seed(SEED_BOUNDS.min_bound + 2) == SEED_BOUNDS.min_bound + 2

    # vectorize get_random_seed via a wrapper for easy evaluation
    default_min_bound = inspect.signature(get_random_seed).parameters['min_bound'].default
    default_max_bound = inspect.signature(get_random_seed).parameters['max_bound'].default
    assert default_min_bound == SEED_BOUNDS.min_bound
    assert default_max_bound == SEED_BOUNDS.max_bound

    def get_random_seed_vec(min_bound=None, max_bound=None):  # passing None for either means no value is provided to get_random_seed

        def get_random_seed_wrapper(random_seed):
            return get_random_seed(random_seed,
                                   min_bound=min_bound if min_bound is not None else default_min_bound,
                                   max_bound=max_bound if max_bound is not None else default_max_bound)

        return np.vectorize(get_random_seed_wrapper)

    # ensure that regardless of the setting of min_bound and max_bound, the output of get_random_seed always stays
    # between the min_bound (inclusive) and max_bound (exclusive), and wraps neatly around that range using modular arithmetic.
    vals = np.arange(-100, 100)

    def make_expected_values(vals, min_bound, max_bound):
        return np.array([i if (min_bound <= i and i < max_bound) else ((i - min_bound) % (max_bound - min_bound)) + min_bound
                         for i in vals])

    np.testing.assert_equal(get_random_seed_vec(min_bound=None, max_bound=None)(vals),
                            make_expected_values(vals, min_bound=SEED_BOUNDS.min_bound, max_bound=SEED_BOUNDS.max_bound))
    np.testing.assert_equal(get_random_seed_vec(min_bound=None, max_bound=10)(vals),
                            make_expected_values(vals, min_bound=SEED_BOUNDS.min_bound, max_bound=10))
    np.testing.assert_equal(get_random_seed_vec(min_bound=-10, max_bound=None)(vals),
                            make_expected_values(vals, min_bound=-10, max_bound=SEED_BOUNDS.max_bound))
    np.testing.assert_equal(get_random_seed_vec(min_bound=0, max_bound=5)(vals),
                            make_expected_values(vals, min_bound=0, max_bound=5))
    np.testing.assert_equal(get_random_seed_vec(min_bound=-5, max_bound=0)(vals),
                            make_expected_values(vals, min_bound=-5, max_bound=0))
    np.testing.assert_equal(get_random_seed_vec(min_bound=-5, max_bound=5)(vals),
                            make_expected_values(vals, min_bound=-5, max_bound=5))
    np.testing.assert_equal(get_random_seed_vec(min_bound=5, max_bound=10)(vals),
                            make_expected_values(vals, min_bound=5, max_bound=10))
    np.testing.assert_equal(get_random_seed_vec(min_bound=-10, max_bound=-5)(vals),
                            make_expected_values(vals, min_bound=-10, max_bound=-5))


def test_class_property():
    class MockClass:
        name = "MockClass"

        @classproperty
        def caps_name(cls):
            return cls.name.upper()

    assert MockClass.caps_name == "MOCKCLASS"


def test_get_importable_subclasses_wont_get_custom_classes():

    class ChildClass(ComponentBase):
        pass

    assert ChildClass not in get_importable_subclasses(ComponentBase)


@patch('importlib.import_module')
def test_import_or_warn_errors(dummy_importlib):
    def _mock_import_function(library_str):
        if library_str == "_evalml":
            raise ImportError("Mock ImportError executed!")
        if library_str == "attr_error_lib":
            raise Exception("Mock Exception executed!")

    dummy_importlib.side_effect = _mock_import_function

    with pytest.warns(UserWarning, match="Missing optional dependency '_evalml'"):
        import_or_raise("_evalml", warning=True)
    with pytest.warns(UserWarning, match="Missing optional dependency '_evalml'. Please use pip to install _evalml. Additional error message"):
        import_or_raise("_evalml", "Additional error message", warning=True)
    with pytest.warns(UserWarning, match="An exception occurred while trying to import `attr_error_lib`: Mock Exception executed!"):
        import_or_raise("attr_error_lib", warning=True)


def test_check_random_state_equality():
    assert check_random_state_equality(get_random_state(1), get_random_state(1))

    rs_1 = get_random_state(1)
    rs_2 = get_random_state(2)
    assert not check_random_state_equality(rs_1, rs_2)

    # Test equality
    rs_1.set_state(tuple(['MT19937', np.array([1] * 624), 0, 1, 0.1]))
    rs_2.set_state(tuple(['MT19937', np.array([1] * 624), 0, 1, 0.1]))
    assert check_random_state_equality(rs_1, rs_2)

    # Test numpy array value not equal
    rs_1.set_state(tuple(['MT19937', np.array([0] * 624), 0, 1, 0.1]))
    rs_2.set_state(tuple(['MT19937', np.array([1] * 624), 1, 1, 0.1]))
    assert not check_random_state_equality(rs_1, rs_2)

    # Test non-numpy array value not equal
    rs_1.set_state(tuple(['MT19937', np.array([1] * 624), 0, 1, 0.1]))
    rs_2.set_state(tuple(['MT19937', np.array([1] * 624), 1, 1, 0.1]))
    assert not check_random_state_equality(rs_1, rs_2)
