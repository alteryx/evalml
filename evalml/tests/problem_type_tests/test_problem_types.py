import numpy as np
import pandas as pd
import pytest

from evalml.problem_types import (
    ProblemTypes,
    detect_problem_type,
    handle_problem_types,
    is_binary,
    is_classification,
    is_multiclass,
    is_regression,
    is_time_series,
)


@pytest.fixture
def correct_problem_types():
    # Unit tests expect this order
    correct_problem_types = [
        ProblemTypes.REGRESSION,
        ProblemTypes.MULTICLASS,
        ProblemTypes.BINARY,
        ProblemTypes.TIME_SERIES_REGRESSION,
        ProblemTypes.TIME_SERIES_BINARY,
        ProblemTypes.TIME_SERIES_MULTICLASS,
    ]
    yield correct_problem_types


def test_handle_string(correct_problem_types):
    problem_types = [
        "regression",
        ProblemTypes.MULTICLASS,
        "binary",
        ProblemTypes.TIME_SERIES_REGRESSION,
        "time series binary",
        "time series multiclass",
    ]
    for problem_type in zip(problem_types, correct_problem_types):
        assert handle_problem_types(problem_type[0]) == problem_type[1]

    problem_type = "fake"
    error_msg = "Problem type 'fake' does not exist"
    with pytest.raises(KeyError, match=error_msg):
        handle_problem_types(problem_type) == ProblemTypes.REGRESSION


def test_handle_problem_types(correct_problem_types):
    for problem_type in correct_problem_types:
        assert handle_problem_types(problem_type) == problem_type


def test_handle_incorrect_type():
    error_msg = "`handle_problem_types` was not passed a str or ProblemTypes object"
    with pytest.raises(ValueError, match=error_msg):
        handle_problem_types(5)


def test_detect_problem_type_error():
    y_empty = pd.Series([])
    y_one_value = pd.Series([1, 1, 1, 1, 1, 1])
    y_nan = pd.Series([np.nan, np.nan, 0])
    y_all_nan = pd.Series([np.nan, np.nan])

    with pytest.raises(ValueError, match="Less than 2"):
        detect_problem_type(y_empty)
    with pytest.raises(ValueError, match="Less than 2"):
        detect_problem_type(y_one_value)
    with pytest.raises(ValueError, match="Less than 2"):
        detect_problem_type(y_nan)
    with pytest.raises(ValueError, match="Less than 2"):
        detect_problem_type(y_all_nan)


def test_detect_problem_type_binary():
    y_binary = pd.Series([1, 0, 1, 0, 0, 1])
    y_bool = pd.Series([True, False, True, True, True])
    y_float = pd.Series([1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    y_object = pd.Series(["yes", "no", "no", "yes"])
    y_categorical = pd.Series(["yes", "no", "no", "yes"], dtype="category")
    y_null = pd.Series([None, 0, 1, 1, 1])

    assert detect_problem_type(y_binary) == ProblemTypes.BINARY
    assert detect_problem_type(y_bool) == ProblemTypes.BINARY
    assert detect_problem_type(y_float) == ProblemTypes.BINARY
    assert detect_problem_type(y_object) == ProblemTypes.BINARY
    assert detect_problem_type(y_categorical) == ProblemTypes.BINARY
    assert detect_problem_type(y_null) == ProblemTypes.BINARY


def test_detect_problem_type_multiclass():
    y_multi = pd.Series([1, 2, 0, 2, 0, 0, 1])
    y_categorical = pd.Series(["yes", "no", "maybe", "no"], dtype="category")
    y_classes = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, np.nan] * 5)
    y_float = pd.Series([1.0, 2, 3])
    y_obj = pd.Series(["y", "n", "m"])
    y_neg = pd.Series([1, 2, -3])

    assert detect_problem_type(y_multi) == ProblemTypes.MULTICLASS
    assert detect_problem_type(y_categorical) == ProblemTypes.MULTICLASS
    assert detect_problem_type(y_classes) == ProblemTypes.MULTICLASS
    assert detect_problem_type(y_float) == ProblemTypes.MULTICLASS
    assert detect_problem_type(y_obj) == ProblemTypes.MULTICLASS
    assert detect_problem_type(y_neg) == ProblemTypes.MULTICLASS


def test_detect_problem_type_regression():
    y_values = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, np.nan])
    y_float = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11.0])

    assert detect_problem_type(y_values) == ProblemTypes.REGRESSION
    assert detect_problem_type(y_float) == ProblemTypes.REGRESSION


def test_numeric_extensions():
    y_Int64 = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype="Int64")
    y_Int64_null = pd.Series(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, None],
        dtype="Int64",
    )

    assert detect_problem_type(y_Int64) == ProblemTypes.REGRESSION
    assert detect_problem_type(y_Int64_null) == ProblemTypes.REGRESSION


def test_nan_none_na():
    y_none = pd.Series([None])
    y_pdna = pd.Series([pd.NA])
    y_nan = pd.Series([np.nan])
    y_all_null = pd.Series([None, np.nan, np.nan])

    with pytest.raises(ValueError, match="Less than 2"):
        detect_problem_type(y_none)
    with pytest.raises(ValueError, match="Less than 2"):
        detect_problem_type(y_pdna)
    with pytest.raises(ValueError, match="Less than 2"):
        detect_problem_type(y_nan)
    with pytest.raises(ValueError, match="Less than 2"):
        detect_problem_type(y_all_null)


def test_string_repr():
    assert ProblemTypes.BINARY.value == ProblemTypes.BINARY.__str__()
    assert ProblemTypes.MULTICLASS.value == ProblemTypes.MULTICLASS.__str__()
    assert ProblemTypes.REGRESSION.value == ProblemTypes.REGRESSION.__str__()


def test_all_problem_types():
    expected = [
        ProblemTypes.BINARY,
        ProblemTypes.MULTICLASS,
        ProblemTypes.REGRESSION,
        ProblemTypes.TIME_SERIES_REGRESSION,
        ProblemTypes.TIME_SERIES_BINARY,
        ProblemTypes.TIME_SERIES_MULTICLASS,
    ]
    assert ProblemTypes.all_problem_types == expected


@pytest.mark.parametrize("problem_type", ProblemTypes.all_problem_types)
def test_type_checks(problem_type):
    assert is_regression(problem_type) == (
        problem_type in [ProblemTypes.REGRESSION, ProblemTypes.TIME_SERIES_REGRESSION]
    )
    assert is_binary(problem_type) == (
        problem_type in [ProblemTypes.BINARY, ProblemTypes.TIME_SERIES_BINARY]
    )
    assert is_multiclass(problem_type) == (
        problem_type in [ProblemTypes.MULTICLASS, ProblemTypes.TIME_SERIES_MULTICLASS]
    )
    assert is_classification(problem_type) == (
        problem_type
        in [
            ProblemTypes.BINARY,
            ProblemTypes.MULTICLASS,
            ProblemTypes.TIME_SERIES_BINARY,
            ProblemTypes.TIME_SERIES_MULTICLASS,
        ]
    )
    assert is_time_series(problem_type) == (
        problem_type
        in [
            ProblemTypes.TIME_SERIES_BINARY,
            ProblemTypes.TIME_SERIES_MULTICLASS,
            ProblemTypes.TIME_SERIES_REGRESSION,
        ]
    )
