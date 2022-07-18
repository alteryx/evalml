"""Utility methods for the ProblemTypes enum in EvalML."""
import pandas as pd
from pandas.api.types import is_numeric_dtype

from evalml.problem_types.problem_types import ProblemTypes


def handle_problem_types(problem_type):
    """Handles problem_type by either returning the ProblemTypes or converting from a str.

    Args:
        problem_type (str or ProblemTypes): Problem type that needs to be handled.

    Returns:
        ProblemTypes enum

    Raises:
        KeyError: If input is not a valid ProblemTypes enum value.
        ValueError: If input is not a string or ProblemTypes object.

    Examples:
        >>> assert handle_problem_types("regression") == ProblemTypes.REGRESSION
        >>> assert handle_problem_types("TIME SERIES BINARY") == ProblemTypes.TIME_SERIES_BINARY
        >>> assert handle_problem_types("Multiclass") == ProblemTypes.MULTICLASS
    """
    if isinstance(problem_type, str):
        try:
            tpe = ProblemTypes._all_values[problem_type.upper()]
        except KeyError:
            raise KeyError("Problem type '{}' does not exist".format(problem_type))
        return tpe
    if isinstance(problem_type, ProblemTypes):
        return problem_type
    raise ValueError(
        "`handle_problem_types` was not passed a str or ProblemTypes object",
    )


def detect_problem_type(y):
    """Determine the type of problem is being solved based on the targets (binary vs multiclass classification, regression). Ignores missing and null data.

    Args:
        y (pd.Series): The target labels to predict.

    Returns:
        ProblemType: ProblemType Enum

    Examples:
        >>> y = pd.Series([0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1])
        >>> assert detect_problem_type(y) == ProblemTypes.BINARY
        ...
        >>> y = pd.Series([1, 2, 3, 2, 1, 1, 1, 2, 2, 3, 3])
        >>> assert detect_problem_type(y) == ProblemTypes.MULTICLASS
        ...
        >>> y = pd.Series([1.6, 4.2, 3.3, 2.9, 4, 1, 5.5, 2, -2, -3.2, 3])
        >>> assert detect_problem_type(y) == ProblemTypes.REGRESSION

    Raises:
        ValueError: If the input has less than two classes.
    """
    y = pd.Series(y).dropna()
    num_classes = y.nunique()
    if num_classes < 2:
        raise ValueError("Less than 2 classes detected! Target unusable for modeling")
    if num_classes == 2:
        return ProblemTypes.BINARY
    if is_numeric_dtype(y.dtype):
        if num_classes > 10:
            return ProblemTypes.REGRESSION
    return ProblemTypes.MULTICLASS


def is_regression(problem_type):
    """Determines if the provided problem_type is a regression problem type.

    Args:
        problem_type (str or ProblemTypes): type of supervised learning problem. See evalml.problem_types.ProblemType.all_problem_types for a full list.

    Returns:
        bool: Whether or not the provided problem_type is a regression problem type.

    Examples:
        >>> assert is_regression("Regression")
        >>> assert is_regression(ProblemTypes.REGRESSION)
        >>> assert is_regression(ProblemTypes.TIME_SERIES_REGRESSION)
    """
    return handle_problem_types(problem_type) in [
        ProblemTypes.REGRESSION,
        ProblemTypes.TIME_SERIES_REGRESSION,
    ]


def is_binary(problem_type):
    """Determines if the provided problem_type is a binary classification problem type.

    Args:
        problem_type (str or ProblemTypes): type of supervised learning problem. See evalml.problem_types.ProblemType.all_problem_types for a full list.

    Returns:
        bool: Whether or not the provided problem_type is a binary classification problem type.

    Examples:
        >>> assert is_binary("Binary")
        >>> assert is_binary(ProblemTypes.BINARY)
        >>> assert is_binary(ProblemTypes.TIME_SERIES_BINARY)
    """
    return handle_problem_types(problem_type) in [
        ProblemTypes.BINARY,
        ProblemTypes.TIME_SERIES_BINARY,
    ]


def is_multiclass(problem_type):
    """Determines if the provided problem_type is a multiclass classification problem type.

    Args:
        problem_type (str or ProblemTypes): type of supervised learning problem. See evalml.problem_types.ProblemType.all_problem_types for a full list.

    Returns:
        bool: Whether or not the provided problem_type is a multiclass classification problem type.

    Examples:
        >>> assert is_multiclass("Multiclass")
        >>> assert is_multiclass(ProblemTypes.MULTICLASS)
        >>> assert is_multiclass(ProblemTypes.TIME_SERIES_MULTICLASS)
    """
    return handle_problem_types(problem_type) in [
        ProblemTypes.MULTICLASS,
        ProblemTypes.TIME_SERIES_MULTICLASS,
    ]


def is_classification(problem_type):
    """Determines if the provided problem_type is a classification problem type.

    Args:
        problem_type (str or ProblemTypes): type of supervised learning problem. See evalml.problem_types.ProblemType.all_problem_types for a full list.

    Returns:
        bool: Whether or not the provided problem_type is a classification problem type.

    Examples:
        >>> assert is_classification("Multiclass")
        >>> assert is_classification(ProblemTypes.TIME_SERIES_BINARY)
        >>> assert not is_classification(ProblemTypes.REGRESSION)
    """
    return is_binary(problem_type) or is_multiclass(problem_type)


def is_time_series(problem_type):
    """Determines if the provided problem_type is a time series problem type.

    Args:
        problem_type (str or ProblemTypes): type of supervised learning problem. See evalml.problem_types.ProblemType.all_problem_types for a full list.

    Returns:
        bool: Whether or not the provided problem_type is a time series problem type.

    Examples:
        >>> assert is_time_series("time series regression")
        >>> assert is_time_series(ProblemTypes.TIME_SERIES_BINARY)
        >>> assert not is_time_series(ProblemTypes.REGRESSION)
    """
    return handle_problem_types(problem_type) in [
        ProblemTypes.TIME_SERIES_BINARY,
        ProblemTypes.TIME_SERIES_MULTICLASS,
        ProblemTypes.TIME_SERIES_REGRESSION,
    ]
