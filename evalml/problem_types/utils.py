import pandas as pd
from pandas.api.types import is_numeric_dtype

from .problem_types import ProblemTypes


def handle_problem_types(problem_type):
    """Handles problem_type by either returning the ProblemTypes or converting from a str.

    Arguments:
        problem_type (str or ProblemTypes): Problem type that needs to be handled

    Returns:
        ProblemTypes
    """
    if isinstance(problem_type, str):
        try:
            tpe = ProblemTypes._all_values[problem_type.upper()]
        except KeyError:
            raise KeyError('Problem type \'{}\' does not exist'.format(problem_type))
        return tpe
    if isinstance(problem_type, ProblemTypes):
        return problem_type
    raise ValueError('`handle_problem_types` was not passed a str or ProblemTypes object')


def detect_problem_type(y):
    """Determine the type of problem is being solved based on the targets (binary vs multiclass classification, regression)
        Ignores missing and null data

    Arguments:
        y (pd.Series): the target labels to predict

    Returns:
        ProblemType: ProblemType Enum

    Example:
        >>> y = pd.Series([0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1])
        >>> problem_type = detect_problem_type(y)
        >>> assert problem_type == ProblemTypes.BINARY
    """
    y = pd.Series(y).dropna()
    num_classes = y.nunique()
    if num_classes < 2:
        raise ValueError("Less than 2 classes detected! Target unusable for modeling")
    if num_classes == 2:
        return ProblemTypes.BINARY
    if is_numeric_dtype(y.dtype):
        if (num_classes > 10):
            return ProblemTypes.REGRESSION
    return ProblemTypes.MULTICLASS


def is_regression(problem_type):
    """Determines if the provided problem_type is a regression problem type

    Arguments:
        problem_type (str or ProblemTypes): type of supervised learning problem. See evalml.problem_types.ProblemType.all_problem_types for a full list.

    Returns:
        bool: Whether or not the provided problem_type is a regression problem type.
"""
    return handle_problem_types(problem_type) in [ProblemTypes.REGRESSION, ProblemTypes.TIME_SERIES_REGRESSION]


def is_binary(problem_type):
    """Determines if the provided problem_type is a binary classification problem type

    Arguments:
        problem_type (str or ProblemTypes): type of supervised learning problem. See evalml.problem_types.ProblemType.all_problem_types for a full list.

    Returns:
        bool: Whether or not the provided problem_type is a binary classification problem type.
"""
    return handle_problem_types(problem_type) in [ProblemTypes.BINARY, ProblemTypes.TIME_SERIES_BINARY]


def is_multiclass(problem_type):
    """Determines if the provided problem_type is a multiclass classification problem type

    Arguments:
        problem_type (str or ProblemTypes): type of supervised learning problem. See evalml.problem_types.ProblemType.all_problem_types for a full list.

    Returns:
        bool: Whether or not the provided problem_type is a multiclass classification problem type.
"""
    return handle_problem_types(problem_type) in [ProblemTypes.MULTICLASS, ProblemTypes.TIME_SERIES_MULTICLASS]


def is_classification(problem_type):
    """Determines if the provided problem_type is a classification problem type

    Arguments:
        problem_type (str or ProblemTypes): type of supervised learning problem. See evalml.problem_types.ProblemType.all_problem_types for a full list.

    Returns:
        bool: Whether or not the provided problem_type is a classification problem type.
"""
    return is_binary(problem_type) or is_multiclass(problem_type)


def is_time_series(problem_type):
    """Determines if the provided problem_type is a time series problem type

    Arguments:
        problem_type (str or ProblemTypes): type of supervised learning problem. See evalml.problem_types.ProblemType.all_problem_types for a full list.

    Returns:
        bool: Whether or not the provided problem_type is a time series problem type.
"""
    return handle_problem_types(problem_type) in [ProblemTypes.TIME_SERIES_BINARY,
                                                  ProblemTypes.TIME_SERIES_MULTICLASS,
                                                  ProblemTypes.TIME_SERIES_REGRESSION]
