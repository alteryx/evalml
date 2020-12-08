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
