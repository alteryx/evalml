import pandas as pd

from .problem_types import ProblemTypes


def handle_problem_types(problem_type):
    """Handles problem_type by either returning the ProblemTypes or converting from a str.

    Args:
        problem_type (str or ProblemTypes): problem type that needs to be handled

    Returns:
        ProblemTypes
    """
    if isinstance(problem_type, str):
        try:
            tpe = ProblemTypes[problem_type.upper()]
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
        String: string representation for the problem type

    Example:
        >>> y = pd.Series([0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1])
        >>> problem_type = detect_problem_type(y)
        >>> assert problem_type == 'binary'
    """
    y = pd.Series(y).dropna()
    num_classes = y.nunique()
    if num_classes < 2:
        raise ValueError("Less than 2 classes detected!")
    elif num_classes == 2:
        return "binary"
    else:
        if pd.api.types.is_numeric_dtype(y):
            if (num_classes > 10):
                return 'regression'
        return 'multiclass'
