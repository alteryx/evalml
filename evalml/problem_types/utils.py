from .problem_types import ProblemTypes
import pandas as pd


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

    Arguments:
        y (pd.Series): the target labels to predict

    Returns:
        String: string representation for the problem type
    """
    y = pd.Series(y)
    num_classes = y.nunique()
    if num_classes < 2:
        raise ValueError("Less than 2 classes detected!")
    elif num_classes == 2:
        return "binary"
    else:
        if pd.api.types.is_float_dtype(y):
            y2 = y.copy().astype('int64')
            if all(y == y2):
                # if all floats are equivalent to their int counterpart (ie 1.0 == 1)
                return 'multiclass'
            return 'regression'
        return 'multiclass'