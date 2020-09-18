import numpy as np
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
        Treats np.nan as a separate value

    Arguments:
        y (pd.Series): the target labels to predict

    Returns:
        String: string representation for the problem type
    """
    y = pd.Series(y)
    y2 = y.copy().dropna()
    y = y.fillna(np.nan)
    num_classes = y.nunique(dropna=False)
    if num_classes < 2:
        raise ValueError("Less than 2 classes detected!")
    elif num_classes == 2:
        return "binary"
    else:
        if pd.api.types.is_numeric_dtype(y):
            # if we have too many classes or too many classes per length of data
            print(y2.dtype)
            if (num_classes >= 10) or (num_classes > (0.5 * len(y))):
                print(len(y), num_classes)
                return 'regression'
            # else if remaining values are ints, is multiclass
            elif pd.api.types.is_float_dtype(y2):
                print(y2.values)
                if pd.api.types.is_integer_dtype(pd.Series(y2.values)):
                    return 'multiclass'
                return 'regression'
        return 'multiclass'
