from .problem_types import ProblemTypes


def handle_problem_types(problem_type):
    """Handles problem_type by either returning the ProblemTypes or converting to a str

    Args:
        problem_types (str/ProblemTypes) : problem type that needs to be handled

    Returns:
        ProblemType
    """

    if isinstance(problem_type, str):
        try:
            tp = ProblemTypes[problem_type.upper()]
        except KeyError:
            raise KeyError('Problem type \'{}\' does not exist'.format(problem_type))
        return tp
    if isinstance(problem_type, ProblemTypes):
        return problem_type
    assert ValueError('`handle_problem_types` was not passed a str or ProblemTypes object')