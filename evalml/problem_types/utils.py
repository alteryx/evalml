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
