from .problem_types import ProblemTypes


def handle_problem_types(problem_types):
    if isinstance(problem_types, ProblemTypes):
        return problem_types
    elif isinstance(problem_types, str):
        try:
            return ProblemTypes[problem_types.upper()]
        except KeyError:
            raise KeyError('Problem type \'{}\' does not exist'.format(problem_types))
    elif isinstance(problem_types, list):
        types = list()
        for problem_type in problem_types:
            if isinstance(problem_type, ProblemTypes):
                types.append(problem_type)
            elif isinstance(problem_type, str):
                try:
                    types.append(ProblemTypes[problem_type.upper()])
                except KeyError:
                    raise KeyError('Problem type \'{}\' does not exist'.format(problem_type))
        return types
