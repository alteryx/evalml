from enum import Enum


class ProblemTypes(Enum):
    """Enum for type of machine learning problem: BINARY, MULTICLASS, or REGRESSION."""
    BINARY = 'binary'
    MULTICLASS = 'multiclass'
    REGRESSION = 'regression'

    def __str__(self):
        problem_type_dict = {ProblemTypes.BINARY.name: "binary",
                             ProblemTypes.MULTICLASS.name: "multiclass",
                             ProblemTypes.REGRESSION.name: "regression"}
        return problem_type_dict[self.name]
