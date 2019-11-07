from enum import Enum


class ProblemTypes(Enum):
    """Enum for type of machine learning problem: BINARY, MULTICLASS, or REGRESSION"""
    BINARY = 'binary'
    MULTICLASS = 'multiclass'
    REGRESSION = 'regression'

    def __str__(self):
        if self.value in [ProblemTypes.BINARY.value, ProblemTypes.MULTICLASS.value]:
            return "{} Classifier".format(self.value.title())
        return self.value.title()
