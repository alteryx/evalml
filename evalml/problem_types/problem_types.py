from enum import Enum


class ProblemTypes(Enum):
    """Enum for type of machine learning problem: BINARY, MULTICLASS, or REGRESSION"""
    BINARY = 'binary'
    MULTICLASS = 'multiclass'
    REGRESSION = 'regression'
