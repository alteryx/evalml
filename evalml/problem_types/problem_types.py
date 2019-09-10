from enum import Enum


class ProblemTypes(Enum):
    """Enum for type of machine learning problem: BINARY, MULTICLASS, or REGRESSION"""
    BINARY = 'BINARY'
    MULTICLASS = 'MULTICLASS'
    REGRESSION = 'REGRESSION'
