from enum import Enum


class ModelTypes(Enum):
    """Enum for type of machine learning models."""
    RANDOM_FOREST = 'random_forest'
    XGBOOST = 'xgboost'
    LINEAR_MODEL = 'linear_model'
