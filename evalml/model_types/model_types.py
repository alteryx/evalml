from enum import Enum


class ModelTypes(Enum):
    """Enum for type of machine learning models."""
    RANDOM_FOREST = 'random_forest'
    XGBOOST = 'xgboost'
    LINEAR_MODEL = 'linear_model'

    def __str__(self):
        if self.value in [ModelTypes.XGBOOST.value]:
            return "XGBoost Classifier"
        return self.value.replace(" ", "_").title()
