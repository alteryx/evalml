from enum import Enum


class ModelTypes(Enum):
    """Enum for type of machine learning models."""
    RANDOM_FOREST = 'random_forest'
    XGBOOST = 'xgboost'
    LINEAR_MODEL = 'linear_model'
    CATBOOST = 'catboost'

    def __str__(self):
        model_type_dict = {ModelTypes.RANDOM_FOREST.name: "Random Forest",
                           ModelTypes.XGBOOST.name: "XGBoost Classifier",
                           ModelTypes.LINEAR_MODEL.name: "Linear Model",
                           ModelTypes.CATBOOST.name: "CatBoost Classifier"}
        return model_type_dict[self.name]
