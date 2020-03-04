from enum import Enum


class ModelFamily(Enum):
    """Enum for family of machine learning models."""
    RANDOM_FOREST = 'random_forest'
    XGBOOST = 'xgboost'
    LINEAR_MODEL = 'linear_model'
    CATBOOST = 'catboost'

    def __str__(self):
        model_family_dict = {ModelFamily.RANDOM_FOREST.name: "Random Forest",
                           ModelFamily.XGBOOST.name: "XGBoost Classifier",
                           ModelFamily.LINEAR_MODEL.name: "Linear Model",
                           ModelFamily.CATBOOST.name: "CatBoost Classifier"}
        return model_family_dict[self.name]
