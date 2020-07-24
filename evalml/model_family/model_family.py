from enum import Enum


class ModelFamily(Enum):
    """Enum for family of machine learning models."""
    RANDOM_FOREST = 'random_forest'
    XGBOOST = 'xgboost'
    LINEAR_MODEL = 'linear_model'
    CATBOOST = 'catboost'
    EXTRA_TREES = 'extra_trees'
    BASELINE = 'baseline'
    NONE = 'none'

    def __str__(self):
        model_family_dict = {ModelFamily.RANDOM_FOREST.name: "Random Forest",
                             ModelFamily.XGBOOST.name: "XGBoost",
                             ModelFamily.LINEAR_MODEL.name: "Linear",
                             ModelFamily.CATBOOST.name: "CatBoost",
                             ModelFamily.EXTRA_TREES.name: "Extra Trees",
                             ModelFamily.BASELINE.name: "Baseline",
                             ModelFamily.NONE.name: "None"}
        return model_family_dict[self.name]

    def is_tree_estimator(self):
        """Checks whether the estimator's model family uses tree ensembles."""
        tree_estimators = {self.CATBOOST, self.EXTRA_TREES, self.RANDOM_FOREST,
                           self.XGBOOST}
        return self in tree_estimators
