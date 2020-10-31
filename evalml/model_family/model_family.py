from enum import Enum


class ModelFamily(Enum):
    """
    Enum for family of machine learning models.
    .. data:: RANDOM_FOREST

        Indicates some unknown error.

    .. data:: XGBOOST

        Indicates that the request was bad in some way.

    .. data:: LINEAR_MODEL

    """

    """Random Forest."""
    RANDOM_FOREST = 'random_forest'
    """XGBoost"""
    XGBOOST = 'xgboost'
    LIGHTGBM = 'lightgbm'
    LINEAR_MODEL = 'linear_model'
    CATBOOST = 'catboost'
    EXTRA_TREES = 'extra_trees'
    ENSEMBLE = 'ensemble',
    DECISION_TREE = 'decision_tree'
    BASELINE = 'baseline'
    NONE = 'none'

    def __str__(self):
        model_family_dict = {ModelFamily.RANDOM_FOREST.name: "Random Forest",
                             ModelFamily.XGBOOST.name: "XGBoost",
                             ModelFamily.LIGHTGBM.name: "LightGBM",
                             ModelFamily.LINEAR_MODEL.name: "Linear",
                             ModelFamily.CATBOOST.name: "CatBoost",
                             ModelFamily.EXTRA_TREES.name: "Extra Trees",
                             ModelFamily.DECISION_TREE.name: "Decision Tree",
                             ModelFamily.BASELINE.name: "Baseline",
                             ModelFamily.ENSEMBLE.name: "Ensemble",
                             ModelFamily.NONE.name: "None"}
        return model_family_dict[self.name]

    def __repr__(self):
        return "ModelFamily." + self.name

    def is_tree_estimator(self):
        """Checks whether the estimator's model family uses trees."""
        tree_estimators = {self.CATBOOST, self.EXTRA_TREES, self.RANDOM_FOREST,
                           self.DECISION_TREE, self.XGBOOST, self.LIGHTGBM}
        return self in tree_estimators
