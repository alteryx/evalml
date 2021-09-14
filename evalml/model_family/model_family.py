"""Enum for family of machine learning models."""
from enum import Enum


class ModelFamily(Enum):
    """Enum for family of machine learning models."""

    K_NEIGHBORS = "k_neighbors"
    """K Nearest Neighbors model family."""

    RANDOM_FOREST = "random_forest"
    """Random Forest model family."""

    SVM = "svm"
    """SVM model family."""

    XGBOOST = "xgboost"
    """XGBoost model family."""

    LIGHTGBM = "lightgbm"
    """LightGBM model family."""

    LINEAR_MODEL = "linear_model"
    """Linear model family."""

    CATBOOST = "catboost"
    """CatBoost model family."""

    EXTRA_TREES = "extra_trees"
    """Extra Trees model family."""

    ENSEMBLE = "ensemble"
    """Ensemble model family."""

    DECISION_TREE = "decision_tree"
    """Decision Tree model family."""

    ARIMA = "arima"
    """ARIMA model family."""

    BASELINE = "baseline"
    """Baseline model family."""

    PROPHET = "prophet"
    """Prophet model family."""

    NONE = "none"
    """None"""

    def __str__(self):
        """String representation of a ModelFamily enum."""
        model_family_dict = {
            ModelFamily.K_NEIGHBORS.name: "K Nearest Neighbors",
            ModelFamily.RANDOM_FOREST.name: "Random Forest",
            ModelFamily.SVM.name: "SVM",
            ModelFamily.XGBOOST.name: "XGBoost",
            ModelFamily.LIGHTGBM.name: "LightGBM",
            ModelFamily.LINEAR_MODEL.name: "Linear",
            ModelFamily.CATBOOST.name: "CatBoost",
            ModelFamily.EXTRA_TREES.name: "Extra Trees",
            ModelFamily.DECISION_TREE.name: "Decision Tree",
            ModelFamily.BASELINE.name: "Baseline",
            ModelFamily.ENSEMBLE.name: "Ensemble",
            ModelFamily.ARIMA.name: "ARIMA",
            ModelFamily.PROPHET.name: "Prophet",
            ModelFamily.NONE.name: "None",
        }

        return model_family_dict[self.name]

    def __repr__(self):
        """String representation of a ModelFamily enum."""
        return "ModelFamily." + self.name

    def is_tree_estimator(self):
        """Checks whether the estimator's model family uses trees."""
        tree_estimators = {
            self.CATBOOST,
            self.EXTRA_TREES,
            self.RANDOM_FOREST,
            self.DECISION_TREE,
            self.XGBOOST,
            self.LIGHTGBM,
        }
        return self in tree_estimators
