"""Components that select top features based on recursive feature elimination with a Random Forest model."""
from abc import abstractmethod

from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier
from sklearn.ensemble import RandomForestRegressor as SKRandomForestRegressor
from sklearn.feature_selection import RFECV
from skopt.space import Real

from evalml.pipelines.components.transformers.feature_selection.feature_selector import (
    FeatureSelector,
)


class RecursiveFeatureEliminationSelector(FeatureSelector):
    """Selects relevant features using recursive feature elimination."""

    hyperparameter_ranges = {
        "step": Real(0.05, 0.25),
    }
    """{
        "step": Real(0.05, 0.25)
    }"""

    def __init__(
        self,
        step=0.2,
        min_features_to_select=1,
        cv=None,
        scoring=None,
        n_jobs=-1,
        n_estimators=10,
        max_depth=None,
        random_seed=0,
        **kwargs,
    ):
        parameters = {
            "step": step,
            "min_features_to_select": min_features_to_select,
            "cv": cv,
            "scoring": scoring,
            "n_jobs": n_jobs,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
        }
        parameters.update(kwargs)
        estimator = self._get_estimator(
            random_seed=random_seed,
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
        )

        feature_selection = RFECV(
            estimator=estimator,
            step=step,
            min_features_to_select=min_features_to_select,
            cv=cv,
            scoring=scoring,
            **kwargs,
        )
        super().__init__(
            parameters=parameters,
            component_obj=feature_selection,
            random_seed=random_seed,
        )

    @abstractmethod
    def _get_estimator(self, random_seed, n_estimators, max_depth, n_jobs):
        """Return estimator with supplied parameters."""


class RFClassifierRFESelector(RecursiveFeatureEliminationSelector):
    """Selects relevant features using recursive feature elimination with a Random Forest Classifier.

    Args:
        step (int, float): The number of features to eliminate in each iteration. If an integer is specified
            this will represent the number of features to eliminate. If a float is specified this represents
            the percentage of features to eliminate each iteration. The last iteration may drop fewer than this
            number of features in order to satisfy the min_features_to_select constraint. Defaults to 0.2.
        min_features_to_select (int): The minimum number of features to return. Defaults to 1.
        cv (int or None): Number of folds to use for the cross-validation splitting strategy. Defaults to None
            which will use 5 folds.
        scoring (str, callable or None): A string or scorer callable object to specify the scoring method.
        n_jobs (int or None): Number of jobs to run in parallel. -1 uses all processes. Defaults to -1.
        n_estimators (int): The number of trees in the forest. Defaults to 10.
        max_depth (int): Maximum tree depth for base learners. Defaults to None.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "RFE Selector with RF Classifier"

    def _get_estimator(self, random_seed, n_estimators, max_depth, n_jobs):
        return SKRandomForestClassifier(
            random_state=random_seed,
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
        )


class RFRegressorRFESelector(RecursiveFeatureEliminationSelector):
    """Selects relevant features using recursive feature elimination with a Random Forest Regressor.

    Args:
        step (int, float): The number of features to eliminate in each iteration. If an integer is specified
            this will represent the number of features to eliminate. If a float is specified this represents
            the percentage of features to eliminate each iteration. The last iteration may drop fewer than this
            number of features in order to satisfy the min_features_to_select constraint. Defaults to 0.2.
        min_features_to_select (int): The minimum number of features to return. Defaults to 1.
        cv (int or None): Number of folds to use for the cross-validation splitting strategy. Defaults to None
            which will use 5 folds.
        scoring (str, callable or None): A string or scorer callable object to specify the scoring method.
        n_jobs (int or None): Number of jobs to run in parallel. -1 uses all processes. Defaults to -1.
        n_estimators (int): The number of trees in the forest. Defaults to 10.
        max_depth (int): Maximum tree depth for base learners. Defaults to None.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "RFE Selector with RF Regressor"

    def _get_estimator(self, random_seed, n_estimators, max_depth, n_jobs):
        return SKRandomForestRegressor(
            random_state=random_seed,
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
        )
