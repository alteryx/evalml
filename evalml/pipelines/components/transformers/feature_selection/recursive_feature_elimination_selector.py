"""Components that select top features based on recursive feature elimination with a Random Forest model."""
from abc import abstractmethod

from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier
from sklearn.ensemble import RandomForestRegressor as SKRandomForestRegressor
from sklearn.feature_selection import RFECV
from skopt.space import Integer

from evalml.pipelines.components.transformers.feature_selection.feature_selector import (
    FeatureSelector,
)


class RecursiveFeatureEliminationSelector(FeatureSelector):
    """Selects relevant features using recursive feature elimination."""

    hyperparameter_ranges = {
        "perc": Integer(0, 100),
    }
    """{
        "percent_features": Real(0.01, 1),
        "threshold": ["mean", "median"],
    }"""

    def __init__(
        self,
        step=5,
        min_features_to_select=10,
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
    """Selects relevant features using recursive feature elimination with a Random Forest Classifier."""

    name = "RFE Selector with RF Classifier"

    def _get_estimator(self, random_seed, n_estimators, max_depth, n_jobs):
        return SKRandomForestClassifier(
            random_state=random_seed,
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
        )


class RFRegressorRFESelector(RecursiveFeatureEliminationSelector):
    """Selects relevant features using recursive feature elimination with a Random Forest Regressor."""

    name = "RFE Selector with RF Regressor"

    def _get_estimator(self, random_seed, n_estimators, max_depth, n_jobs):
        return SKRandomForestRegressor(
            random_state=random_seed,
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
        )
