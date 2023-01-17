from abc import abstractmethod

from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier
from sklearn.ensemble import RandomForestRegressor as SKRandomForestRegressor
from sklearn.feature_selection import RFECV
from skopt.space import Integer

from evalml.pipelines.components.transformers.feature_selection.feature_selector import (
    FeatureSelector,
)


class RecursiveFeatureEliminationSelector(FeatureSelector):
    """Selects relevant features using RFE

    Args:
        estimator (Estimator): The maximum number of features to select.
            If both percent_features and number_features are specified, take the greater number of features. Defaults to 0.5.
            Defaults to None.
        n_estimators (int or string): The number of estimators.
        perc (int): percentile used as our threshold for comparison between shadow and real features
        alpha (float): Level at which the corrected p-values will get rejected in both
            correction steps.
        two_step (boolean): if False, will use Bonferroni correction only
        max_iter (int): maximum number of iterations to perform
        n_jobs (int or None): Number of jobs to run in parallel. -1 uses all processes. Defaults to -1.
        random_seed (int): Seed for the random number generator. Defaults to 0.
        early_stopping (boolean): whether to use early stopping
        n_iter_no_change (int): Maximum number of iterations to do without confirming a tentative feature
    """

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
        """Return estimator with supplied parameters"""


class RFClassifierRFESelector(RecursiveFeatureEliminationSelector):
    name = "RFE Selector with RF Classifier"

    def _get_estimator(self, random_seed, n_estimators, max_depth, n_jobs):
        return SKRandomForestClassifier(
            random_state=random_seed,
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
        )


class RFRegressorRFESelector(RecursiveFeatureEliminationSelector):
    name = "RFE Selector with RF Regressor"

    def _get_estimator(self, random_seed, n_estimators, max_depth, n_jobs):
        return SKRandomForestRegressor(
            random_state=random_seed,
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
        )
