"""Random Forest Regressor."""
from typing import Dict, List, Optional, Union

import pandas as pd
from sklearn.ensemble import RandomForestRegressor as SKRandomForestRegressor
from skopt.space import Integer

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.pipelines.components.utils import (
    get_prediction_intevals_for_tree_regressors,
)
from evalml.problem_types import ProblemTypes


class RandomForestRegressor(Estimator):
    """Random Forest Regressor.

    Args:
        n_estimators (float): The number of trees in the forest. Defaults to 100.
        max_depth (int): Maximum tree depth for base learners. Defaults to 6.
        n_jobs (int or None): Number of jobs to run in parallel. -1 uses all processes. Defaults to -1.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Random Forest Regressor"
    hyperparameter_ranges = {
        "n_estimators": Integer(10, 1000),
        "max_depth": Integer(1, 32),
    }
    """{
        "n_estimators": Integer(10, 1000),
        "max_depth": Integer(1, 32),
    }"""
    model_family = ModelFamily.RANDOM_FOREST
    """ModelFamily.RANDOM_FOREST"""
    supported_problem_types = [
        ProblemTypes.REGRESSION,
        ProblemTypes.TIME_SERIES_REGRESSION,
    ]
    """[
        ProblemTypes.REGRESSION,
        ProblemTypes.TIME_SERIES_REGRESSION,
    ]"""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        n_jobs: int = -1,
        random_seed: Union[int, float] = 0,
        **kwargs,
    ):
        parameters = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "n_jobs": n_jobs,
        }
        parameters.update(kwargs)

        rf_regressor = SKRandomForestRegressor(random_state=random_seed, **parameters)
        super().__init__(
            parameters=parameters,
            component_obj=rf_regressor,
            random_seed=random_seed,
        )

    def get_prediction_intervals(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        coverage: List[float] = None,
        predictions: pd.Series = None,
    ) -> Dict[str, pd.Series]:
        """Find the prediction intervals using the fitted RandomForestRegressor.

        Args:
            X (pd.DataFrame): Data of shape [n_samples, n_features].
            y (pd.Series): Target data. Optional.
            coverage (list[float]): A list of floats between the values 0 and 1 that the upper and lower bounds of the
                prediction interval should be calculated for.
            predictions (pd.Series): Optional list of predictions to use. If None, will generate predictions using `X`.

        Returns:
            dict: Prediction intervals, keys are in the format {coverage}_lower or {coverage}_upper.
        """
        if coverage is None:
            coverage = [0.95]
        X, _ = self._manage_woodwork(X, y)
        X = X.ww.select(exclude="Datetime")

        if predictions is None:
            predictions = self._component_obj.predict(X)
        estimators = self._component_obj.estimators_
        return get_prediction_intevals_for_tree_regressors(
            X,
            predictions,
            coverage,
            estimators,
        )
