"""Random Forest Regressor."""
from sklearn.ensemble import RandomForestRegressor as SKRandomForestRegressor
from skopt.space import Integer

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
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
        self, n_estimators=100, max_depth=6, n_jobs=-1, random_seed=0, **kwargs
    ):
        parameters = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "n_jobs": n_jobs,
        }
        parameters.update(kwargs)

        rf_regressor = SKRandomForestRegressor(random_state=random_seed, **parameters)
        super().__init__(
            parameters=parameters, component_obj=rf_regressor, random_seed=random_seed
        )
