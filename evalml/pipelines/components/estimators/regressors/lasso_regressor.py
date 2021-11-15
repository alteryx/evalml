"""Ridge Regressor."""
from sklearn.linear_model import Lasso as SKLassoRegression

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes


class LassoRegressor(Estimator):
    """Linear Regressor.

    Args:
        solver ({"newton-cg", "lbfgs", "liblinear", "sag", "saga"}): Algorithm to use in the optimization problem.
            For small datasets, "liblinear" is a good choice, whereas "sag" and "saga" are faster for large ones.
            For multiclass problems, only "newton-cg", "sag", "saga" and "lbfgs" handle multinomial loss; "liblinear" is limited to one-versus-rest schemes.

            - "newton-cg", "lbfgs", "sag" and "saga" handle L2 or no penalty
            - "liblinear" and "saga" also handle L1 penalty
            - "saga" also supports "elasticnet" penalty
            - "liblinear" does not support setting penalty='none'

            Defaults to "lbfgs".

        n_jobs (int or None): Number of jobs to run in parallel. -1 uses all threads. Defaults to -1.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Ridge Regressor"
    hyperparameter_ranges = {}
    model_family = ModelFamily.LINEAR_MODEL
    """ModelFamily.LINEAR_MODEL"""
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
        n_jobs=-1,
        random_seed=0,
        **kwargs,
    ):
        parameters = {
        }
        parameters.update(kwargs)
        linear_regressor = SKLassoRegression(**parameters)
        super().__init__(
            parameters=parameters,
            component_obj=linear_regressor,
            random_seed=random_seed,
        )

    @property
    def feature_importance(self):
        """Feature importance for fitted linear regressor."""
        return self._component_obj.coef_
