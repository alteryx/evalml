"""Linear Regressor."""
from sklearn.linear_model import LinearRegression as SKLinearRegression

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes


class LinearRegressor(Estimator):
    """Linear Regressor.

    Args:
        fit_intercept (boolean): Whether to calculate the intercept for this model.
            If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
            Defaults to True.
        normalize (boolean): If True, the regressors will be normalized before regression
            by subtracting the mean and dividing by the l2-norm.
            This parameter is ignored when fit_intercept is set to False. Defaults to False.
        n_jobs (int or None): Number of jobs to run in parallel. -1 uses all threads. Defaults to -1.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Linear Regressor"
    hyperparameter_ranges = {"fit_intercept": [True, False], "normalize": [True, False]}
    """{
        "fit_intercept": [True, False],
        "normalize": [True, False]
    }"""
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
        self, fit_intercept=True, normalize=False, n_jobs=-1, random_seed=0, **kwargs
    ):
        parameters = {
            "fit_intercept": fit_intercept,
            "normalize": normalize,
            "n_jobs": n_jobs,
        }
        parameters.update(kwargs)
        linear_regressor = SKLinearRegression(**parameters)
        super().__init__(
            parameters=parameters,
            component_obj=linear_regressor,
            random_seed=random_seed,
        )

    @property
    def feature_importance(self):
        """Feature importance for fitted linear regressor."""
        return self._component_obj.coef_
