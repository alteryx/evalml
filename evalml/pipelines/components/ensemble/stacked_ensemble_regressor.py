"""Stacked Ensemble Regressor."""
from evalml.model_family import ModelFamily
from evalml.pipelines.components import ElasticNetRegressor
from evalml.pipelines.components.ensemble import StackedEnsembleBase
from evalml.problem_types import ProblemTypes


class StackedEnsembleRegressor(StackedEnsembleBase):
    """Stacked Ensemble Regressor.

    Arguments:
        final_estimator (Estimator or subclass): The regressor used to combine the base estimators. If None, uses ElasticNetRegressor.
        n_jobs (int or None): Integer describing level of parallelism used for pipelines. None and 1 are equivalent.
            If set to -1, all CPUs are used. For n_jobs greater than -1, (n_cpus + 1 + n_jobs) are used. Defaults to -1.
            - Note: there could be some multi-process errors thrown for values of `n_jobs != 1`. If this is the case, please use `n_jobs = 1`.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Stacked Ensemble Regressor"
    model_family = ModelFamily.ENSEMBLE
    """ModelFamily.ENSEMBLE"""
    supported_problem_types = [
        ProblemTypes.REGRESSION,
        ProblemTypes.TIME_SERIES_REGRESSION,
    ]
    """[
        ProblemTypes.REGRESSION,
        ProblemTypes.TIME_SERIES_REGRESSION,
    ]"""
    hyperparameter_ranges = {}
    """{}"""
    _default_final_estimator = ElasticNetRegressor
