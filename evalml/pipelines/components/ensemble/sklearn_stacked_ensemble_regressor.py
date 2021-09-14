"""Scikit-learn Stacked Ensemble Regressor."""
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import KFold

from evalml.model_family import ModelFamily
from evalml.pipelines.components import LinearRegressor
from evalml.pipelines.components.ensemble import SklearnStackedEnsembleBase
from evalml.problem_types import ProblemTypes


class SklearnStackedEnsembleRegressor(SklearnStackedEnsembleBase):
    """Scikit-learn Stacked Ensemble Regressor.

    Args:
        input_pipelines (list(PipelineBase or subclass obj)): List of pipeline instances to use as the base estimators.
            This must not be None or an empty list or else EnsembleMissingPipelinesError will be raised.
        final_estimator (Estimator or subclass): The regressor used to combine the base estimators. If None, uses LinearRegressor.
        cv (int, cross-validation generator or an iterable): Determines the cross-validation splitting strategy used to train final_estimator.
            For int/None inputs, KFold is used. Defaults to None.
            Possible inputs for cv are:

            - None: 3-fold cross validation
            - int: the number of folds in a (Stratified) KFold
            - An scikit-learn cross-validation generator object
            - An iterable yielding (train, test) splits
        n_jobs (int or None): Non-negative integer describing level of parallelism used for pipelines.
            None and 1 are equivalent. If set to -1, all CPUs are used. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.
            Defaults to -1.
            - Note: there could be some multi-process errors thrown for values of `n_jobs != 1`. If this is the case, please use `n_jobs = 1`.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Sklearn Stacked Ensemble Regressor"
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
    _stacking_estimator_class = StackingRegressor
    _default_final_estimator = LinearRegressor
    _default_cv = KFold
