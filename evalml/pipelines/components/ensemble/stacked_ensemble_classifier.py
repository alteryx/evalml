"""Stacked Ensemble Classifier."""
from evalml.model_family import ModelFamily
from evalml.pipelines.components import ElasticNetClassifier
from evalml.pipelines.components.ensemble import StackedEnsembleBase
from evalml.problem_types import ProblemTypes


class StackedEnsembleClassifier(StackedEnsembleBase):
    """Stacked Ensemble Classifier.

    Arguments:
        final_estimator (Estimator or subclass): The classifier used to combine the base estimators. If None, uses ElasticNetClassifier.
        n_jobs (int or None): Integer describing level of parallelism used for pipelines. None and 1 are equivalent.
            If set to -1, all CPUs are used. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Defaults to -1.
            - Note: there could be some multi-process errors thrown for values of `n_jobs != 1`. If this is the case, please use `n_jobs = 1`.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Stacked Ensemble Classifier"
    model_family = ModelFamily.ENSEMBLE
    """ModelFamily.ENSEMBLE"""
    supported_problem_types = [
        ProblemTypes.BINARY,
        ProblemTypes.MULTICLASS,
        ProblemTypes.TIME_SERIES_BINARY,
        ProblemTypes.TIME_SERIES_MULTICLASS,
    ]
    """[
        ProblemTypes.BINARY,
        ProblemTypes.MULTICLASS,
        ProblemTypes.TIME_SERIES_BINARY,
        ProblemTypes.TIME_SERIES_MULTICLASS,
    ]"""
    hyperparameter_ranges = {}
    """{}"""
    _default_final_estimator = ElasticNetClassifier
