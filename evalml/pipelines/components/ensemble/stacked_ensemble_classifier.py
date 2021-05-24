from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedKFold

from evalml.model_family import ModelFamily
from evalml.pipelines.components import LogisticRegressionClassifier
from evalml.pipelines.components.ensemble import StackedEnsembleBase
from evalml.problem_types import ProblemTypes


class StackedEnsembleClassifier(StackedEnsembleBase):
    """Stacked Ensemble Classifier."""
    name = "Stacked Ensemble Classifier"
    model_family = ModelFamily.ENSEMBLE
    supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS, ProblemTypes.TIME_SERIES_BINARY,
                               ProblemTypes.TIME_SERIES_MULTICLASS]
    hyperparameter_ranges = {}
    _stacking_estimator_class = StackingClassifier
    _default_final_estimator = LogisticRegressionClassifier
    _default_cv = StratifiedKFold

    def __init__(self, input_pipelines=None, final_estimator=None,
                 cv=None, n_jobs=-1, random_seed=0, **kwargs):
        """Stacked ensemble classifier.

        Arguments:
            input_pipelines (list(PipelineBase or subclass obj)): List of pipeline instances to use as the base estimators.
                This must not be None or an empty list or else EnsembleMissingPipelinesError will be raised.
            final_estimator (Estimator or subclass): The classifier used to combine the base estimators. If None, uses LogisticRegressionClassifier.
            cv (int, cross-validation generator or an iterable): Determines the cross-validation splitting strategy used to train final_estimator.
                For int/None inputs, if the estimator is a classifier and y is either binary or multiclass, StratifiedKFold is used. Defaults to None.
                Possible inputs for cv are:

                - None: 3-fold cross validation
                - int: the number of folds in a (Stratified) KFold
                - An scikit-learn cross-validation generator object
                - An iterable yielding (train, test) splits
            n_jobs (int or None): Non-negative integer describing level of parallelism used for pipelines.
                None and 1 are equivalent. If set to -1, all CPUs are used. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.
                Defaults to None.
                - Note: there could be some multi-process errors thrown for values of `n_jobs != 1`. If this is the case, please use `n_jobs = 1`.
            random_seed (int): Seed for the random number generator. Defaults to 0.
        """
        super().__init__(input_pipelines=input_pipelines, final_estimator=final_estimator,
                         cv=cv, n_jobs=n_jobs, random_seed=random_seed, **kwargs)
