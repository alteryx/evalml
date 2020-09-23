from sklearn.ensemble import ExtraTreesClassifier as SKExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression as LogisticRegression

from evalml.model_family import ModelFamily
from evalml.pipelines.components import LogisticRegressionClassifier
from evalml.pipelines.components.ensemble import StackedEnsembleBase
from evalml.problem_types import ProblemTypes


class StackedEnsembleClassifier(StackedEnsembleBase):
    """Stacked Ensemble Classifier."""
    name = "Stacked Ensemble Classifier"
    model_family = ModelFamily.ENSEMBLE
    supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]
    hyperparameter_ranges = {}
    _stacking_estimator_class = StackingClassifier
    # _default_final_estimator = LogisticRegressionClassifier
    _default_final_estimator = LogisticRegression()

    _input_pipelines = [RandomForestClassifier(), LogisticRegression(), SKExtraTreesClassifier()]

    def __init__(self, input_pipelines=None, final_estimator=None,
                 cv=None, n_jobs=-1, random_state=0, **kwargs):
        """Stacked ensemble classifier.

        Arguments:
            input_pipelines (list(PipelineBase or subclass)): List of PipelineBase objects to use as the base estimators.
                This must not be None or an empty list or else EnsembleMissingPipelinesError will be raised.
            final_estimator (Estimator or subclass): The classifier used to combine the base estimators. If None, uses LogisticRegressionClassifier.
            cv (int, cross-validation generator or an iterable): Determines the cross-validation splitting strategy used to train final_estimator.
                For int/None inputs, if the estimator is a classifier and y is either binary or multiclass, StratifiedKFold is used. In all other cases, KFold is used.
                Possible inputs for cv are:
                    - None: 5-fold cross validation
                    - int: the number of folds in a (Stratified) KFold
                    - An scikit-learn cross-validation generator object
                    - An iterable yielding (train, test) splits
            n_jobs (int or None): Non-negative integer describing level of parallelism used for pipelines.
                None and 1 are equivalent. If set to -1, all CPUs are used. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.
            random_state (int, np.random.RandomState): seed for the random number generator
        """
        super().__init__(input_pipelines=input_pipelines, final_estimator=final_estimator,
                         cv=cv, n_jobs=n_jobs, random_state=random_state, **kwargs)
