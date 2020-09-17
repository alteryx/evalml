from evalml.exceptions import EnsembleMissingPipelinesError
from evalml.model_family import ModelFamily
from evalml.pipelines.components import Estimator

_nonstackable_model_families = [ModelFamily.BASELINE, ModelFamily.NONE]


class StackedEnsembleBase(Estimator):
    """Stacked Ensemble Base Class."""
    model_family = ModelFamily.ENSEMBLE
    _stacking_estimator_class = None
    _default_final_estimator = None

    def __init__(self, input_pipelines=None, final_estimator=None, cv=None, n_jobs=-1, random_state=0, **kwargs):
        """Stacked ensemble base class.

        Arguments:
            input_pipelines (list(PipelineBase or subclass)): List of PipelineBase objects to use as the base estimators.
                This must not be None or an empty list or else EnsembleMissingPipelinesError will be raised.
            final_estimator (Estimator or subclass): The estimator used to combine the base estimators.
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
        if not input_pipelines:
            raise EnsembleMissingPipelinesError("`input_pipelines` must not be None or an empty list.")
        contains_non_stackable = [pipeline for pipeline in input_pipelines if pipeline.model_family in _nonstackable_model_families]
        if contains_non_stackable:
            raise ValueError("Estimators with any of the following model families cannot be used as base estimators: {}".format(_nonstackable_model_families))
        parameters = {
            "input_pipelines": input_pipelines,
            "final_estimator": final_estimator,
            "cv": cv,
            "n_jobs": n_jobs
        }
        parameters.update(kwargs)

        if final_estimator is None:
            final_estimator = self._default_final_estimator()
        estimators = [pipeline.estimator for pipeline in input_pipelines]
        component_without_obj = [estimator for estimator in estimators + [final_estimator] if estimator._component_obj is None]
        if component_without_obj:
            raise ValueError("All estimators and final_estimator must have a valid ._component_obj")
        sklearn_parameters = {
            "estimators": [(estimator.name + f"({idx})", estimator._component_obj) for idx, estimator in enumerate(estimators)],
            "final_estimator": final_estimator._component_obj,
            "cv": cv,
            "n_jobs": n_jobs
        }
        sklearn_parameters.update(kwargs)
        super().__init__(parameters=parameters,
                         component_obj=self._stacking_estimator_class(**sklearn_parameters),
                         random_state=random_state)

    @property
    def feature_importance(self):
        raise NotImplementedError("feature_importance is not implemented for StackedEnsembleClassifier and StackedEnsembleRegressor")
