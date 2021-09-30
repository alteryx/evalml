"""Stacked Ensemble Base Class."""
import warnings

from evalml.exceptions import EnsembleMissingPipelinesError
from evalml.model_family import ModelFamily
from evalml.pipelines.components import Estimator
from evalml.pipelines.components.utils import scikit_learn_wrapped_estimator
from evalml.utils import classproperty

_nonstackable_model_families = [ModelFamily.BASELINE, ModelFamily.NONE]


class SklearnStackedEnsembleBase(Estimator):
    """Stacked Ensemble Base Class.

    Args:
        input_pipelines (list(PipelineBase or subclass obj)): List of pipeline instances to use as the base estimators.
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
            Defaults to -1.
            - Note: there could be some multi-process errors thrown for values of `n_jobs != 1`. If this is the case, please use `n_jobs = 1`.
        random_seed (int): Seed for the random number generator. Defaults to 0.

    Raises:
        EnsembleMissingPipelinesError: If `input_pipelines` is None or an empty list.
        ValueError: If any of the input pipelines cannot be used in a stacked ensemble.

    """

    model_family = ModelFamily.ENSEMBLE
    """ModelFamily.ENSEMBLE"""
    _stacking_estimator_class = None
    _default_final_estimator = None
    _default_cv = None

    def __init__(
        self,
        input_pipelines=None,
        final_estimator=None,
        cv=None,
        n_jobs=-1,
        random_seed=0,
        **kwargs,
    ):
        if not input_pipelines:
            raise EnsembleMissingPipelinesError(
                "`input_pipelines` must not be None or an empty list."
            )
        if [
            pipeline
            for pipeline in input_pipelines
            if pipeline.model_family in _nonstackable_model_families
        ]:
            raise ValueError(
                "Pipelines with any of the following model families cannot be used as base pipelines: {}".format(
                    _nonstackable_model_families
                )
            )

        parameters = {
            "input_pipelines": input_pipelines,
            "final_estimator": final_estimator,
            "cv": cv,
            "n_jobs": n_jobs,
        }
        parameters.update(kwargs)

        if len(set([pipeline.problem_type for pipeline in input_pipelines])) > 1:
            raise ValueError("All pipelines must have the same problem type.")

        cv = cv or self._default_cv(n_splits=3, random_state=random_seed, shuffle=True)
        estimators = [
            scikit_learn_wrapped_estimator(pipeline) for pipeline in input_pipelines
        ]
        final_estimator = scikit_learn_wrapped_estimator(
            final_estimator or self._default_final_estimator()
        )
        sklearn_parameters = {
            "estimators": [
                (f"({idx})", estimator) for idx, estimator in enumerate(estimators)
            ],
            "final_estimator": final_estimator,
            "cv": cv,
            "n_jobs": n_jobs,
        }
        sklearn_parameters.update(kwargs)
        super().__init__(
            parameters=parameters,
            component_obj=self._stacking_estimator_class(**sklearn_parameters),
            random_seed=random_seed,
        )
        warnings.warn(
            "Scikit-learn based ensemblers will be completely removed in the next release. Utilize the new `StackedEnsembleRegressor` or `StackedEnsembleClassifier` ensembler instead.",
            DeprecationWarning,
        )

    @property
    def feature_importance(self):
        """Not implemented for SklearnStackedEnsembleClassifier and SklearnStackedEnsembleRegressor."""
        raise NotImplementedError(
            "feature_importance is not implemented for SklearnStackedEnsembleClassifier and SklearnStackedEnsembleRegressor"
        )

    @classproperty
    def default_parameters(cls):
        """Returns the default parameters for stacked ensemble classes.

        Returns:
            dict: Default parameters for this component.
        """
        return {
            "final_estimator": None,
            "cv": None,
            "n_jobs": -1,
        }
