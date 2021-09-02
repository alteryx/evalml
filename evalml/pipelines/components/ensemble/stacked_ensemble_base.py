from evalml.model_family import ModelFamily
from evalml.pipelines.components import Estimator
from evalml.utils import classproperty

_nonstackable_model_families = [ModelFamily.BASELINE, ModelFamily.NONE]


class StackedEnsembleBase(Estimator):
    """Stacked Ensemble Base Class.

    Arguments:
        final_estimator (Estimator or subclass): The estimator used to combine the base estimators.
        cv (int, cross-validation generator or an iterable): Determines the cross-validation splitting strategy used to train final_estimator.
            For int/None inputs, if the estimator is a classifier and y is either binary or multiclass, StratifiedKFold is used. In all other cases, KFold is used.
            Possible inputs for cv are:

            - None: 5-fold cross validation
            - int: the number of folds in a (Stratified) KFold
            - An scikit-learn cross-validation generator object
            - An iterable yielding (train, test) splits
        n_jobs (int or None): Integer describing level of parallelism used for pipelines. None and 1 are equivalent.
            If set to -1, all CPUs are used. For n_jobs greater than -1, (n_cpus + 1 + n_jobs) are used. Defaults to -1.
            - Note: there could be some multi-process errors thrown for values of `n_jobs != 1`. If this is the case, please use `n_jobs = 1`.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    model_family = ModelFamily.ENSEMBLE
    """ModelFamily.ENSEMBLE"""
    _default_final_estimator = None

    def __init__(
        self,
        final_estimator=None,
        n_jobs=-1,
        random_seed=0,
        **kwargs,
    ):
        final_estimator = final_estimator or self._default_final_estimator()

        parameters = {
            "final_estimator": final_estimator,
            "n_jobs": n_jobs,
        }
        parameters.update(kwargs)

        super().__init__(
            parameters=parameters,
            component_obj=final_estimator,
            random_seed=random_seed,
        )

    @property
    def feature_importance(self):
        """Not implemented for StackedEnsembleClassifier and StackedEnsembleRegressor"""
        raise NotImplementedError(
            "feature_importance is not implemented for StackedEnsembleClassifier and StackedEnsembleRegressor"
        )

    @classproperty
    def default_parameters(cls):
        """Returns the default parameters for stacked ensemble classes.

        Returns:
            dict: default parameters for this component.
        """
        return {
            "final_estimator": None,
            "n_jobs": -1,
        }
