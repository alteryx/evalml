from sklearn.ensemble import StackingRegressor

from evalml.exceptions import EnsembleMissingEstimatorsError
from evalml.model_family import ModelFamily
from evalml.pipelines.components import LinearRegressor
from evalml.pipelines.components.ensemble import EnsembleBase
from evalml.problem_types import ProblemTypes


class StackedEnsembleRegressor(EnsembleBase):
    """Stacked Ensemble Regressor."""
    name = "Stacked Ensemble Regressor"
    model_family = ModelFamily.ENSEMBLE
    supported_problem_types = [ProblemTypes.REGRESSION]
    hyperparameter_ranges = {}

    def __init__(self, final_estimator=None, cv=None, n_jobs=-1, random_state=0, **kwargs):
        """Stacked ensemble regressor.

        Arguments:
            final_estimator (Estimator or subclass): The regressor used to combine the base estimators. If None, uses LinearRegressor.
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
            **kwargs: 'estimators' containing a list of Estimator objects must be passed as a keyword argument, or else EnsembleMissingEstimatorsError will be raised
        """
        if 'estimators' not in kwargs:
            raise EnsembleMissingEstimatorsError("`estimators` must be passed to the constructor as a keyword argument")
        estimators = kwargs.get('estimators')
        parameters = {
            "estimators": estimators,
            "final_estimator": final_estimator,
            "cv": cv,
            "n_jobs": n_jobs
        }
        sklearn_parameters = parameters.copy()
        parameters.update(kwargs)
        if final_estimator is None:
            final_estimator = LinearRegressor()
        sklearn_parameters.update({"final_estimator": final_estimator._component_obj})
        sklearn_parameters.update({"estimators": [(estimator.name, estimator._component_obj) for estimator in estimators]})
        self._stacked_classifier = StackingRegressor(**sklearn_parameters)
        super().__init__(parameters=parameters,
                         component_obj=self._stacked_classifier,
                         random_state=random_state)

    @property
    def feature_importance(self):
        raise NotImplementedError("feature_importance is not implemented for StackedEnsembleRegressor")
