from sklearn.ensemble import StackingRegressor

from evalml.exceptions import EnsembleMissingEstimatorsError
from evalml.model_family import ModelFamily
from evalml.pipelines.components.ensemble import EnsembleBase
from evalml.problem_types import ProblemTypes


class StackedEnsembleRegressor(EnsembleBase):
    """Stacked Ensemble Regressor."""
    name = "Stacked Ensemble Regressor"
    model_family = ModelFamily.ENSEMBLE
    supported_problem_types = [ProblemTypes.REGRESSION]
    hyperparameter_ranges = {}

    def __init__(self, final_estimator=None, cv=None, n_jobs=-1, random_state=0, **kwargs):
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
        if final_estimator is not None:
            sklearn_parameters.update({"final_estimator": final_estimator._component_obj})
        sklearn_parameters.update({"estimators": [(estimator.name, estimator._component_obj) for estimator in estimators]})
        self._stacked_classifier = StackingRegressor(**sklearn_parameters)
        super().__init__(parameters=parameters,
                         component_obj=self._stacked_classifier,
                         random_state=random_state)

    @property
    def feature_importance(self):
        raise NotImplementedError("feature_importance is not implemented for StackedEnsembleRegressor")
