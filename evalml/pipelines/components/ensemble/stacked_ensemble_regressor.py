from sklearn.ensemble import StackingRegressor

from evalml.model_family import ModelFamily
from evalml.pipelines.components.ensemble import EnsembleBase
from evalml.problem_types import ProblemTypes


class StackedEnsembleRegressor(EnsembleBase):
    """Stacked Ensemble Regressor."""
    name = "Stacked Ensemble Regressor"
    model_family = ModelFamily.ENSEMBLE
    supported_problem_types = [ProblemTypes.REGRESSION]

    def __init__(self, estimators, final_estimator=None, cv=None, n_jobs=-1, random_state=0, **kwargs):
        parameters = {
            "estimators": estimators,
            "final_estimator": final_estimator,
            "cv": cv,
            "n_jobs": n_jobs
        }
        parameters.update(kwargs)
        stacked_regressor = StackingRegressor(**parameters)
        super().__init__(parameters=parameters,
                         component_obj=stacked_regressor,
                         random_state=random_state)

    # @classmethod def make_ensemble(X, y, estimators_list, final_estimator_cls)
