from sklearn.ensemble import StackingClassifier

from evalml.model_family import ModelFamily
from evalml.pipelines.components.ensemble import EnsembleBase
from evalml.problem_types import ProblemTypes


class StackedEnsembleClassifier(EnsembleBase):

    """Stacked Ensemble Classifier."""
    name = "Stacked Ensemble Classifier"
    model_family = ModelFamily.ENSEMBLE
    supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

    def __init__(self, estimators, final_estimator=None, cv=None, n_jobs=-1, random_state=0, **kwargs):
        parameters = {
            "estimators": estimators,
            "final_estimator": final_estimator,
            "cv": cv,
            "n_jobs": n_jobs
        }
        parameters.update(kwargs)
        stacked_classifier = StackingClassifier(**parameters)
        super().__init__(parameters=parameters,
                         component_obj=stacked_classifier,
                         random_state=random_state)
