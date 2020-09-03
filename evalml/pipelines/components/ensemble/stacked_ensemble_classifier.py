from sklearn.ensemble import StackingClassifier

from evalml.model_family import ModelFamily
from evalml.pipelines.components import LogisticRegressionClassifier
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
        sklearn_parameters = parameters.copy()
        parameters.update(kwargs)
        if final_estimator is None:
            self._final_estimator = LogisticRegressionClassifier()
        else:
            self._final_estimator = final_estimator
        sklearn_parameters.update({"estimators": [(f'{estimator.name} ({index})', estimator._component_obj) for index, estimator in enumerate(estimators)]})
        sklearn_parameters.update({"final_estimator": self._final_estimator._component_obj})
        self._stacked_classifier = StackingClassifier(**sklearn_parameters)
        super().__init__(parameters=parameters,
                         component_obj=self._stacked_classifier,
                         random_state=random_state)

    def fit(self, X, y=None):
        # TODO: write note as to why this is done
        self._component_obj.fit(X, y)
        self._final_estimator._is_fitted = True
        self._final_estimator._component_obj = self._stacked_classifier.final_estimator_

    @property
    def feature_importance(self):
        """Returns importance associated with each feature.

        Returns:
            list(float): importance associated with each feature
        """
        return self._final_estimator.feature_importance
