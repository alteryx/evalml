import numpy as np
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
            self.final_estimator = LogisticRegressionClassifier()
        else:
            self.final_estimator = final_estimator
        sklearn_parameters.update({"estimators": [(estimator.name, estimator._component_obj) for estimator in estimators]})
        sklearn_parameters.update({"final_estimator": self.final_estimator._component_obj})
        stacked_classifier = StackingClassifier(**sklearn_parameters)
        self.stacked_classifier = stacked_classifier
        super().__init__(parameters=parameters,
                         component_obj=stacked_classifier,
                         random_state=random_state)

    def fit(self, X, y=None):
        self._component_obj.fit(X, y)
        self.final_estimator._is_fitted = True
        return self

    @property
    def feature_importance(self):
        """Returns importance associated with each feature.

        Returns:
            list(float): importance associated with each feature
        """
        if self.parameters["final_estimator"] is None:
            coef_ = self._component_obj.final_estimator_.coef_
            # binary classification case
            if len(coef_) <= 2:
                return coef_[0]
            else:
                # multiclass classification case
                return np.linalg.norm(coef_, axis=0, ord=2)
        print ("STACKED CLASSIFIER FINAL ESTIMATOR:", self.stacked_classifier.final_estimator.__repr__())
        print ("FINAL ESTIMATOR:", self.final_estimator.__repr__())
        print ("FINAL ESTIMATOR OBJ:", self.final_estimator._component_obj.__repr__())
        # return self.stacked_classifier.final_estimator.feature_importances_
        print (self.stacked_classifier.final_estimator == self.final_estimator._component_obj)
        return self.final_estimator.feature_importance
