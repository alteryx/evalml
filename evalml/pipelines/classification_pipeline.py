
from collections import OrderedDict

import pandas as pd

from evalml.objectives import get_objective
from evalml.pipelines import PipelineBase


class ClassificationPipeline(PipelineBase):

    def predict_proba(self, X):
        """Make probability estimates for labels.

        Args:
            X (pd.DataFrame or np.array) : data of shape [n_samples, n_features]

        Returns:
            pd.DataFrame : probability estimates
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X = self._transform(X)
        proba = self.estimator.predict_proba(X)
        return proba

    def score(self, X, y, objectives):
        """Evaluate model performance on current and additional objectives

        Args:
            X (pd.DataFrame or np.array) : data of shape [n_samples, n_features]
            y (pd.Series) : true labels of length [n_samples]
            objectives (list): list of objectives to score

        Returns:
            dict: ordered dictionary of objective scores
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        objectives = [get_objective(o) for o in objectives]
        y_predicted = None
        y_predicted_proba = None

        scores = OrderedDict()
        for objective in objectives:
            if objective.score_needs_proba:
                if y_predicted_proba is None:
                    y_predicted_proba = self.predict_proba(X)
                    y_predicted_proba = y_predicted_proba
                y_predictions = y_predicted_proba
            else:
                if y_predicted is None:
                    y_predicted = self.predict(X)
                y_predictions = y_predicted
            scores.update({objective.name: objective.score(y_predictions, y, X)})

        return scores
