from collections import OrderedDict

import pandas as pd

from evalml.objectives import get_objective
from evalml.pipelines.classification_pipeline import ClassificationPipeline


class MulticlassClassificationPipeline(ClassificationPipeline):
    problem_types = ['multiclass']

    def score(self, X, y, objectives):
        """Evaluate model performance on current and additional objectives

        Arguments:
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
                scores.update({objective.name: objective.score(y_predicted_proba, y, X=X)})
            else:
                if y_predicted is None:
                    y_predicted = self.predict(X)
                scores.update({objective.name: objective.score(y_predicted, y, X=X)})
        return scores
