import pandas as pd

from evalml.objectives import get_objective
from evalml.pipelines.classification_pipeline import ClassificationPipeline
from evalml.problem_types import ProblemTypes


class BinaryClassificationPipeline(ClassificationPipeline):
    """Pipeline subclass for all binary classification pipelines."""
    threshold = None
    problem_type = ProblemTypes.BINARY

    def _predict(self, X, objective=None):
        """Make predictions using selected features.

        Arguments:
            X (pd.DataFrame or np.array): data of shape [n_samples, n_features]
            objective (Object or string): the objective to use to make predictions

        Returns:
            pd.Series: estimated labels
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X_t = self._transform(X)

        if objective is not None:
            objective = get_objective(objective, return_instance=True)
            if objective.problem_type != self.problem_type:
                raise ValueError("You can only use a binary classification objective to make predictions for a binary classification pipeline.")

        if self.threshold is None:
            return self.estimator.predict(X_t)
        ypred_proba = self.predict_proba(X)
        ypred_proba = ypred_proba.iloc[:, 1]
        if objective is None:
            return ypred_proba > self.threshold
        return objective.decision_function(ypred_proba, threshold=self.threshold, X=X)

    def predict_proba(self, X):
        """Make probability estimates for labels. Assumes that the column at index 1 represents the positive label case.

        Arguments:
            X (pd.DataFrame or np.array): data of shape [n_samples, n_features]

        Returns:
            pd.DataFrame: probability estimates
        """
        return super().predict_proba(X)

    @staticmethod
    def _score(X, y, predictions, objective):
        """Given data, model predictions or predicted probabilities computed on the data, and an objective, evaluate and return the objective score.
        """
        if predictions.ndim > 1:
            predictions = predictions.iloc[:, 1]
        return ClassificationPipeline._score(X, y, predictions, objective)
