from collections import OrderedDict

import pandas as pd

from evalml.objectives import get_objective
from evalml.pipelines.classification_pipeline import ClassificationPipeline
from evalml.problem_types import ProblemTypes


class BinaryClassificationPipeline(ClassificationPipeline):

    threshold = None
    supported_problem_types = ['binary']

    def predict(self, X, objective=None):
        """Make predictions using selected features.

        Arguments:
            X (pd.DataFrame or np.array) : data of shape [n_samples, n_features]
            objective (Object or string): the objective to use to make predictions
        Returns:
            pd.Series : estimated labels
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X_t = self._transform(X)

        if objective is not None:
            objective = get_objective(objective)
            if objective.problem_type != ProblemTypes.BINARY:
                raise ValueError("You can only use a binary classification objective to make predictions for a binary classification pipeline.")

        if self.threshold is None:
            return self.estimator.predict(X_t)
        ypred_proba = self.predict_proba(X)
        ypred_proba = ypred_proba[:, 1]
        if objective is None:
            return ypred_proba > self.threshold
        return objective.decision_function(ypred_proba, threshold=self.threshold, X=X)

    def score(self, X, y, objectives):
        """Evaluate model performance on objectives

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
                    y_predicted_proba = y_predicted_proba[:, 1]
                y_predictions = y_predicted_proba
            else:
                if y_predicted is None:
                    y_predicted = self.predict(X, objective)
                y_predictions = y_predicted
            scores.update({objective.name: objective.score(y_predictions, y, X=X)})

        return scores

    def get_plot_data(self, X, y, plot_metrics):
        """Generates plotting data for the pipeline for each specified plot metric

        Args:
            X (pd.DataFrame or np.array) : data of shape [n_samples, n_features]
            y (pd.Series) : true labels of length [n_samples]
            plot_metrics (list): list of plot metrics to generate data for

        Returns:
            dict: ordered dictionary of plot metric data (scores)
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        y_predicted = None
        y_predicted_proba = None
        scores = OrderedDict()
        for plot_metric in plot_metrics:
            if plot_metric.score_needs_proba:
                if y_predicted_proba is None:
                    y_predicted_proba = self.predict_proba(X)
                    y_predicted_proba = y_predicted_proba[:, 1]
                y_predictions = y_predicted_proba
            else:
                if y_predicted is None:
                    y_predicted = self.predict(X)
                y_predictions = y_predicted
            scores.update({plot_metric.name: plot_metric.score(y_predictions, y)})
        return scores
