from collections import OrderedDict

import pandas as pd
from sklearn.model_selection import train_test_split

from evalml.objectives import get_objective
from evalml.pipelines.classification_pipeline import ClassificationPipeline


class BinaryClassificationPipeline(ClassificationPipeline):

    def fit(self, X, y, objective=None, objective_fit_size=0.2):
        """Build a model

        Arguments:
            X (pd.DataFrame or np.array): the input training data of shape [n_samples, n_features]

            y (pd.Series): the target training labels of length [n_samples]

            objective (Object or string): the objective to optimize

            objective_fit_size (float): the proportion of the dataset to include in the test split.
        Returns:

            self

        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        if objective is not None:
            objective = get_objective(objective)
            if objective.needs_fitting:
                X, X_objective, y, y_objective = train_test_split(X, y, test_size=objective_fit_size, random_state=self.random_state)

        self._fit(X, y)

        if objective is not None:
            if objective.needs_fitting:
                y_predicted_proba = self.predict_proba(X_objective)
                y_predicted_proba = y_predicted_proba[:, 1]

                if objective.uses_extra_columns:
                    objective.fit(y_predicted_proba, y_objective, X_objective)
                else:
                    objective.fit(y_predicted_proba, y_objective)
        return self

    def predict(self, X, objective=None):
        """Make predictions using selected features.

        Args:
            X (pd.DataFrame or np.array) : data of shape [n_samples, n_features]
            objective (Object or string): the objective to use to predict

        Returns:
            pd.Series : estimated labels
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_t = self._transform(X)

        if objective is not None:
            objective = get_objective(objective)
            if objective.needs_fitting:
                y_predicted_proba = self.predict_proba(X)
                y_predicted_proba = y_predicted_proba[:, 1]
                if objective.uses_extra_columns:
                    return objective.predict(y_predicted_proba, X)
                else:
                    return objective.predict(y_predicted_proba)

        return self.estimator.predict(X_t)

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
                    y_predicted_proba = y_predicted_proba[:, 1]
                y_predictions = y_predicted_proba
            else:
                if y_predicted is None:
                    y_predicted = self.predict(X, objective)
                y_predictions = y_predicted

            if objective.uses_extra_columns:
                scores.update({objective.name: objective.score(y_predictions, y, X)})
            else:
                scores.update({objective.name: objective.score(y_predictions, y)})
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
