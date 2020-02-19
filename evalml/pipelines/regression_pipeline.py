from collections import OrderedDict

import pandas as pd
from sklearn.model_selection import train_test_split

from .components import Estimator, handle_component
from .pipeline_plots import PipelinePlots

from evalml.objectives import get_objective
from evalml.pipelines import PipelineBase
from evalml.utils import Logger


class RegressionPipeline(PipelineBase):

    # Necessary for "Plotting" documentation, since Sphinx does not work well with instance attributes.
    plot = PipelinePlots

    def fit(self, X, y, objective_fit_size=.2):
        """Build a model

        Arguments:
            X (pd.DataFrame or np.array): the input training data of shape [n_samples, n_features]

            y (pd.Series): the target training labels of length [n_samples]

            feature_types (list, optional): list of feature types. either numeric of categorical.
                categorical features will automatically be encoded

        Returns:

            self

        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        if self.objective.needs_fitting:
            X, X_objective, y, y_objective = train_test_split(X, y, test_size=objective_fit_size, random_state=self.random_state)

        self._fit(X, y)

        if self.objective.needs_fitting:
            y_predicted = self.predict_proba(X_objective)

            if self.objective.uses_extra_columns:
                self.objective.fit(y_predicted, y_objective, X_objective)
            else:
                self.objective.fit(y_predicted, y_objective)
        return self

    def predict(self, X):
        """Make predictions using selected features.

        Args:
            X (pd.DataFrame or np.array) : data of shape [n_samples, n_features]

        Returns:
            pd.Series : estimated labels
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_t = self._transform(X)

        if self.objective and self.objective.needs_fitting:
            y_predicted = self.predict_proba(X)

            if self.objective.uses_extra_columns:
                return self.objective.predict(y_predicted, X)

            return self.objective.predict(y_predicted)

        return self.estimator.predict(X_t)

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

        if proba.shape[1] <= 2:
            return proba[:, 1]
        else:
            return proba

    def score(self, X, y, other_objectives=None):
        """Evaluate model performance on current and additional objectives

        Args:
            X (pd.DataFrame or np.array) : data of shape [n_samples, n_features]
            y (pd.Series) : true labels of length [n_samples]
            other_objectives (list): list of other objectives to score

        Returns:
            float, dict:  score, ordered dictionary of other objective scores
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        other_objectives = other_objectives or []
        other_objectives = [get_objective(o) for o in other_objectives]
        y_predicted = None
        y_predicted_proba = None

        scores = []
        for objective in [self.objective] + other_objectives:
            if objective.score_needs_proba:
                if y_predicted_proba is None:
                    y_predicted_proba = self.predict_proba(X)
                y_predictions = y_predicted_proba
            else:
                if y_predicted is None:
                    y_predicted = self.predict(X)
                y_predictions = y_predicted

            if objective.uses_extra_columns:
                scores.append(objective.score(y_predictions, y, X))
            else:
                scores.append(objective.score(y_predictions, y))
        if not other_objectives:
            return scores[0], {}

        other_scores = OrderedDict(zip([n.name for n in other_objectives], scores[1:]))

        return scores[0], other_scores
