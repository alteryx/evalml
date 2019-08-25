import pandas as pd
from sklearn.model_selection import train_test_split

from evalml.objectives import get_objective


class PipelineBase:
    def __init__(self, objective, random_state=0):
        self.objective = objective
        self.random_state = random_state

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
        # make everything pandas objects
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        self.input_feature_names = X.columns.tolist()

        if self.objective.needs_fitting:
            X, X_objective, y, y_objective = train_test_split(X, y, test_size=objective_fit_size)

        self.pipeline.fit(X, y)

        if self.objective.needs_fitting:
            if self.objective.needs_proba:
                y_predicted = self.predict_proba(X_objective)
            else:
                y_predicted = self.predict(X_objective)

            if self.objective.uses_extra_columns:
                self.objective.fit(y_predicted, y_objective, X_objective)
            else:
                self.objective.fit(y_predicted, y_objective)

        return self

    def predict(self, X):
        """Make predictions using selected features.

        Args:
            X (DataFrame) : features

        Returns:
            Series : estimated labels
        """
        if self.objective and self.objective.needs_fitting:
            if self.objective.needs_proba:
                y_predicted = self.predict_proba(X)
            else:
                y_predicted = self.predict(X)

            if self.objective.uses_extra_columns:
                return self.objective.predict(y_predicted, X)

            return self.objective.predict(y_predicted)

        return self.pipeline.predict(X)

    def predict_proba(self, X):
        """Make probability estimates for labels.

        Args:
            X (DataFrame) : features

        Returns:
            DataFrame : probability estimates
        """

        return self.pipeline.predict_proba(X)[:, 1]

    def score(self, X, y, other_objectives=None):
        """Evaluate model performance

        Args:
            X (DataFrame) : features for model predictions
            y (Series) : true labels
            other_objectives (list): list of other objectives to score

        Returns:
            score, dictionary of other objective scores
        """
        other_objectives = other_objectives or []

        y_predicted = self.predict(X)

        scores = []
        for objective in [self.objective] + other_objectives:
            objective = get_objective(objective)

            if objective.uses_extra_columns:
                scores.append(objective.score(y_predicted, y, X))
            else:
                scores.append(objective.score(y_predicted, y))

        if not other_objectives:
            return scores[0]

        other_scores = dict(zip([n.name for n in other_objectives], scores[1:]))

        return scores[0], other_scores
