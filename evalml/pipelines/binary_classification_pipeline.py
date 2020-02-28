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

    def predict(self, X, objective):
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
