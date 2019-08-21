from sklearn.model_selection import train_test_split


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

        if self.objective.needs_fitting:
            X, X_objective, y, y_objective = train_test_split(X, y, test_size=objective_fit_size)

        self.pipeline.fit(X, y)

        if self.objective.needs_fitting:
            y_prob_predicted = self.predict_proba(X_objective)
            if self.objective.uses_extra_columns:
                self.objective.fit(y_prob_predicted, y_objective, X_objective)
            else:
                self.objective.fit(y_prob_predicted, y_objective)

        return self

    def predict(self, X):
        """Make predictions using selected features.

        Args:
            X (DataFrame) : features

        Returns:
            Series : estimated labels
        """
        if self.objective and self.objective.needs_fitting:
            y_prob_predicted = self.predict_proba(X)
            return self.objective.predict(y_prob_predicted)

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
            score, list of other objective scores
        """
        other_objectives = other_objectives or []

        y_predicted_proba = None
        y_predicted_label = None

        scores = []
        for objective in [self.objective] + other_objectives:
            # only call the predict methods once
            if objective.needs_proba:
                if y_predicted_proba is None:
                    y_predicted_proba = self.predict_proba(X)

                y_predicted = y_predicted_proba

            else:
                if y_predicted_label is None:
                    y_predicted_label = self.predict(X)

                y_predicted = y_predicted_label

            if objective.uses_extra_columns:
                scores.append(objective.score(y_predicted, y, X))
            else:
                scores.append(objective.score(y_predicted, y))

        if not other_objectives:
            return scores[0]

        return scores[0], scores[1:]

