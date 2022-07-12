"""Binary classification pipeline mix-in class."""


class BinaryClassificationPipelineMixin:
    """Binary classification pipeline mix-in class."""

    _threshold = None

    @property
    def threshold(self):
        """Threshold used to make a prediction. Defaults to None."""
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        self._threshold = value

    def _predict_with_objective(self, X, ypred_proba, objective):
        ypred_proba = ypred_proba.iloc[:, 1]
        if objective is None:
            return ypred_proba > self.threshold
        return objective.decision_function(ypred_proba, threshold=self.threshold, X=X)

    def _compute_predictions(self, X, y, objectives, time_series=False):
        """Compute predictions/probabilities based on objectives."""
        y_predicted = None
        y_predicted_proba = None
        if any(o.score_needs_proba for o in objectives) or self.threshold is not None:
            y_predicted_proba = (
                self.predict_proba(X, y) if time_series else self.predict_proba(X)
            )
        if any(not o.score_needs_proba for o in objectives) and self.threshold is None:
            y_predicted = (
                self._predict(X, y, pad=True) if time_series else self._predict(X)
            )
        return y_predicted, y_predicted_proba

    def _select_y_pred_for_score(self, X, y, y_pred, y_pred_proba, objective):
        y_pred_to_use = y_pred
        if self.threshold is not None and not objective.score_needs_proba:
            y_pred_to_use = self._predict_with_objective(X, y_pred_proba, objective)
        return y_pred_to_use

    def optimize_threshold(self, X, y, y_pred_proba, objective):
        """Optimize the pipeline threshold given the objective to use. Only used for binary problems with objectives whose thresholds can be tuned.

        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series): Input target values.
            y_pred_proba (pd.Series): The predicted probabilities of the target outputted by the pipeline.
            objective (ObjectiveBase): The objective to threshold with. Must have a tunable threshold.

        Raises:
            ValueError: If objective is not optimizable.
        """
        if self.can_tune_threshold_with_objective(objective):
            if self._encoder is not None:
                y = self._encode_targets(y)
            self.threshold = objective.optimize_threshold(y_pred_proba, y, X)
        else:
            raise ValueError(
                "Problem type must be binary and objective must be optimizable.",
            )
