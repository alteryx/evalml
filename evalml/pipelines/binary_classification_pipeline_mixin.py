from evalml.exceptions import PipelineScoreError


class BinaryClassificationPipelineMixin():
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
            y_predicted_proba = self.predict_proba(X, y) if time_series else self.predict_proba(X)
        if any(not o.score_needs_proba for o in objectives) and self.threshold is None:
            y_predicted = self._predict(X, y, pad=True) if time_series else self._predict(X)
        return y_predicted, y_predicted_proba

    def _select_y_pred_for_score(self, X, y, y_pred, y_pred_proba, objective):
        y_pred_to_use = y_pred
        if self.threshold is not None and not objective.score_needs_proba:
            y_pred_to_use = self._predict_with_objective(X, y_pred_proba, objective)
        return y_pred_to_use

    # def _score_all_objectives(self, X, y, y_pred, y_pred_proba, objectives):
    #     scored_successfully = OrderedDict()
    #     exceptions = OrderedDict()
    #     for objective in objectives:
    #         try:
    #             if not objective.is_defined_for_problem_type(self.problem_type):
    #                 raise ValueError(f'Invalid objective {objective.name} specified for problem type {self.problem_type}')
    #             y_pred = self._select_y_pred_for_score(X, y, y_pred,)
    #             # y_pred_to_use = y_pred
    #             # if self.threshold is not None and not objective.score_needs_proba:
    #             #     y_pred_to_use = self._predict_with_objective(X, y_pred_proba, objective)
    #             # score = self._score(X, y, y_pred_proba if objective.score_needs_proba else y_pred_to_use, objective)
    #             scored_successfully.update({objective.name: score})
    #         except Exception as e:
    #             tb = traceback.format_tb(sys.exc_info()[2])
    #             exceptions[objective.name] = (e, tb)
    #     if exceptions:
    #         # If any objective failed, throw an PipelineScoreError
    #         raise PipelineScoreError(exceptions, scored_successfully)
    #     # No objectives failed, return the scores
    #     return scored_successfully
