from evalml.objectives import get_objective
from evalml.pipelines.classification_pipeline import ClassificationPipeline
from evalml.problem_types import ProblemTypes
from evalml.utils import infer_feature_types


class BinaryClassificationPipeline(ClassificationPipeline):
    """Pipeline subclass for all binary classification pipelines."""
    _threshold = None
    problem_type = ProblemTypes.BINARY

    @property
    def threshold(self):
        """Threshold used to make a prediction. Defaults to None."""
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        self._threshold = value

    def _predict(self, X, objective=None):
        """Make predictions using selected features.

        Arguments:
            X (ww.DataTable, pd.DataFrame): Data of shape [n_samples, n_features]
            objective (Object or string): The objective to use to make predictions

        Returns:
            ww.DataColumn: Estimated labels
        """

        if objective is not None:
            objective = get_objective(objective, return_instance=True)
            if not objective.is_defined_for_problem_type(self.problem_type):
                raise ValueError("You can only use a binary classification objective to make predictions for a binary classification pipeline.")

        if self.threshold is None:
            return self._component_graph.predict(X)
        ypred_proba = self.predict_proba(X).to_dataframe()
        ypred_proba = ypred_proba.iloc[:, 1]
        if objective is None:
            return infer_feature_types(ypred_proba > self.threshold)
        return infer_feature_types(objective.decision_function(ypred_proba, threshold=self.threshold, X=X))

    def predict_proba(self, X):
        """Make probability estimates for labels. Assumes that the column at index 1 represents the positive label case.

        Arguments:
            X (ww.DataTable, pd.DataFrame or np.ndarray): Data of shape [n_samples, n_features]

        Returns:
            ww.DataTable: Probability estimates
        """
        return super().predict_proba(X)

    @staticmethod
    def _score(X, y, predictions, objective):
        """Given data, model predictions or predicted probabilities computed on the data, and an objective, evaluate and return the objective score.
        """
        if predictions.ndim > 1:
            predictions = predictions.iloc[:, 1]
        return ClassificationPipeline._score(X, y, predictions, objective)

    def score(self, X, y, objectives):
        """Evaluate model performance on objectives

        Arguments:
            X (ww.DataTable, pd.DataFrame or np.ndarray): Data of shape [n_samples, n_features]
            y (ww.DataColumn, pd.Series, or np.ndarray): True labels of length [n_samples]
            objectives (list): List of objectives to score

        Returns:
            dict: Ordered dictionary of objective scores
        """
        y = infer_feature_types(y)
        y = _convert_woodwork_types_wrapper(y.to_series())
        objectives = self.create_objectives(objectives)
        y = self._encode_targets(y)
        y_predicted, y_predicted_proba = self._compute_predictions(X, y, objectives)
        if y_predicted is not None:
            y_predicted = _convert_woodwork_types_wrapper(y_predicted.to_series())
        if y_predicted_proba is not None:
            y_predicted_proba = _convert_woodwork_types_wrapper(y_predicted_proba.to_dataframe())
        return self._score_all_objectives(X, y, y_predicted, y_predicted_proba, objectives)

    def _score_all_objectives(self, X, y, y_pred, y_pred_proba, objectives):
        """Given data, model predictions or predicted probabilities computed on the data, and an objective, evaluate and return the objective score.

        Will raise a PipelineScoreError if any objectives fail.
        Arguments:
            X (pd.DataFrame): The feature matrix.
            y (pd.Series): The target data.
            y_pred (pd.Series): The pipeline predictions.
            y_pred_proba (pd.Dataframe, pd.Series, None): The predicted probabilities for classification problems.
                Will be a DataFrame for multiclass problems and Series otherwise. Will be None for regression problems.
            objectives (list): List of objectives to score.

        Returns:
            dict: Ordered dictionary with objectives and their scores.
        """
        scored_successfully = OrderedDict()
        exceptions = OrderedDict()
        for objective in objectives:
            try:
                if not objective.is_defined_for_problem_type(self.problem_type):
                    raise ValueError(f'Invalid objective {objective.name} specified for problem type {self.problem_type}')
                score = self._score(X, y, y_pred_proba if objective.score_needs_proba else y_pred, objective)
                scored_successfully.update({objective.name: score})
            except Exception as e:
                tb = traceback.format_tb(sys.exc_info()[2])
                exceptions[objective.name] = (e, tb)
        if exceptions:
            # If any objective failed, throw an PipelineScoreError
            raise PipelineScoreError(exceptions, scored_successfully)
        # No objectives failed, return the scores
        return scored_successfully

    def _compute_predictions(self, X, y, objectives, time_series=False):
        """Compute predictions/probabilities based on objectives."""
        y_predicted = None
        y_predicted_proba = None
        # TODO
        if any(o.score_needs_proba for o in objectives):
            y_predicted_proba = self.predict_proba(X, y) if time_series else self.predict_proba(X)
        if any(not o.score_needs_proba for o in objectives):
            if self.threshold is None:
                y_predicted = self._predict(X, y, objective, pad=True) if time_series else self._predict(X, objective)
            else:
                if y_predicted_proba is None:
                    y_predicted_proba = self.predict_proba(X, y) if time_series else self.predict_proba(X)
        return y_predicted, y_predicted_proba


    def _predict(self, X, objective=None):
        if self.threshold is None:
            return self._component_graph.predict(X)
        ypred_proba = self.predict_proba(X).to_dataframe()
        ypred_proba = ypred_proba.iloc[:, 1]
        if objective is None:
            return infer_feature_types(ypred_proba > self.threshold)
        return infer_feature_types(objective.decision_function(ypred_proba, threshold=self.threshold, X=X))
