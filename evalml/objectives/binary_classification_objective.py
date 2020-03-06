from scipy.optimize import minimize_scalar

from .objective_base import ObjectiveBase

from evalml.problem_types import ProblemTypes


class BinaryClassificationObjective(ObjectiveBase):
    can_optimize_threshold = False
    optimal = None
    threshold = None
    problem_type = ProblemTypes.BINARY

    # def optimize_threshold(self, ypred_proba, y_true, X=None):
    #     """Learn and optimize the threshold objective function based on the predictions from a model.
    #     Arguments:
    #         ypred_proba (list): the predictions from the model. If needs_proba is True,
    #             it is the probability estimates
    #         y_true (list): the ground truth for the predictions.
    #         X (pd.DataFrame): any extra columns that are needed from training
    #             data to fit.
    #     Returns:
    #         optimal threshold
    #     """
    #     def cost(threshold):
    #         predictions = self.decision_function(ypred_proba, threshold, X=X)
    #         cost = self.objective_function(ypred_proba, y_true, X=X)
    #         return -cost if self.greater_is_better else cost
    #     optimal = minimize_scalar(cost, method='Golden', options={"maxiter": 100})
    #     self.optimal = optimal.x
    #     return self.optimal

    def optimize_threshold(self, ypred_proba, y_true, X=None):
        """Learn the objective function based on the predictions from a model.
            TODO: formerly the fit() function
        Arguments:
            y_predicted (list): the predictions from the model. If needs_proba is True,
                it is the probability estimates

            y_true (list): the ground truth for the predictions.

            extra_cols (pd.DataFrame): any extra columns that are needed from training
                data to fit.

        Returns:
            self
        """

        def cost(threshold):
            predictions = self.decision_function(ypred_proba=ypred_proba, classification_threshold=threshold, X=X)
            cost = self.objective_function(predictions, y_true, X=X)
            return -cost if self.greater_is_better else cost
        optimal = minimize_scalar(cost, method='Golden', options={"maxiter": 100})
        self.optimal = optimal  # is this necessary?
        self.threshold = self.optimal.x

        return self.threshold

    def predict(self, y_predicted, X=None):
        """Apply the learned objective function to the output of a model.

        Arguments:
            y_predicted: the prediction to transform to final prediction

        Returns:
            predictions
        """
        if self.threshold is not None:
            predictions = self.decision_function(y_predicted, self.threshold, X)
        else:
            predictions = self.decision_function(y_predicted, 0.0, X)  # todo
        return predictions

    def decision_function(self, ypred_proba, classification_threshold=0.0, X=None):
        """Apply the learned objective function to the output of a model.
        note to self (delete later): old predict()
        Arguments:
            ypred_proba: the prediction to transform to final prediction
        Returns:
            predictions
        """
        return ypred_proba > classification_threshold

    def score(self, y_predicted, y_true, X=None):
        """Calculate score from applying fitted objective to predicted values

        If a higher score is better than a lower score, set greater_is_better attribute to True

        Arguments:
            y_predicted (list): the predictions from the model. If needs_proba is True,
                it is the probability estimates

            y_true (list): the ground truth for the predictions.

            X (pd.DataFrame): any extra columns that are needed from training
                data to fit. Only provided if uses_extra_columns is True.

        Returns:
            score

        """
        return self.objective_function(y_predicted, y_true, X)
