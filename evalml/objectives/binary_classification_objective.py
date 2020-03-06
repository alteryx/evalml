import pandas as pd
from scipy.optimize import minimize_scalar

from .objective_base import ObjectiveBase

from evalml.problem_types import ProblemTypes


class BinaryClassificationObjective(ObjectiveBase):
    problem_type = ProblemTypes.BINARY
    can_optimize_threshold = False
    threshold = None

    def optimize_threshold(self, y_predicted, y_true, X=None):
        """Learn and optimize the objective function based on the predictions from a model.

        Arguments:
            y_predicted (list): the probability estimatrs from the model.

            y_true (list): the ground truth for the predictions.

            X (pd.DataFrame): any extra columns that are needed from training data.

        Returns:
            optimal threshold
        """
        def cost(threshold):
            predictions = self.decision_function(ypred_proba=y_predicted, threshold=threshold, X=X)
            cost = self.objective_function(predictions, y_true, X=X)
            return -cost if self.greater_is_better else cost

        optimal = minimize_scalar(cost, method='Golden', options={"maxiter": 100})
        self.threshold = optimal.x
        return self.threshold

    def decision_function(self, ypred_proba, threshold=0.0, X=None):
        """Apply the learned objective function to the output of a model.
        Arguments:
            ypred_proba (list): the prediction to transform to final prediction
            threshold (float): threshold used to make a prediction. Defaults to 0.
        Returns:
            predictions
        """
        if not isinstance(ypred_proba, pd.Series):
            ypred_proba = pd.Series(ypred_proba)
        return ypred_proba > threshold

    def predict(self, y_predicted, X=None):
        """Apply the learned objective function to the output of a model.

        Arguments:
            y_predicted (list): the prediction to transform to final prediction
            X (pd.DataFrame): extra data used to make a prediction
        Returns:
            predictions
        """
        if self.threshold is not None:
            predictions = self.decision_function(y_predicted, self.threshold, X)
        else:
            # raise Exception("Didn't optimize for objective first!")
            predictions = self.decision_function(y_predicted, 0.0, X)  # todo
        return predictions
