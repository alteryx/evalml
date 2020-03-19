import pandas as pd
from scipy.optimize import minimize_scalar

from .objective_base import ObjectiveBase

from evalml.problem_types import ProblemTypes


class BinaryClassificationObjective(ObjectiveBase):
    problem_type = ProblemTypes.BINARY
    can_optimize_threshold = False
    # threshold = None

    # def _optimize_threshold(self, X_objective, y_objective, objective):
    #     y_predicted_proba = self.predict_proba(X_objective)
    #     y_predicted_proba = y_predicted_proba[:, 1]
    #     objective_to_optimize = objective
    #     # for f1/auc to use accuracy by default
    #     if objective is None or not objective.can_optimize_threshold:
    #         objective_to_optimize = Accuracy()
    #     self.threshold = objective_to_optimize.optimize_threshold(y_predicted_proba, y_objective, X=X_objective)
    #     self.optimized_objective = objective_to_optimize




    def optimize_threshold(self, y_predicted, y_true, X=None):
        """Learn a binary classification threshold which optimizes the current objective.

        Arguments:
            y_predicted (list): the probability estimators from the model.

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
        return optimal.x

    def decision_function(self, ypred_proba, threshold=0.5, X=None):
        """Apply a learned threshold to predicted probabilities to get predicted classes.

        Arguments:
            ypred_proba (list): the prediction to transform to final prediction

            threshold (float): threshold used to make a prediction. Defaults to 0.5.

            X (pd.DataFrame): any extra columns that are needed from training data.

        Returns:
            predictions
        """
        if not isinstance(ypred_proba, pd.Series):
            ypred_proba = pd.Series(ypred_proba)
        return ypred_proba > threshold
