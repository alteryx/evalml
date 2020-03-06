from scipy.optimize import minimize_scalar

from .objective_base import ObjectiveBase

from evalml.problem_types import ProblemTypes


class BinaryClassificationObjective(ObjectiveBase):
    can_optimize_threshold = False
    threshold = None
    problem_type = ProblemTypes.BINARY

    def optimize_threshold(self, ypred_proba, y_true, X=None):
        """Learn and optimize the objective function based on the predictions from a model.

        Arguments:
            ypred_proba (list): the probability estimatrs from the model.

            y_true (list): the ground truth for the predictions.

            X (pd.DataFrame): any extra columns that are needed from training data.

        Returns:
            optimal threshold
        """
        def cost(threshold):
            predictions = self.decision_function(ypred_proba=ypred_proba, classification_threshold=threshold, X=X)
            cost = self.objective_function(predictions, y_true, X=X)
            return -cost if self.greater_is_better else cost
        optimal = minimize_scalar(cost, method='Golden', options={"maxiter": 100})
        self.threshold = optimal.x

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
            print("using default")
            predictions = self.decision_function(y_predicted, 0.5, X)  # todo
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
