from scipy.optimize import minimize_scalar


class ObjectiveBase:
    needs_fitting = False
    greater_is_better = True
    needs_proba = False
    uses_extra_columns = False

    def __init__(self):
        pass

    def fit(self, y_predicted, y_true, extra_cols=None):
        """Learn the objective function based on the predictions from a model.

        If needs_fitting is false, this method won't be called

        Arguments:
            y_predicted (list): the predictions from the model. If needs_proba is True,
                it is the probability estimates

            y_true (list): the ground truth for the predictions.

            extra_cols (extra_cols): any extra columns that are needed from training
                data to fit. Only provided if uses_extra_columns is True.

        Returns:
            self
        """
        def cost(threshold):
            if extra_cols is not None:
                predictions = self.decision_function(y_predicted, extra_cols, threshold)
                cost = self.objective_function(predictions, y_true, extra_cols)
            else:
                predictions = self.decision_function(y_predicted, threshold)
                cost = self.objective_function(predictions, y_true)

            if self.greater_is_better:
                return -cost

            return cost

        self.optimal = minimize_scalar(cost, method='Golden', options={"maxiter": 100})
        self.threshold = self.optimal.x
        print("minimzed", self.threshold)
        return self

    def predict(self, y_predicted, extra_cols=None):
        """Apply the learned objective function to the output of a model.

        If needs_fitting is false, this method won't be called

        Arguments:
            y_predicted: the prediction to transform to final prediction

        Returns:
            predictions
        """

        if extra_cols is not None:
            predictions = self.decision_function(y_predicted, extra_cols, self.threshold)
        else:
            predictions = self.decision_function(y_predicted, self.threshold)

        return predictions

    def score(self, y_predicted, y_true, extra_cols=None):
        """Calculate score from applying fitted objective to predicted values

        If a higher score is better than a lower score, set greater_is_better attribute to True

        Arguments:
            y_predicted (list): the predictions from the model. If needs_proba is True,
                it is the probability estimates

            y_true (list): the ground truth for the predictions.

            extra_cols (extra_cols): any extra columns that are needed from training
                data to fit. Only provided if uses_extra_columns is True.

        Returns:
            score

        """
        if extra_cols is not None:
            return self.objective_function(y_predicted, y_true, extra_cols)
        else:
            return self.objective_function(y_predicted, y_true)
