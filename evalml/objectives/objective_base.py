from scipy.optimize import minimize_scalar

from evalml.problem_types import handle_problem_types


class ObjectiveBase:
    needs_fitting = False
    greater_is_better = True
    score_needs_proba = False
    uses_extra_columns = False
    problem_types = []

    def __init__(self, verbose=False):
        self.verbose = verbose

    @classmethod
    def supports_problem_type(cls, problem_type):
        """ Checks if objective supports given ProblemType

        Arguments:
            problem_type(str or ProblemType): problem type to check
        Returns:
            bool: whether objective supports ProblemType
        """
        problem_type = handle_problem_types(problem_type)
        if problem_type in cls.problem_types:
            return True
        return False

    def fit(self, y_predicted, y_true, extra_cols=None):
        """Learn the objective function based on the predictions from a model.

        If needs_fitting is false, this method won't be called

        Arguments:
            y_predicted (list): the predictions from the model. If needs_proba is True,
                it is the probability estimates

            y_true (list): the ground truth for the predictions.

            extra_cols (pd.DataFrame): any extra columns that are needed from training
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

        if self.verbose:
            print("Best threshold found at: ", self.threshold)

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

            extra_cols (pd.DataFrame): any extra columns that are needed from training
                data to fit. Only provided if uses_extra_columns is True.

        Returns:
            score

        """
        if extra_cols is not None:
            return self.objective_function(y_predicted, y_true, extra_cols)
        else:
            return self.objective_function(y_predicted, y_true)
