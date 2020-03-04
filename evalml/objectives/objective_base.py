
from evalml.problem_types import handle_problem_types


class ObjectiveBase:
    name = None
    greater_is_better = True
    score_needs_proba = False
    uses_extra_columns = False
    problem_types = []

    def __init__(self, verbose=False):
        self.verbose = verbose

    def objective_function(self, y_predicted, y_true, X=None):
        raise NotImplementedError

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
