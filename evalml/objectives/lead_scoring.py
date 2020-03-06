import pandas as pd

from .binary_classification_objective import BinaryClassificationObjective

from evalml.problem_types import ProblemTypes


class LeadScoring(BinaryClassificationObjective):
    """Lead scoring"""
    name = "Lead Scoring"
    problem_type = ProblemTypes.BINARY

    greater_is_better = True
    name = "Lead Scoring"

    def __init__(self, true_positives=1, false_positives=-1):
        """Create instance.

        Arguments:
            label (int) : label to optimize threshold for
            true_positives (int) : reward for a true positive
            false_positives (int) : cost for a false positive. Should be negative.
        """
        self.true_positives = true_positives
        self.false_positives = false_positives

        super().__init__()

    def decision_function(self, y_predicted, threshold, X=None):
        #  TODO: necessary?
        if not isinstance(y_predicted, pd.Series):
            y_predicted = pd.Series(y_predicted)

        return y_predicted > threshold

    def objective_function(self, y_predicted, y_true, X=None):
        if not isinstance(y_predicted, pd.Series):
            y_predicted = pd.Series(y_predicted)

        if not isinstance(y_true, pd.Series):
            y_true = pd.Series(y_true)

        true_positives = (y_true & y_predicted).sum()
        false_positives = (~y_true & y_predicted).sum()

        profit = self.true_positives * true_positives
        profit += self.false_positives * false_positives

        profit_per_lead = profit / len(y_true)

        return profit_per_lead
