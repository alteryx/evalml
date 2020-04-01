import pandas as pd

from .binary_classification_objective import BinaryClassificationObjective


class LeadScoring(BinaryClassificationObjective):
    """Lead scoring"""
    name = "Lead Scoring"
    greater_is_better = True
    score_needs_proba = False

    def __init__(self, true_positives=1, false_positives=-1):
        """Create instance.

        Arguments:
            true_positives (int) : reward for a true positive
            false_positives (int) : cost for a false positive. Should be negative.
        """
        self.true_positives = true_positives
        self.false_positives = false_positives
        super().__init__()

    def objective_function(self, y_predicted, y_true, X=None):
        """Calculate the profit per lead.

            Arguments:
                y_predicted (pd.Series): predicted labels
                y_true (pd.Series): true labels
                X (pd.DataFrame): None, not used.

            Returns:
                float: profit per lead
        """
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
