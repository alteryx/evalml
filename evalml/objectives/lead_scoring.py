import math

from .binary_classification_objective import BinaryClassificationObjective


class LeadScoring(BinaryClassificationObjective):
    """Lead scoring."""
    name = "Lead Scoring"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = math.inf
    is_bounded_like_percentage = False  # Range (-Inf, Inf)

    def __init__(self, true_positives=1, false_positives=-1):
        """Create instance.

        Arguments:
            true_positives (int): Reward for a true positive
            false_positives (int): Cost for a false positive. Should be negative.
        """
        self.true_positives = true_positives
        self.false_positives = false_positives

    def objective_function(self, y_true, y_predicted, X=None):
        """Calculate the profit per lead.

            Arguments:
                y_predicted (ww.DataColumn, pd.Series): Predicted labels
                y_true (ww.DataColumn, pd.Series): True labels
                X (ww.DataTable, pd.DataFrame): Ignored.

            Returns:
                float: Profit per lead
        """
        y_true = self._standardize_input_type(y_true)
        y_predicted = self._standardize_input_type(y_predicted)

        true_positives = (y_true & y_predicted).sum()
        false_positives = (~y_true & y_predicted).sum()

        profit = self.true_positives * true_positives
        profit += self.false_positives * false_positives

        profit_per_lead = profit / len(y_true)

        return profit_per_lead
