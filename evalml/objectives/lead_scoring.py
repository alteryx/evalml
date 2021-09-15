"""Lead scoring objective."""
import math

from .binary_classification_objective import BinaryClassificationObjective


class LeadScoring(BinaryClassificationObjective):
    """Lead scoring.

    Args:
        true_positives (int): Reward for a true positive. Defaults to 1.
        false_positives (int): Cost for a false positive. Should be negative. Defaults to -1.
    """

    name = "Lead Scoring"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = math.inf
    is_bounded_like_percentage = False  # Range (-Inf, Inf)
    expected_range = [float("-inf"), float("inf")]

    def __init__(self, true_positives=1, false_positives=-1):
        self.true_positives = true_positives
        self.false_positives = false_positives

    def objective_function(self, y_true, y_predicted, X=None, sample_weight=None):
        """Calculate the profit per lead.

        Args:
            y_predicted (pd.Series): Predicted labels
            y_true (pd.Series): True labels
            X (pd.DataFrame): Ignored.
            sample_weight (pd.DataFrame): Ignored.

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
        # penalty if our estimator only predicts 1 output by making the score 0
        same_class_penalty = (2 - len(set(y_predicted))) * abs(profit_per_lead)

        return profit_per_lead - same_class_penalty
