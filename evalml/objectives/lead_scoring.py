from .objective_base import ObjectiveBase


class LeadScoring(ObjectiveBase):
    """Lead scoring"""
    name = "Lead Scoring"
    needs_fitting = True
    greater_is_better = True
    fit_needs_proba = True
    score_needs_proba = False
    name = "Lead Scoring"

    def __init__(self, true_positives=1, false_positives=-1, verbose=False):
        """Create instance.

        Args:
            label (int) : label to optimize threshold for
            true_positives (int) : reward for a true positive
            false_positives (int) : cost for a false positive. Should be negative.
        """
        self.true_positives = true_positives
        self.false_positives = false_positives

        super().__init__(verbose=verbose)

    def decision_function(self, y_predicted, threshold):
        return y_predicted > threshold

    def objective_function(self, y_predicted, y_true):
        true_positives = (y_true & y_predicted).sum()
        false_positives = (~y_true & y_predicted).sum()

        profit = self.true_positives * true_positives
        profit += self.false_positives * false_positives

        profit_per_lead = profit / len(y_true)

        return profit_per_lead
