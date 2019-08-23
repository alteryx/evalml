from .objective_base import ObjectiveBase


class LeadScoring(ObjectiveBase):
    """Finds the optimal threshold for probability estimates of a label."""
    needs_fitting = True
    greater_is_better = True
    needs_proba = True
    name = "Lead Scoring"

    def __init__(self, true_positives=1, false_positives=1, verbose=True):
        """Create instance.

        Args:
            label (int) : label to optimize threshold for
            true_positives (int) : reward for a true positive
            false_positives (int) : cost for a false positive
            verbose (bool) : whether to print while optimizing threshold
        """
        self.true_positives = true_positives
        self.false_positives = false_positives
        self.verbose = verbose

    def decision_function(self, y_predicted, threshold):
        return y_predicted > threshold

    def objective_function(self, y_predicted, y_true):
        true_positives = (y_true & y_predicted).sum()
        false_positives = (~y_true & y_predicted).sum()

        profit = self.true_positives * true_positives
        profit += self.false_positives * false_positives

        return profit
