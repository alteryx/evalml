from scipy.optimize import minimize_scalar

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

    def fit(self, y_predicted, y_true):
        """Optimize threshold on probability estimates of the label.

        Args:
            y_predicted (DataFrame) : probability estimates of labels in train set
            y_true (DataFrame) : true labels in train set

        Returns:
            LeadScoring : instance of self
        """

        if self.verbose:
            print('Searching for optimal threshold.')

        def cost(threshold):
            return -self.score_for_threshold(y_predicted, y_true, threshold)

        self.optimal = minimize_scalar(cost, bounds=(0, 1), method='Bounded')

        if self.verbose:
            info = 'Optimal threshold found at {:.2f}'
            print(info.format(self.optimal.x))

        self.threshold = self.optimal.x
        return self

    def predict(self, y_predicted):
        """Predicts using the optimized threshold.

        Args:
            y_predicted (DataFrame) : probability estimates for each label

        Returns:
            Series : estimated labels using optimized threshold
        """
        return y_predicted > self.threshold

    def score(self, y_predicted, y_true):
        """The cost function for threshold-based predictions.

        Args:
            y_predicted (DataFrame) : probability estimates of labels
            y_true (DataFrame) : true labels
        """
        return self.score_for_threshold(y_predicted, y_true, self.threshold)

    def score_for_threshold(self, y_predicted, y_true, threshold):
        y_predicted = y_predicted

        y_hat_label = y_predicted > threshold

        true_positives = (y_true & y_hat_label).sum()
        false_positives = (~y_true & y_hat_label).sum()

        loss = self.true_positives * true_positives
        loss += self.false_positives * false_positives

        return loss
