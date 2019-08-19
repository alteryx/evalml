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

    def fit(self, y_prob, y):
        """Optimize threshold on probability estimates of the label.

        Args:
            y_prob (DataFrame) : probability estimates of labels in train set
            y (DataFrame) : true labels in train set

        Returns:
            LeadScoring : instance of self
        """

        if self.verbose:
            print('Searching for optimal threshold.')

        def cost(threshold):
            return -self.score_for_threshold(y, y_prob, threshold)

        self.optimal = minimize_scalar(cost, bounds=(0, 1), method='Bounded')

        if self.verbose:
            info = 'Optimal threshold found at {:.2f}'
            print(info.format(self.optimal.x))

        self.threshold = self.optimal.x
        return self

    def predict(self, y_prob):
        """Predicts using the optimized threshold.

        Args:
            y_prob (DataFrame) : probability estimates for each label

        Returns:
            Series : estimated labels using optimized threshold
        """
        return y_prob > self.threshold

    def score(self, y_true, y_prob):
        """The cost function for threshold-based predictions.

        Args:
            y (DataFrame) : true labels
            y_prob (DataFrame) : probability estimates of labels
        """
        return self.score_for_threshold(y_true, y_prob, self.threshold)

    def score_for_threshold(self, y_true, y_prob, threshold):
        y_prob = y_prob[:, 1]  # get true column

        y_hat_label = y_prob > threshold

        true_positives = (y_true & y_hat_label).sum()
        false_positives = (~y_true & y_hat_label).sum()

        loss = self.true_positives * true_positives
        loss += self.false_positives * false_positives

        return loss
