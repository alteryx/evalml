from scipy.optimize import minimize_scalar


class LeadScoring:
    """Finds the optimal threshold for probability estimates of a label."""

    def __init__(self, label=1, true_positives=1, false_positives=1, threshold=.5, verbose=True):
        """Create instance.

        Args:
            label (int) : label to optimize threshold for
            true_positives (int) : reward for a true positive
            false_positives (int) : cost for a false positive
            threshold (float) : starting value for threshold
            verbose (bool) : whether to print while optimizing threshold
        """
        self.label = label
        self.weights = {
            'true_positives': true_positives,
            'false_positives': -false_positives,
        }
        self.threshold = threshold
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
            self.threshold = threshold
            return -self.score(y, y_prob)

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
        return y_prob[self.label].gt(self.threshold).astype(int)

    def score(self, y, y_prob):
        """The cost function for threshold-based predictions.

        Args:
            y (DataFrame) : true labels
            y_prob (DataFrame) : probability estimates of labels
        """
        y_label = y.eq(self.label).astype(int)
        y_hat_label = self.predict(y_prob)

        true_positives = (y_label & y_hat_label).sum()
        false_positives = (~y_label & y_hat_label).sum()

        loss = self.weights['true_positives'] * true_positives
        loss += self.weights['false_positives'] * false_positives

        return loss
