from scipy.optimize import minimize_scalar


class FraudDetection():
    """Finds the optimal threshold for fraud detection."""
    def __init__(self, label=1, retry_percentage=.5, interchange_fee=.02,
                 fraud_payout_percentage=1.0, threshold=.5, verbose=True):
        """Create instance.

        Args:
            label (int) : label to optimize threshold for
            verbose (bool) : whether to print while optimizing threshold
        """
        self.label = label
        self.retry_percentage = retry_percentage
        self.interchange_fee = interchange_fee
        self.fraud_payout_percentage = fraud_payout_percentage
        self.threshold = threshold
        self.verbose = verbose

    def fit(self, y_prob, y, value):
        """Optimize threshold on probability estimates of the label.

        Args:
            y_prob (DataFrame) : probability estimates of labels in train set
            y (DataFrame) : true labels in train set

        Returns:
            LeadScoring : instance of self
        """
        def cost(threshold):
            self.threshold = threshold
            return -self.score(y, y_prob, value)

        if self.verbose:
            print('Searching for optimal threshold.')

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

    def score(self, y, y_prob, value):
        """The cost function for threshold-based predictions.

        Args:
            y (DataFrame) : true labels
            y_prob (DataFrame) : probability estimates of labels
        """
        y_label = y.eq(self.label).astype(int)

        fraud_cost = value * self.fraud_payout_percentage
        interchange_cost = value * (1 - self.retry_percentage) * self.interchange_fee

        y_hat_label = self.predict(y_prob)

        # payout fraud
        false_negatives = (y_label & ~y_hat_label) * fraud_cost

        # lost fees
        false_positives = (~y_label & y_hat_label) * interchange_cost

        loss = false_negatives.sum() + false_positives.sum()

        return loss
