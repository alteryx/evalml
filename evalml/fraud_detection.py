from scipy.optimize import minimize_scalar


class FraudDetection():
    def __init__(self, label=1, retry_percentage=.5, interchange_fee=.02,
                 fraud_payout_percentage=1.0, threshold=.5, verbose=True):
        self.label = label
        self.retry_percentage = retry_percentage
        self.interchange_fee = interchange_fee
        self.fraud_payout_percentage = fraud_payout_percentage
        self.threshold = threshold
        self.verbose = verbose

    def fit(self, y_prob, y, value):
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
        return y_prob[self.label].gt(self.threshold).astype(int)

    def score(self, y, y_prob, value):
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
