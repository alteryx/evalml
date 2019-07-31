from scipy.optimize import minimize_scalar


class LeadScoring:
    def __init__(self, label=1, true_positives=1, false_positives=1, threshold=.5, verbose=True):
        self.label = label
        self.weights = {
            'true_positives': true_positives,
            'false_positives': -false_positives,
        }
        self.threshold = threshold
        self.verbose = verbose

    def fit(self, y_prob, y):
        def cost(threshold):
            self.threshold = threshold
            return -self.score(y, y_prob)

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

    def score(self, y, y_prob):
        y_label = y.eq(self.label).astype(int)
        y_hat_label = self.predict(y_prob)

        true_positives = (y_label & y_hat_label).sum()
        false_positives = (~y_label & y_hat_label).sum()

        loss = self.weights['true_positives'] * true_positives
        loss += self.weights['false_positives'] * false_positives

        return loss
