from .objective_base import ObjectiveBase


class FraudDetection(ObjectiveBase):
    needs_fitting = True
    greater_is_better = False
    uses_extra_columns = True
    needs_proba = True
    name = "Fraud Detection"
    """Finds the optimal threshold for fraud detection."""

    def __init__(self, label=1, retry_percentage=.5, interchange_fee=.02,
                 fraud_payout_percentage=1.0, amount_col='amount', verbose=True):
        """Create instance.

        Args:
            label (int) : label to optimize threshold for
            verbose (bool) : whether to print while optimizing threshold
        """
        self.label = label
        self.retry_percentage = retry_percentage
        self.interchange_fee = interchange_fee
        self.fraud_payout_percentage = fraud_payout_percentage
        self.amount_col = amount_col
        self.verbose = verbose

    def decision_function(self, y_predicted, extra_cols, threshold):
        return (y_predicted * extra_cols[self.amount_col]) > threshold

    def objective_function(self, y_predicted, y_true, extra_cols):
        fraud_cost = extra_cols[self.amount_col] * self.fraud_payout_percentage

        interchange_cost = extra_cols[self.amount_col] * (1 - self.retry_percentage) * self.interchange_fee

        # payout fraud
        false_negatives = (y_true & ~y_predicted) * fraud_cost

        # lost fees
        false_positives = (~y_true & y_predicted) * interchange_cost

        loss = false_negatives.sum() + false_positives.sum()

        return loss
