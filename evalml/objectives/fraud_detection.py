from .objective_base import ObjectiveBase


class FraudDetection(ObjectiveBase):
    """Score the money lost to transactional fraud"""
    name = "Fraud Detection"
    needs_fitting = True
    greater_is_better = False
    uses_extra_columns = True
    needs_proba = True

    def __init__(self, retry_percentage=.5, interchange_fee=.02,
                 fraud_payout_percentage=1.0, amount_col='amount', verbose=False):
        """Create instance of FraudDetection

        Args:
            retry_percentage (float): what percentage of customers will retry a transaction if it
                is declined? Between 0 and 1. Defaults to .5

            interchange_fee (float): how much of each successful transaction do you collect?
                Between 0 and 1. Defaults to .02

            fraud_payout_percentage (float):  how percentage of fraud will you be unable to collect.
                Between 0 and 1. Defaults to 1.0

            amount_col (str): name of column in data that contains the amount. defaults to "amount"
        """
        self.retry_percentage = retry_percentage
        self.interchange_fee = interchange_fee
        self.fraud_payout_percentage = fraud_payout_percentage
        self.amount_col = amount_col
        super().__init__(verbose=verbose)

    def decision_function(self, y_predicted, extra_cols, threshold):
        """Determine if transaction is fraud given predicted probabilities,
            dataframe with transaction amount, and threshold"""

        transformed_probs = (y_predicted * extra_cols[self.amount_col])
        return transformed_probs > threshold

    def objective_function(self, y_predicted, y_true, extra_cols):
        """Calculate amount lost to fraud given predictions, true values, and dataframe
            with transaction amount"""

        # extract transaction using the amount columns in users data
        transaction_amount = extra_cols[self.amount_col]

        # amount paid if transaction is fraud
        fraud_cost = transaction_amount * self.fraud_payout_percentage

        # money made from interchange fees on transaction
        interchange_cost = transaction_amount * (1 - self.retry_percentage) * self.interchange_fee

        # calculate cost of missing fraudulent transactions
        false_negatives = (y_true & ~y_predicted) * fraud_cost

        # calculate money lost from fees
        false_positives = (~y_true & y_predicted) * interchange_cost

        loss = false_negatives.sum() + false_positives.sum()

        return loss
