
from .binary_classification_objective import BinaryClassificationObjective


class FraudCost(BinaryClassificationObjective):
    """Score the percentage of money lost of the total transaction amount process due to fraud."""
    name = "Fraud Cost"
    greater_is_better = False
    score_needs_proba = False
    perfect_score = 0.0
    is_bounded_like_percentage = True

    def __init__(self, retry_percentage=.5, interchange_fee=.02,
                 fraud_payout_percentage=1.0, amount_col='amount'):
        """Create instance of FraudCost

        Arguments:
            retry_percentage (float): What percentage of customers that will retry a transaction if it
                is declined. Between 0 and 1. Defaults to .5

            interchange_fee (float): How much of each successful transaction you can collect.
                Between 0 and 1. Defaults to .02

            fraud_payout_percentage (float): Percentage of fraud you will not be able to collect.
                Between 0 and 1. Defaults to 1.0

            amount_col (str): Name of column in data that contains the amount. Defaults to "amount"
        """
        self.retry_percentage = retry_percentage
        self.interchange_fee = interchange_fee
        self.fraud_payout_percentage = fraud_payout_percentage
        self.amount_col = amount_col

    def decision_function(self, ypred_proba, threshold=0.0, X=None):
        """Determine if a transaction is fraud given predicted probabilities, threshold, and dataframe with transaction amount.

        Arguments:
            ypred_proba (ww.DataColumn, pd.Series): Predicted probablities
            threshold (float): Dollar threshold to determine if transaction is fraud
            X (ww.DataTable, pd.DataFrame): Data containing transaction amounts

        Returns:
            pd.Series: pd.Series of predicted fraud labels using X and threshold
        """
        if X is not None:
            X = self._standardize_input_type(X)
        ypred_proba = self._standardize_input_type(ypred_proba)
        transformed_probs = (ypred_proba.values * X[self.amount_col])
        return transformed_probs > threshold

    def objective_function(self, y_true, y_predicted, X):
        """Calculate amount lost to fraud per transaction given predictions, true values, and dataframe with transaction amount.

        Arguments:
            y_predicted (ww.DataColumn, pd.Series): Predicted fraud labels
            y_true (ww.DataColumn, pd.Series): True fraud labels
            X (ww.DataTable, pd.DataFrame): Data with transaction amounts

        Returns:
            float: Amount lost to fraud per transaction
        """
        X = self._standardize_input_type(X)
        y_true = self._standardize_input_type(y_true)
        y_predicted = self._standardize_input_type(y_predicted)
        self.validate_inputs(y_true, y_predicted)

        # extract transaction using the amount columns in users data
        try:
            transaction_amount = X[self.amount_col]
        except KeyError:
            raise ValueError("`{}` is not a valid column in X.".format(self.amount_col))

        # amount paid if transaction is fraud
        fraud_cost = transaction_amount * self.fraud_payout_percentage

        # money made from interchange fees on transaction
        interchange_cost = transaction_amount * (1 - self.retry_percentage) * self.interchange_fee

        # calculate cost of missing fraudulent transactions
        false_negatives = (y_true & ~y_predicted) * fraud_cost

        # calculate money lost from fees
        false_positives = (~y_true & y_predicted) * interchange_cost

        loss = false_negatives.sum() + false_positives.sum()

        loss_per_total_processed = loss / transaction_amount.sum()

        return loss_per_total_processed
