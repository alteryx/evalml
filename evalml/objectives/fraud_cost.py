import pandas as pd

from .binary_classification_objective import BinaryClassificationObjective

from evalml.problem_types import ProblemTypes


class FraudCost(BinaryClassificationObjective):
    """Score the percentage of money lost of the total transaction amount process due to fraud"""
    name = "Fraud Cost"
    problem_types = [ProblemTypes.BINARY]
    greater_is_better = False

    def __init__(self, retry_percentage=.5, interchange_fee=.02,
                 fraud_payout_percentage=1.0, amount_col='amount', verbose=False):
        """Create instance of FraudCost

        Arguments:
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

    def decision_function(self, ypred_proba, classification_threshold, X):
        """Determine if transaction is fraud given predicted probabilities, dataframe with transaction amount, and threshold

            Arguments:
                ypred_proba (pd.Series): predicted labels
                X (pd.DataFrame): extra data needed
                classification_threshold (float): dollar threshold to determine if transaction is fraud

            Returns:
                pd.Series: series of predicted fraud label using extra cols and classification_threshold
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if not isinstance(ypred_proba, pd.Series):
            ypred_proba = pd.Series(ypred_proba)

        transformed_probs = (ypred_proba.values * X[self.amount_col])
        return transformed_probs > classification_threshold

    def objective_function(self, y_predicted, y_true, X):
        """Calculate amount lost to fraud per transaction given predictions, true values, and dataframe with transaction amount

            Arguments:
                y_predicted (pd.Series): predicted fraud labels
                y_true (pd.Series): true fraud labels
                X (pd.DataFrame): extra data needed

            Returns:
                float: amount lost to fraud per transaction
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if not isinstance(y_predicted, pd.Series):
            y_predicted = pd.Series(y_predicted)

        if not isinstance(y_true, pd.Series):
            y_true = pd.Series(y_true)

        # extract transaction using the amount columns in users data
        transaction_amount = X[self.amount_col]

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
