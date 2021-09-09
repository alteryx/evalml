"""Score the percentage of money lost of the total transaction amount process due to fraud."""
from .binary_classification_objective import BinaryClassificationObjective


class FraudCost(BinaryClassificationObjective):
    """Score the percentage of money lost of the total transaction amount process due to fraud.

    Args:
        retry_percentage (float): What percentage of customers that will retry a transaction if it
            is declined. Between 0 and 1. Defaults to 0.5.
        interchange_fee (float): How much of each successful transaction you pay.
            Between 0 and 1. Defaults to 0.02.
        fraud_payout_percentage (float): Percentage of fraud you will not be able to collect.
            Between 0 and 1. Defaults to 1.0.
        amount_col (str): Name of column in data that contains the amount. Defaults to "amount".
    """

    name = "Fraud Cost"
    greater_is_better = False
    score_needs_proba = False
    perfect_score = 0.0
    is_bounded_like_percentage = True
    expected_range = [0, float("inf")]

    def __init__(
        self,
        retry_percentage=0.5,
        interchange_fee=0.02,
        fraud_payout_percentage=1.0,
        amount_col="amount",
    ):
        self.retry_percentage = retry_percentage
        self.interchange_fee = interchange_fee
        self.fraud_payout_percentage = fraud_payout_percentage
        self.amount_col = amount_col

    def objective_function(self, y_true, y_predicted, X, sample_weight=None):
        """Calculate amount lost to fraud per transaction given predictions, true values, and dataframe with transaction amount.

        Args:
            y_predicted (pd.Series): Predicted fraud labels.
            y_true (pd.Series): True fraud labels.
            X (pd.DataFrame): Data with transaction amounts.
            sample_weight (pd.DataFrame): Ignored.

        Returns:
            float: Amount lost to fraud per transaction.

        Raises:
            ValueError: If amount_col is not a valid column in the input data.
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

        # money paid from interchange fees on transaction
        interchange_cost = (
            transaction_amount * (1 - self.retry_percentage) * self.interchange_fee
        )

        # calculate cost of missing fraudulent transactions
        false_negatives = (y_true & ~y_predicted) * fraud_cost

        # calculate money lost from fees
        false_positives = (~y_true & y_predicted) * interchange_cost

        # add a penalty if we output naive predictions
        all_one_prediction_cost = (2 - len(set(y_predicted))) * fraud_cost.sum()
        loss = false_negatives.sum() + false_positives.sum() + all_one_prediction_cost

        loss_per_total_processed = loss / transaction_amount.sum()

        return loss_per_total_processed
