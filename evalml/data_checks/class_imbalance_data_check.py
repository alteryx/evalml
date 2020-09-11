import pandas as pd

from .data_check import DataCheck
from .data_check_message import DataCheckWarning


class ClassImbalanceDataCheck(DataCheck):
    """Classification problems, checks if any target labels are imbalanced beyond a threshold"""

    def validate(self, X, y, threshold=0.10):
        """Checks if any target labels are imbalanced beyond a threshold for binary and multiclass problems
        Ignores nan values in target labels if they appear

        Arguments:
            X (pd.DataFrame, pd.Series, np.array, list): Features. Ignored.
            y: Target labels to check for imbalanced data.
            threshold (float, optional): The minimum threshold allowed for class imbalance before a warning is raised.
                                        A perfectly balanced dataset would have a threshold of (1/n_classes), ie 0.50 for binary classes.
                                        Defaults to 0.10

        Returns:
            list (DataCheckWarning): list with DataCheckWarnings if imbalance in classes is less than the threshold.

        Example:
            >>> X = pd.DataFrame({})
            >>> y = pd.Series([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
            >>> threshold = 0.10
            >>> target_check = ClassImbalanceDataCheck()
            >>> assert target_check.validate(X, y, threshold) == [DataCheckWarning("The following labels fall below 10% of the target: [0]", "ClassImbalanceDataCheck")]
        """
        if threshold <= 0 or threshold > 0.5:
            raise ValueError("Provided threshold {} is not within the range (0, 0.5]".format(threshold))

        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        messages = []
        counts = y.value_counts(normalize=True)
        below_threshold = counts.where(counts < threshold).dropna()
        # if there are items that occur less than the threshold, add them to the list of messages
        if len(below_threshold):
            warning_msg = "The following labels fall below {:.0f}% of the target: {}"
            messages.append(DataCheckWarning(warning_msg.format(threshold * 100, below_threshold.index.tolist()), self.name))
        return messages
