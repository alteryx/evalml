import pandas as pd

from .data_check import DataCheck
from .data_check_message import DataCheckError, DataCheckWarning
from .data_check_message_type import DataCheckMessageType


class ClassImbalanceDataCheck(DataCheck):
    """Checks if any target labels are imbalanced beyond a threshold. Use for classification problems"""

    def __init__(self, threshold=0.1, num_cv_folds=3):
        """Check if any of the target labels are imbalanced, or if the number of values for each target
           are below 2 times the number of cv folds

        Arguments:
            threshold (float): The minimum threshold allowed for class imbalance before a warning is raised.
                A perfectly balanced dataset would have a threshold of (1/n_classes), ie 0.50 for binary classes.
                Defaults to 0.10
            num_cv_folds (int): The number of cross-validation folds. Must be positive. Choose 0 to ignore this warning.
        """
        if threshold <= 0 or threshold > 0.5:
            raise ValueError("Provided threshold {} is not within the range (0, 0.5]".format(threshold))
        self.threshold = threshold
        if num_cv_folds < 0:
            raise ValueError("Provided number of CV folds {} is less than 0".format(num_cv_folds))
        self.cv_folds = num_cv_folds * 2

    def validate(self, X, y):
        """Checks if any target labels are imbalanced beyond a threshold for binary and multiclass problems
        Ignores nan values in target labels if they appear

        Arguments:
            X (pd.DataFrame, pd.Series, np.ndarray, list): Features. Ignored.
            y: Target labels to check for imbalanced data.

        Returns:
            dict: Dictionary with DataCheckWarnings if imbalance in classes is less than the threshold,
                  and DataCheckErrors if the number of values for each target is below 2 * num_cv_folds.

        Example:
            >>> X = pd.DataFrame({})
            >>> y = pd.Series([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
            >>> target_check = ClassImbalanceDataCheck(threshold=0.10)
            >>> assert target_check.validate(X, y) == {DataCheckMessageType.ERROR: DataCheckError("The number of instances of these targets is less than 2 * the number of cross folds = 6 instances: [0]", "ClassImbalanceDataCheck"),
                                                       DataCheckMessageType.WARNING: DataCheckWarning("The following labels fall below 10% of the target: [0]", "ClassImbalanceDataCheck")]
        """
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        messages = {
            DataCheckMessageType.WARNING: [],
            DataCheckMessageType.ERROR: []
        }
        fold_counts = y.value_counts(normalize=False)
        # search for targets that occur less than twice the number of cv folds first
        below_threshold_folds = fold_counts.where(fold_counts < self.cv_folds).dropna()
        if len(below_threshold_folds):
            error_msg = "The number of instances of these targets is less than 2 * the number of cross folds = {} instances: {}"
            messages[DataCheckMessageType.ERROR].append(DataCheckError(error_msg.format(self.cv_folds, below_threshold_folds.index.tolist()), self.name))

        counts = fold_counts / fold_counts.sum()
        below_threshold = counts.where(counts < self.threshold).dropna()
        # if there are items that occur less than the threshold, add them to the list of messages
        if len(below_threshold):
            warning_msg = "The following labels fall below {:.0f}% of the target: {}"
            messages[DataCheckMessageType.WARNING].append(DataCheckWarning(warning_msg.format(self.threshold * 100, below_threshold.index.tolist()), self.name))
        return messages
