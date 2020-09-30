import pandas as pd

from .data_check import DataCheck
from .data_check_message import DataCheckWarning


class HighVarianceCVDataCheck(DataCheck):
    """Checks if the variance between folds in cross-validation is higher than an acceptable threshhold."""

    def __init__(self, threshold=0.2):
        """Check if there is higher variance among cross-validation results.

        Arguments:
            threshold (float): The minimum threshold allowed for high variance before a warning is raised.
                Defaults to 0.2 and must be above 0.
        """
        if threshold <= 0:
            raise ValueError(f"Provided threshold {threshold} is greater than 0.")
        self.threshold = threshold

    def validate(self, cv_scores):
        """Checks cross-validation scores and issues an warning if variance is higher than specified threshhold.

        Arguments:
            cv_scores (pd.Series, np.array, list): list of scores of each cross-validation fold

        Returns:
            list (DataCheckWarning): list with DataCheckWarnings if imbalance in classes is less than the threshold.

        Example:
            >>> cv_scores = pd.Series([0, 1, 1, 1])
            >>> target_check = HighVarianceCVDataCheck(threshold=0.10)
            >>> assert target_check.validate(cv_scores) == [DataCheckWarning("High variance (variance >= 0.2) within cross validation scores. Model may not perform as estimated on unseen data.", "HighVarianceCVDataCheck")]
        """
        if not isinstance(cv_scores, pd.Series):
            cv_scores = pd.Series(cv_scores)

        messages = []
        high_variance_cv = (cv_scores.std() / cv_scores.mean()) > self.threshold
        # if there are items that occur less than the threshold, add them to the list of messages
        if len(high_variance_cv):
            warning_msg = "High variance (variance >= {self.threshhold}) within cross validation scores. Model may not perform as estimated on unseen data."
            messages.append(DataCheckWarning(warning_msg, self.name))
        return messages
