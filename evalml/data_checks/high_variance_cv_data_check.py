import pandas as pd

from evalml.data_checks import (
    DataCheck,
    DataCheckMessageCode,
    DataCheckWarning
)


class HighVarianceCVDataCheck(DataCheck):
    """Checks if the variance between folds in cross-validation is higher than an acceptable threshold."""

    def __init__(self, threshold=0.2):
        """Check if there is higher variance among cross-validation results.

        Arguments:
            threshold (float): The minimum threshold allowed for high variance before a warning is raised.
                Defaults to 0.2 and must be above 0.
        """
        if threshold < 0:
            raise ValueError(f"Provided threshold {threshold} needs to be greater than 0.")
        self.threshold = threshold

    def validate(self, pipeline_name, cv_scores):
        """Checks cross-validation scores and issues an warning if variance is higher than specified threshhold.

        Arguments:
            pipeline_name (str): name of pipeline that produced cv_scores
            cv_scores (pd.Series, np.ndarray, list): list of scores of each cross-validation fold

        Returns:
            dict: Dictionary with DataCheckWarnings if imbalance in classes is less than the threshold.

        Example:
            >>> cv_scores = pd.Series([0, 1, 1, 1])
            >>> check = HighVarianceCVDataCheck(threshold=0.10)
            >>> assert check.validate("LogisticRegressionPipeline", cv_scores) == {"warnings": [{"message": "High coefficient of variation (cv >= 0.1) within cross validation scores. LogisticRegressionPipeline may not perform as estimated on unseen data.",\
                                                                                                 "data_check_name": "HighVarianceCVDataCheck",\
                                                                                                 "level": "warning",\
                                                                                                 "code": "HIGH_VARIANCE",\
                                                                                                 "details": {"variance": 2.0/3.0, "pipeline_name": "LogisticRegressionPipeline"}}],\
                                                                                   "errors": []}
        """
        messages = {
            "warnings": [],
            "errors": []
        }
        if not isinstance(cv_scores, pd.Series):
            cv_scores = pd.Series(cv_scores)

        variance = 0
        if cv_scores.mean() == 0:
            high_variance_cv = False
        else:
            variance = abs(cv_scores.std() / cv_scores.mean())
            high_variance_cv = abs(cv_scores.std() / cv_scores.mean()) > self.threshold
        # if there are items that occur less than the threshold, add them to the list of messages
        if high_variance_cv:
            warning_msg = f"High coefficient of variation (cv >= {self.threshold}) within cross validation scores. {pipeline_name} may not perform as estimated on unseen data."
            DataCheck._add_message(DataCheckWarning(message=warning_msg,
                                                    data_check_name=self.name,
                                                    message_code=DataCheckMessageCode.HIGH_VARIANCE,
                                                    details={"variance": variance, "pipeline_name": pipeline_name}), messages)
        return messages
