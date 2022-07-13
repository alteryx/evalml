"""Data check that checks if any of the target labels are imbalanced, or if the number of values for each target are below 2 times the number of CV folds.

Use for classification problems.
"""
import numpy as np

from evalml.data_checks import (
    DataCheck,
    DataCheckError,
    DataCheckMessageCode,
    DataCheckWarning,
)
from evalml.utils import infer_feature_types


class ClassImbalanceDataCheck(DataCheck):
    """Check if any of the target labels are imbalanced, or if the number of values for each target are below 2 times the number of CV folds. Use for classification problems.

    Args:
        threshold (float): The minimum threshold allowed for class imbalance before a warning is raised.
            This threshold is calculated by comparing the number of samples in each class to the sum of samples in that class and the majority class.
            For example, a multiclass case with [900, 900, 100] samples per classes 0, 1, and 2, respectively,
            would have a 0.10 threshold for class 2 (100 / (900 + 100)). Defaults to 0.10.
        min_samples (int): The minimum number of samples per accepted class. If the minority class is both below the threshold and min_samples,
            then we consider this severely imbalanced. Must be greater than 0. Defaults to 100.
        num_cv_folds (int): The number of cross-validation folds. Must be positive. Choose 0 to ignore this warning. Defaults to 3.
        test_size (None, float, int): Percentage of test set size. Used to calculate class imbalance prior to splitting the
            data into training and validation/test sets.

    Raises:
        ValueError: If threshold is not within 0 and 0.5
        ValueError: If min_samples is not greater than 0
        ValueError: If number of cv folds is negative
        ValueError: If test_size is not between 0 and 1
    """

    def __init__(self, threshold=0.1, min_samples=100, num_cv_folds=3, test_size=None):
        if threshold <= 0 or threshold > 0.5:
            raise ValueError(
                "Provided threshold {} is not within the range (0, 0.5]".format(
                    threshold,
                ),
            )
        self.threshold = threshold
        if min_samples <= 0:
            raise ValueError(
                "Provided value min_samples {} is not greater than 0".format(
                    min_samples,
                ),
            )
        self.min_samples = min_samples
        if num_cv_folds < 0:
            raise ValueError(
                "Provided number of CV folds {} is less than 0".format(num_cv_folds),
            )
        self.cv_folds = num_cv_folds * 2
        if test_size is not None:
            if not (isinstance(test_size, (int, float)) and 0 < test_size <= 1):
                raise ValueError(
                    "Parameter test_size must be a number between 0 and less than or equal to 1",
                )
            self.test_size = test_size
        else:
            self.test_size = 1

    def validate(self, X, y):
        """Check if any target labels are imbalanced beyond a threshold for binary and multiclass problems.

        Ignores NaN values in target labels if they appear.

        Args:
            X (pd.DataFrame, np.ndarray): Features. Ignored.
            y (pd.Series, np.ndarray): Target labels to check for imbalanced data.

        Returns:
            dict: Dictionary with DataCheckWarnings if imbalance in classes is less than the threshold,
                  and DataCheckErrors if the number of values for each target is below 2 * num_cv_folds.

        Examples:
            >>> import pandas as pd
            ...
            >>> X = pd.DataFrame()
            >>> y = pd.Series([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

            In this binary example, the target class 0 is present in fewer than 10% (threshold=0.10) of instances, and fewer than 2 * the number
            of cross folds (2 * 3 = 6). Therefore, both a warning and an error are returned as part of the Class Imbalance Data Check.
            In addition, if a target is present with fewer than `min_samples` occurrences (default is 100) and is under the threshold,
            a severe class imbalance warning will be raised.

            >>> class_imb_dc = ClassImbalanceDataCheck(threshold=0.10)
            >>> assert class_imb_dc.validate(X, y) == [
            ...     {
            ...         "message": "The number of instances of these targets is less than 2 * the number of cross folds = 6 instances: [0]",
            ...         "data_check_name": "ClassImbalanceDataCheck",
            ...         "level": "error",
            ...         "code": "CLASS_IMBALANCE_BELOW_FOLDS",
            ...         "details": {"target_values": [0], "rows": None, "columns": None},
            ...         "action_options": []
            ...     },
            ...     {
            ...         "message": "The following labels fall below 10% of the target: [0]",
            ...         "data_check_name": "ClassImbalanceDataCheck",
            ...         "level": "warning",
            ...         "code": "CLASS_IMBALANCE_BELOW_THRESHOLD",
            ...         "details": {"target_values": [0], "rows": None, "columns": None},
            ...         "action_options": []
            ...     },
            ...     {
            ...         "message": "The following labels in the target have severe class imbalance because they fall under 10% of the target and have less than 100 samples: [0]",
            ...         "data_check_name": "ClassImbalanceDataCheck",
            ...         "level": "warning",
            ...         "code": "CLASS_IMBALANCE_SEVERE",
            ...         "details": {"target_values": [0], "rows": None, "columns": None},
            ...         "action_options": []
            ...     }
            ... ]


            In this multiclass example, the target class 0 is present in fewer than 30% of observations, however with 1 cv fold, the minimum
            number of instances required is 2 * 1 = 2. Therefore a warning, but not an error, is raised.

            >>> y = pd.Series([0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2])
            >>> class_imb_dc = ClassImbalanceDataCheck(threshold=0.30, min_samples=5, num_cv_folds=1)
            >>> assert class_imb_dc.validate(X, y) == [
            ...     {
            ...         "message": "The following labels fall below 30% of the target: [0]",
            ...         "data_check_name": "ClassImbalanceDataCheck",
            ...         "level": "warning",
            ...         "code": "CLASS_IMBALANCE_BELOW_THRESHOLD",
            ...         "details": {"target_values": [0], "rows": None, "columns": None},
            ...         "action_options": []
            ...     },
            ...     {
            ...         "message": "The following labels in the target have severe class imbalance because they fall under 30% of the target and have less than 5 samples: [0]",
            ...         "data_check_name": "ClassImbalanceDataCheck",
            ...         "level": "warning",
            ...         "code": "CLASS_IMBALANCE_SEVERE",
            ...         "details": {"target_values": [0], "rows": None, "columns": None},
            ...         "action_options": []
            ...     }
            ... ]
            ...
            >>> y = pd.Series([0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
            >>> class_imb_dc = ClassImbalanceDataCheck(threshold=0.30, num_cv_folds=1)
            >>> assert class_imb_dc.validate(X, y) == []
        """
        messages = []

        y = infer_feature_types(y)

        fold_counts = y.value_counts(normalize=False, sort=True)
        fold_counts = np.floor(fold_counts * self.test_size).astype(int)
        if len(fold_counts) == 0:
            return messages
        # search for targets that occur less than twice the number of cv folds first
        below_threshold_folds = fold_counts.where(fold_counts < self.cv_folds).dropna()
        if len(below_threshold_folds):
            below_threshold_values = below_threshold_folds.index.tolist()
            error_msg = "The number of instances of these targets is less than 2 * the number of cross folds = {} instances: {}"
            messages.append(
                DataCheckError(
                    message=error_msg.format(
                        self.cv_folds,
                        sorted(below_threshold_values),
                    ),
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_FOLDS,
                    details={"target_values": sorted(below_threshold_values)},
                ).to_dict(),
            )

        counts = fold_counts / (fold_counts + fold_counts.values[0])
        below_threshold = counts.where(counts < self.threshold).dropna()
        # if there are items that occur less than the threshold, add them to the list of results
        if len(below_threshold):
            below_threshold_values = below_threshold.index.tolist()
            warning_msg = "The following labels fall below {:.0f}% of the target: {}"
            messages.append(
                DataCheckWarning(
                    message=warning_msg.format(
                        self.threshold * 100,
                        below_threshold_values,
                    ),
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_THRESHOLD,
                    details={"target_values": below_threshold_values},
                ).to_dict(),
            )
        sample_counts = fold_counts.where(fold_counts < self.min_samples).dropna()
        if len(below_threshold) and len(sample_counts):
            sample_count_values = sample_counts.index.tolist()
            severe_imbalance = [v for v in sample_count_values if v in below_threshold]
            warning_msg = "The following labels in the target have severe class imbalance because they fall under {:.0f}% of the target and have less than {} samples: {}"
            messages.append(
                DataCheckWarning(
                    message=warning_msg.format(
                        self.threshold * 100,
                        self.min_samples,
                        severe_imbalance,
                    ),
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.CLASS_IMBALANCE_SEVERE,
                    details={"target_values": severe_imbalance},
                ).to_dict(),
            )
        return messages
