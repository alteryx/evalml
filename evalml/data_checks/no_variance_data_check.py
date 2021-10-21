"""Data check that checks if the target or any of the features have no variance."""
from evalml.data_checks import (
    DataCheck,
    DataCheckAction,
    DataCheckActionCode,
    DataCheckError,
    DataCheckMessageCode,
    DataCheckWarning,
)
from evalml.utils import infer_feature_types


class NoVarianceDataCheck(DataCheck):
    """Check if the target or any of the features have no variance.

    Args:
        count_nan_as_value (bool): If True, missing values will be counted as their own unique value.
            Additionally, if true, will return a DataCheckWarning instead of an error
            if the feature has mostly missing data and only one unique value.
            Defaults to False.
    """

    def __init__(self, count_nan_as_value=False):
        self._dropnan = not count_nan_as_value

    def validate(self, X, y):
        """Check if the target or any of the features have no variance (1 unique value).

        Args:
            X (pd.DataFrame, np.ndarray): The input features.
            y (pd.Series, np.ndarray): The target data.

        Returns:
            dict: A dict of warnings/errors corresponding to features or target with no variance.
        """
        results = {"warnings": [], "errors": [], "actions": []}
        X = infer_feature_types(X)
        y = infer_feature_types(y)

        unique_counts = X.nunique(dropna=self._dropnan).to_dict()
        any_nulls = (X.isnull().any()).to_dict()
        one_unique = []
        one_unique_with_null = []
        zero_unique = []
        for col_name in unique_counts:
            count_unique = unique_counts[col_name]
            has_any_nulls = any_nulls[col_name]
            if count_unique == 0:
                zero_unique.append(col_name)
            elif count_unique == 1:
                one_unique.append(col_name)
            elif count_unique == 2 and not self._dropnan and has_any_nulls:
                one_unique_with_null.append(col_name)

        zero_unique_message = "{} has 0 unique values."
        one_unique_message = "{} has 1 unique value."
        two_unique_with_null_message = "{} has two unique values including nulls. Consider encoding the nulls for this column to be useful for machine learning."
        if zero_unique:
            DataCheck._add_message(
                DataCheckError(
                    message=zero_unique_message.format(
                        (", ").join(["'{}'".format(str(col)) for col in zero_unique]),
                    ),
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.NO_VARIANCE,
                    details={"columns": zero_unique},
                ),
                results,
            )
        if one_unique:
            DataCheck._add_message(
                DataCheckError(
                    message=one_unique_message.format(
                        (", ").join(["'{}'".format(str(col)) for col in one_unique]),
                    ),
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.NO_VARIANCE,
                    details={"columns": one_unique},
                ),
                results,
            )
        if one_unique_with_null:
            DataCheck._add_message(
                DataCheckWarning(
                    message=two_unique_with_null_message.format(
                        (", ").join(
                            ["'{}'".format(str(col)) for col in one_unique_with_null]
                        ),
                    ),
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.NO_VARIANCE_WITH_NULL,
                    details={"columns": one_unique_with_null},
                ),
                results,
            )
        all_cols = zero_unique + one_unique + one_unique_with_null
        if all_cols:
            results["actions"].append(
                DataCheckAction(
                    DataCheckActionCode.DROP_COL,
                    metadata={"columns": all_cols},
                ).to_dict()
            )

        # Check target for variance
        y_name = getattr(y, "name")
        if not y_name:
            y_name = "Y"

        y_unique_count = y.nunique(dropna=self._dropnan)
        y_any_null = y.isnull().any()

        if y_unique_count == 0:
            DataCheck._add_message(
                DataCheckError(
                    message=zero_unique_message.format(y_name),
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.NO_VARIANCE,
                    details={"columns": [y_name]},
                ),
                results,
            )

        elif y_unique_count == 1:
            DataCheck._add_message(
                DataCheckError(
                    message=one_unique_message.format(y_name),
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.NO_VARIANCE,
                    details={"columns": [y_name]},
                ),
                results,
            )

        elif y_unique_count == 2 and not self._dropnan and y_any_null:
            DataCheck._add_message(
                DataCheckWarning(
                    message=two_unique_with_null_message.format(y_name),
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.NO_VARIANCE_WITH_NULL,
                    details={"columns": [y_name]},
                ),
                results,
            )

        return results
