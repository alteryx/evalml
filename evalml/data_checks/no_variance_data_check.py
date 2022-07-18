"""Data check that checks if the target or any of the features have no variance."""
from evalml.data_checks import (
    DataCheck,
    DataCheckActionCode,
    DataCheckActionOption,
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

    def validate(self, X, y=None):
        """Check if the target or any of the features have no variance (1 unique value).

        Args:
            X (pd.DataFrame, np.ndarray): The input features.
            y (pd.Series, np.ndarray): Optional, the target data.

        Returns:
            dict: A dict of warnings/errors corresponding to features or target with no variance.

        Examples:
            >>> import pandas as pd

            Columns or target data that have only one unique value will raise an error.

            >>> X = pd.DataFrame([2, 2, 2, 2, 2, 2, 2, 2], columns=["First_Column"])
            >>> y = pd.Series([1, 1, 1, 1, 1, 1, 1, 1])
            ...
            >>> novar_dc = NoVarianceDataCheck()
            >>> assert novar_dc.validate(X, y) == [
            ...     {
            ...         "message": "'First_Column' has 1 unique value.",
            ...         "data_check_name": "NoVarianceDataCheck",
            ...         "level": "warning",
            ...         "details": {"columns": ["First_Column"], "rows": None},
            ...         "code": "NO_VARIANCE",
            ...         "action_options": [
            ...             {
            ...                 "code": "DROP_COL",
            ...                 "data_check_name": "NoVarianceDataCheck",
            ...                 "parameters": {},
            ...                 "metadata": {"columns": ["First_Column"], "rows": None}
            ...             },
            ...         ]
            ...     },
            ...     {
            ...         "message": "Y has 1 unique value.",
            ...         "data_check_name": "NoVarianceDataCheck",
            ...         "level": "warning",
            ...         "details": {"columns": ["Y"], "rows": None},
            ...         "code": "NO_VARIANCE",
            ...         "action_options": []
            ...     }
            ... ]

            By default, NaNs will not be counted as distinct values. In the first example, there are still two distinct values
            besides None. In the second, there are no distinct values as the target is entirely null.

            >>> X["First_Column"] = [2, 2, 2, 3, 3, 3, None, None]
            >>> y = pd.Series([1, 1, 1, 2, 2, 2, None, None])
            >>> assert novar_dc.validate(X, y) == []
            ...
            ...
            >>> y = pd.Series([None] * 7)
            >>> assert novar_dc.validate(X, y) == [
            ...     {
            ...         "message": "Y has 0 unique values.",
            ...         "data_check_name": "NoVarianceDataCheck",
            ...         "level": "warning",
            ...         "details": {"columns": ["Y"], "rows": None},
            ...         "code": "NO_VARIANCE_ZERO_UNIQUE",
            ...         "action_options":[]
            ...     }
            ... ]

            As None is not considered a distinct value by default, there is only one unique value in X and y.

            >>> X["First_Column"] = [2, 2, 2, 2, None, None, None, None]
            >>> y = pd.Series([1, 1, 1, 1, None, None, None, None])
            >>> assert novar_dc.validate(X, y) == [
            ...     {
            ...         "message": "'First_Column' has 1 unique value.",
            ...         "data_check_name": "NoVarianceDataCheck",
            ...         "level": "warning",
            ...         "details": {"columns": ["First_Column"], "rows": None},
            ...         "code": "NO_VARIANCE",
            ...         "action_options": [
            ...             {
            ...                 "code": "DROP_COL",
            ...                  "data_check_name": "NoVarianceDataCheck",
            ...                  "parameters": {},
            ...                  "metadata": {"columns": ["First_Column"], "rows": None}
            ...             },
            ...         ]
            ...     },
            ...     {
            ...         "message": "Y has 1 unique value.",
            ...         "data_check_name": "NoVarianceDataCheck",
            ...         "level": "warning",
            ...         "details": {"columns": ["Y"], "rows": None},
            ...         "code": "NO_VARIANCE",
            ...         "action_options": []
            ...     }
            ... ]

            If count_nan_as_value is set to True, then NaNs are counted as unique values. In the event that there is an
            adequate number of unique values only because count_nan_as_value is set to True, a warning will be raised so
            the user can encode these values.

            >>> novar_dc = NoVarianceDataCheck(count_nan_as_value=True)
            >>> assert novar_dc.validate(X, y) == [
            ...     {
            ...         "message": "'First_Column' has two unique values including nulls. Consider encoding the nulls for this column to be useful for machine learning.",
            ...         "data_check_name": "NoVarianceDataCheck",
            ...         "level": "warning",
            ...         "details": {"columns": ["First_Column"], "rows": None},
            ...         "code": "NO_VARIANCE_WITH_NULL",
            ...         "action_options": [
            ...             {
            ...                 "code": "DROP_COL",
            ...                  "data_check_name": "NoVarianceDataCheck",
            ...                  "parameters": {},
            ...                  "metadata": {"columns": ["First_Column"], "rows": None}
            ...             },
            ...         ]
            ...     },
            ...     {
            ...         "message": "Y has two unique values including nulls. Consider encoding the nulls for this column to be useful for machine learning.",
            ...         "data_check_name": "NoVarianceDataCheck",
            ...         "level": "warning",
            ...         "details": {"columns": ["Y"], "rows": None},
            ...         "code": "NO_VARIANCE_WITH_NULL",
            ...         "action_options": []
            ...     }
            ... ]

        """
        messages = []

        X = infer_feature_types(X)
        if y is not None:
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
            messages.append(
                DataCheckWarning(
                    message=zero_unique_message.format(
                        (", ").join(["'{}'".format(str(col)) for col in zero_unique]),
                    ),
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.NO_VARIANCE_ZERO_UNIQUE,
                    details={"columns": zero_unique},
                    action_options=[
                        DataCheckActionOption(
                            DataCheckActionCode.DROP_COL,
                            data_check_name=self.name,
                            metadata={"columns": zero_unique},
                        ),
                    ],
                ).to_dict(),
            )
        if one_unique:
            messages.append(
                DataCheckWarning(
                    message=one_unique_message.format(
                        (", ").join(["'{}'".format(str(col)) for col in one_unique]),
                    ),
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.NO_VARIANCE,
                    details={"columns": one_unique},
                    action_options=[
                        DataCheckActionOption(
                            DataCheckActionCode.DROP_COL,
                            data_check_name=self.name,
                            metadata={"columns": one_unique},
                        ),
                    ],
                ).to_dict(),
            )
        if one_unique_with_null:
            messages.append(
                DataCheckWarning(
                    message=two_unique_with_null_message.format(
                        (", ").join(
                            ["'{}'".format(str(col)) for col in one_unique_with_null],
                        ),
                    ),
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.NO_VARIANCE_WITH_NULL,
                    details={"columns": one_unique_with_null},
                    action_options=[
                        DataCheckActionOption(
                            DataCheckActionCode.DROP_COL,
                            data_check_name=self.name,
                            metadata={"columns": one_unique_with_null},
                        ),
                    ],
                ).to_dict(),
            )

        if y is None:
            return messages

        # Check target for variance
        y_name = getattr(y, "name")
        if not y_name:
            y_name = "Y"

        y_unique_count = y.nunique(dropna=self._dropnan)
        y_any_null = y.isnull().any()

        if y_unique_count == 0:
            messages.append(
                DataCheckWarning(
                    message=zero_unique_message.format(y_name),
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.NO_VARIANCE_ZERO_UNIQUE,
                    details={"columns": [y_name]},
                ).to_dict(),
            )

        elif y_unique_count == 1:
            messages.append(
                DataCheckWarning(
                    message=one_unique_message.format(y_name),
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.NO_VARIANCE,
                    details={"columns": [y_name]},
                ).to_dict(),
            )

        elif y_unique_count == 2 and not self._dropnan and y_any_null:
            messages.append(
                DataCheckWarning(
                    message=two_unique_with_null_message.format(y_name),
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.NO_VARIANCE_WITH_NULL,
                    details={"columns": [y_name]},
                ).to_dict(),
            )

        return messages
