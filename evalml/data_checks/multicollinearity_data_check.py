"""Data check to check if any set features are likely to be multicollinear."""
from evalml.data_checks import DataCheck, DataCheckMessageCode, DataCheckWarning
from evalml.utils import infer_feature_types


class MulticollinearityDataCheck(DataCheck):
    """Check if any set features are likely to be multicollinear.

    Args:
        threshold (float): The threshold to be considered. Defaults to 0.9.
    """

    def __init__(self, threshold=0.9):
        if threshold < 0 or threshold > 1:
            raise ValueError("threshold must be a float between 0 and 1, inclusive.")
        self.threshold = threshold

    def validate(self, X, y=None):
        """Check if any set of features are likely to be multicollinear.

        Args:
            X (pd.DataFrame): The input features to check.
            y (pd.Series): The target. Ignored.

        Returns:
            dict: dict with a DataCheckWarning if there are any potentially multicollinear columns.

        Example:
            >>> import pandas as pd

            Columns in X that are highly correlated with each other will be identified using mutual information.

            >>> col = pd.Series([1, 0, 2, 3, 4] * 15)
            >>> X = pd.DataFrame({"col_1": col, "col_2": col * 3})
            >>> y = pd.Series([1, 0, 0, 1, 0] * 15)
            ...
            >>> multicollinearity_check = MulticollinearityDataCheck(threshold=1.0)
            >>> assert multicollinearity_check.validate(X, y) == [
            ...     {
            ...         "message": "Columns are likely to be correlated: [('col_1', 'col_2')]",
            ...         "data_check_name": "MulticollinearityDataCheck",
            ...         "level": "warning",
            ...         "code": "IS_MULTICOLLINEAR",
            ...         "details": {"columns": [("col_1", "col_2")], "rows": None},
            ...         "action_options": []
            ...     }
            ... ]
        """
        messages = []

        X = infer_feature_types(X)
        mutual_info_df = X.ww.mutual_information()
        if mutual_info_df.empty:
            return messages
        above_threshold = mutual_info_df.loc[
            mutual_info_df["mutual_info"] >= self.threshold
        ]
        correlated_cols = [
            (col_1, col_2)
            for col_1, col_2 in zip(
                above_threshold["column_1"],
                above_threshold["column_2"],
            )
        ]
        if correlated_cols:
            warning_msg = "Columns are likely to be correlated: {}"
            messages.append(
                DataCheckWarning(
                    message=warning_msg.format(correlated_cols),
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.IS_MULTICOLLINEAR,
                    details={"columns": correlated_cols},
                ).to_dict(),
            )
        return messages
