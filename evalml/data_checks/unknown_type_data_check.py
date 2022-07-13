"""Data check that checks if there are high number of Unknown type of columns."""
from evalml.data_checks import DataCheck, DataCheckMessageCode, DataCheckWarning
from evalml.utils import infer_feature_types


class UnknownTypeDataCheck(DataCheck):
    """Check if there are a high number of features that are labelled as unknown by Woodwork."""

    def __init__(
        self,
        unknown_percentage_threshold=0.50,
    ):
        if not 0 <= unknown_percentage_threshold <= 1:
            raise ValueError(
                "`unknown_percentage_threshold` must be a float between 0 and 1, inclusive.",
            )
        self.unknown_percentage_threshold = unknown_percentage_threshold

    def validate(self, X, y=None):
        """Check if there are any rows or columns that have a high percentage of unknown types.

        Args:
            X (pd.DataFrame, np.ndarray): Features.
            y (pd.Series, np.nparray): Ignored. Defaults to None.

        Returns:
            dict: A dictionary with warnings if any columns

        Examples:
            >>> import pandas as pd

            We use the default unknown_percentage_threshold.

            >>> df = pd.DataFrame({
            ...     "all_null": [None, pd.NA, None, None, None],
            ...     "literally_all_null": [None, None, None, None, None],
            ...     "few_null": [1, 2, None, 2, 3],
            ...     "no_null": [1, 2, 3, 4, 5]
            ... })
            ...
            >>> unknown_type_dc = UnknownTypeDataCheck()
        """
        messages = []

        X = infer_feature_types(X)
        row_unknowns = X.ww.select(["Unknown"])
        print(X.ww)
        if (
            len(row_unknowns.columns) / len(X.columns)
            >= self.unknown_percentage_threshold
        ):
            warning_msg = f"{len(row_unknowns.columns)} out of {len(X.columns)} rows are unknown type, meaning the number of rows that are unknown is more than {self.unknown_percentage_threshold*100}%."

            messages.append(
                DataCheckWarning(
                    message=warning_msg,
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.HIGH_NUMBER_OF_UNKNOWN_TYPE,
                    details={"columns": list(row_unknowns.columns)},
                    action_options=[],
                ).to_dict(),
            )
        return messages
