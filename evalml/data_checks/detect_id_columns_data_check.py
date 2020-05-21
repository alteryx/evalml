import pandas as pd

from .data_check import DataCheck
from .data_check_message import DataCheckWarning


class DetectIDColumnsDataCheck(DataCheck):

    def __init__(self, id_threshold=1.0):
        """Check if any of the features are ID columns. Currently performs these simple checks:

            - column name is "id"
            - column name ends in "_id"
            - column contains all unique values (and is not float / boolean)

        Arguments:
            id_threshold (float): the probability threshold to be considered an ID column. Defaults to 1.0.
        """
        if id_threshold < 0 or id_threshold > 1:
            raise ValueError("id_threshold must be a float between 0 and 1, inclusive.")
        self.id_threshold = id_threshold

    def validate(self, X, y=None):
        """Check if any of the features are ID columns. Currently performs these simple checks:

            - column name is "id"
            - column name ends in "_id"
            - column contains all unique values (and is not float / boolean)

        Arguments:
            X (pd.DataFrame): The input features to check
            threshold (float): the probability threshold to be considered an ID column. Defaults to 1.0

        Returns:
            A dictionary of features with column name or index and their probability of being ID columns

        Example:
            >>> df = pd.DataFrame({
            ...     'df_id': [0, 1, 2, 3, 4],
            ...     'x': [10, 42, 31, 51, 61],
            ...     'y': [42, 54, 12, 64, 12]
            ... })
            >>> id_col_check = DetectIDColumnsDataCheck()
            >>> assert id_col_check.validate(df) == [DataCheckWarning("Column 'df_id' is 100.0% or more likely to be an ID column", "DetectIDColumnsDataCheck")]
        """

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        col_names = [str(col) for col in X.columns.tolist()]
        cols_named_id = [col for col in col_names if (col.lower() == "id")]  # columns whose name is "id"
        id_cols = {col: 0.95 for col in cols_named_id}

        non_id_types = ['float16', 'float32', 'float64', 'bool']
        X = X.select_dtypes(exclude=non_id_types)
        check_all_unique = (X.nunique() == len(X))
        cols_with_all_unique = check_all_unique[check_all_unique].index.tolist()  # columns whose values are all unique
        id_cols.update([(str(col), 1.0) if col in id_cols else (str(col), 0.95) for col in cols_with_all_unique])

        col_ends_with_id = [col for col in col_names if str(col).lower().endswith("_id")]  # columns whose name ends with "_id"
        id_cols.update([(col, 1.0) if col in id_cols else (col, 0.95) for col in col_ends_with_id])

        id_cols_above_threshold = {key: value for key, value in id_cols.items() if value >= self.id_threshold}
        warning_msg = "Column '{}' is {}% or more likely to be an ID column"
        return [DataCheckWarning(warning_msg.format(col_name, self.id_threshold * 100), self.name) for col_name in id_cols_above_threshold]
