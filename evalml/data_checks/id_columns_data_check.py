"""Data check that checks if any of the features are likely to be ID columns."""
from evalml.data_checks import (
    DataCheck,
    DataCheckAction,
    DataCheckActionCode,
    DataCheckMessageCode,
    DataCheckWarning,
)
from evalml.utils import infer_feature_types


class IDColumnsDataCheck(DataCheck):
    """Check if any of the features are likely to be ID columns.

    Args:
        id_threshold (float): The probability threshold to be considered an ID column. Defaults to 1.0.
    """

    def __init__(self, id_threshold=1.0):
        if id_threshold < 0 or id_threshold > 1:
            raise ValueError("id_threshold must be a float between 0 and 1, inclusive.")
        self.id_threshold = id_threshold

    def validate(self, X, y=None):
        """Check if any of the features are likely to be ID columns. Currently performs a number of simple checks.

        Checks performed are:

            - column name is "id"
            - column name ends in "_id"
            - column contains all unique values (and is categorical / integer type)

        Args:
            X (pd.DataFrame, np.ndarray): The input features to check.
            y (pd.Series): The target. Defaults to None. Ignored.

        Returns:
            dict: A dictionary of features with column name or index and their probability of being ID columns

        Examples:
            >>> import pandas as pd
            ...
            >>> df = pd.DataFrame({
            ...     'customer_id': [123, 124, 125, 126, 127],
            ...     'Sales': [10, 42, 31, 51, 61]
            ... })
            ...
            >>> id_col_check = IDColumnsDataCheck()
            >>> assert id_col_check.validate(df) == {
            ...     "errors": [],
            ...     "warnings": [{"message": "Columns 'customer_id' are 100.0% or more likely to be an ID column",
            ...                   "data_check_name": "IDColumnsDataCheck",
            ...                   "level": "warning",
            ...                   "code": "HAS_ID_COLUMN",
            ...                   "details": {"columns": ["customer_id"], "rows": None}}],
            ...     "actions": [{"code": "DROP_COL",
            ...                  "data_check_name": "IDColumnsDataCheck",
            ...                  "metadata": {"columns": ["customer_id"], "rows": None}}]}
            ...
            ...
            >>> df = df.rename(columns={"customer_id": "ID"})
            >>> id_col_check = IDColumnsDataCheck()
            >>> assert id_col_check.validate(df) == {
            ...     "errors": [],
            ...     "warnings": [{"message": "Columns 'ID' are 100.0% or more likely to be an ID column",
            ...                   "data_check_name": "IDColumnsDataCheck",
            ...                   "level": "warning",
            ...                   "code": "HAS_ID_COLUMN",
            ...                   "details": {"columns": ["ID"], "rows": None}}],
            ...     "actions": [{"code": "DROP_COL",
            ...                  "data_check_name": "IDColumnsDataCheck",
            ...                  "metadata": {"columns": ["ID"], "rows": None}}]}
            ...
            ...
            >>> df = pd.DataFrame({
            ...    'Country_Rank': [1, 2, 3, 4, 5],
            ...    'Sales': ["very high", "high", "high", "medium", "very low"]
            ... })
            ...
            >>> id_col_check = IDColumnsDataCheck()
            >>> assert id_col_check.validate(df) == {'warnings': [], 'errors': [], 'actions': []}
            ...
            ...
            >>> id_col_check = IDColumnsDataCheck()
            >>> id_col_check = IDColumnsDataCheck(id_threshold=0.95)
            >>> assert id_col_check.validate(df) == {
            ...     'warnings': [{'message': "Columns 'Country_Rank' are 95.0% or more likely to be an ID column",
            ...                   'data_check_name': 'IDColumnsDataCheck',
            ...                   'level': 'warning',
            ...                   'details': {'columns': ['Country_Rank'], 'rows': None},
            ...                   'code': 'HAS_ID_COLUMN'}],
            ...     'errors': [],
            ...     'actions': [{'code': 'DROP_COL',
            ...                  'data_check_name': 'IDColumnsDataCheck',
            ...                  'metadata': {'columns': ['Country_Rank'], 'rows': None}}]}
        """
        results = {"warnings": [], "errors": [], "actions": []}

        X = infer_feature_types(X)

        col_names = [col for col in X.columns]
        cols_named_id = [
            col for col in col_names if (str(col).lower() == "id")
        ]  # columns whose name is "id"
        id_cols = {col: 0.95 for col in cols_named_id}

        X = X.ww.select(include=["Integer", "Categorical"])

        check_all_unique = X.nunique() == len(X)
        cols_with_all_unique = check_all_unique[
            check_all_unique
        ].index.tolist()  # columns whose values are all unique
        id_cols.update(
            [
                (col, 1.0) if col in id_cols else (col, 0.95)
                for col in cols_with_all_unique
            ]
        )

        col_ends_with_id = [
            col for col in col_names if str(col).lower().endswith("_id")
        ]  # columns whose name ends with "_id"
        id_cols.update(
            [
                (col, 1.0) if str(col) in id_cols else (col, 0.95)
                for col in col_ends_with_id
            ]
        )

        id_cols_above_threshold = {
            key: value for key, value in id_cols.items() if value >= self.id_threshold
        }
        if id_cols_above_threshold:
            warning_msg = "Columns {} are {}% or more likely to be an ID column"
            results["warnings"].append(
                DataCheckWarning(
                    message=warning_msg.format(
                        (", ").join(
                            ["'{}'".format(str(col)) for col in id_cols_above_threshold]
                        ),
                        self.id_threshold * 100,
                    ),
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.HAS_ID_COLUMN,
                    details={"columns": list(id_cols_above_threshold)},
                ).to_dict()
            )
            results["actions"].append(
                DataCheckAction(
                    DataCheckActionCode.DROP_COL,
                    data_check_name=self.name,
                    metadata={"columns": list(id_cols_above_threshold)},
                ).to_dict()
            )
        return results
