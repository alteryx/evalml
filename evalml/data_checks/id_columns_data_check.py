
from evalml.data_checks import (
    DataCheck,
    DataCheckAction,
    DataCheckActionCode,
    DataCheckMessageCode,
    DataCheckWarning
)
from evalml.utils import _convert_woodwork_types_wrapper, infer_feature_types


class IDColumnsDataCheck(DataCheck):
    """Check if any of the features are likely to be ID columns."""

    def __init__(self, id_threshold=1.0):
        """Check if any of the features are likely to be ID columns.

        Arguments:
            id_threshold (float): The probability threshold to be considered an ID column. Defaults to 1.0.
        """
        if id_threshold < 0 or id_threshold > 1:
            raise ValueError("id_threshold must be a float between 0 and 1, inclusive.")
        self.id_threshold = id_threshold

    def validate(self, X, y=None):
        """Check if any of the features are likely to be ID columns. Currently performs these simple checks:

            - column name is "id"
            - column name ends in "_id"
            - column contains all unique values (and is categorical / integer type)

        Arguments:
            X (ww.DataTable, pd.DataFrame, np.ndarray): The input features to check

        Returns:
            dict: A dictionary of features with column name or index and their probability of being ID columns

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame({
            ...     'df_id': [0, 1, 2, 3, 4],
            ...     'x': [10, 42, 31, 51, 61],
            ...     'y': [42, 54, 12, 64, 12]
            ... })
            >>> id_col_check = IDColumnsDataCheck()
            >>> assert id_col_check.validate(df) == {"errors": [],\
                                                     "warnings": [{"message": "Column 'df_id' is 100.0% or more likely to be an ID column",\
                                                                   "data_check_name": "IDColumnsDataCheck",\
                                                                   "level": "warning",\
                                                                   "code": "HAS_ID_COLUMN",\
                                                                   "details": {"column": "df_id"}}],\
                                                     "actions": [{"code": "DROP_COL",\
                                                                 "metadata": {"column": "df_id"}}]}
        """
        results = {
            "warnings": [],
            "errors": [],
            "actions": []
        }

        X = infer_feature_types(X)

        col_names = [col for col in X.columns]
        cols_named_id = [col for col in col_names if (str(col).lower() == "id")]  # columns whose name is "id"
        id_cols = {col: 0.95 for col in cols_named_id}

        X = X.select(include=['Integer', 'Categorical'])
        X = _convert_woodwork_types_wrapper(X.to_dataframe())

        check_all_unique = (X.nunique() == len(X))
        cols_with_all_unique = check_all_unique[check_all_unique].index.tolist()  # columns whose values are all unique
        id_cols.update([(col, 1.0) if col in id_cols else (col, 0.95) for col in cols_with_all_unique])

        col_ends_with_id = [col for col in col_names if str(col).lower().endswith("_id")]  # columns whose name ends with "_id"
        id_cols.update([(col, 1.0) if str(col) in id_cols else (col, 0.95) for col in col_ends_with_id])

        id_cols_above_threshold = {key: value for key, value in id_cols.items() if value >= self.id_threshold}
        warning_msg = "Column '{}' is {}% or more likely to be an ID column"
        results["warnings"].extend([DataCheckWarning(message=warning_msg.format(col_name, self.id_threshold * 100),
                                                     data_check_name=self.name,
                                                     message_code=DataCheckMessageCode.HAS_ID_COLUMN,
                                                     details={"column": col_name}).to_dict()
                                    for col_name in id_cols_above_threshold])
        results["actions"].extend([DataCheckAction(DataCheckActionCode.DROP_COL,
                                                   metadata={"column": col_name}).to_dict()
                                   for col_name in id_cols_above_threshold])
        return results
