"""Data check that checks if any of the features are likely to be ID columns."""
from evalml.data_checks import (
    DataCheck,
    DataCheckActionCode,
    DataCheckActionOption,
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

            Columns that end in "_id" and are completely unique are likely to be ID columns.

            >>> df = pd.DataFrame({
            ...     "profits": [25, 15, 15, 31, 19],
            ...     "customer_id": [123, 124, 125, 126, 127],
            ...     "Sales": [10, 42, 31, 51, 61]
            ... })
            ...
            >>> id_col_check = IDColumnsDataCheck()
            >>> assert id_col_check.validate(df) == [
            ...     {
            ...         "message": "Columns 'customer_id' are 100.0% or more likely to be an ID column",
            ...         "data_check_name": "IDColumnsDataCheck",
            ...         "level": "warning",
            ...         "code": "HAS_ID_COLUMN",
            ...         "details": {"columns": ["customer_id"], "rows": None},
            ...         "action_options": [
            ...             {
            ...                 "code": "DROP_COL",
            ...                 "data_check_name": "IDColumnsDataCheck",
            ...                 "parameters": {},
            ...                 "metadata": {"columns": ["customer_id"], "rows": None}
            ...             }
            ...         ]
            ...    }
            ... ]

            Columns named "ID" with all unique values will also be identified as ID columns.

            >>> df = df.rename(columns={"customer_id": "ID"})
            >>> id_col_check = IDColumnsDataCheck()
            >>> assert id_col_check.validate(df) == [
            ...     {
            ...         "message": "Columns 'ID' are 100.0% or more likely to be an ID column",
            ...         "data_check_name": "IDColumnsDataCheck",
            ...         "level": "warning",
            ...         "code": "HAS_ID_COLUMN",
            ...         "details": {"columns": ["ID"], "rows": None},
            ...         "action_options": [
            ...            {
            ...                 "code": "DROP_COL",
            ...                 "data_check_name": "IDColumnsDataCheck",
            ...                 "parameters": {},
            ...                 "metadata": {"columns": ["ID"], "rows": None}
            ...             }
            ...         ]
            ...     }
            ... ]

            Despite being all unique, "Country_Rank" will not be identified as an ID column as id_threshold is set to 1.0
            by default and its name doesn't indicate that it's an ID.

            >>> df = pd.DataFrame({
            ...    "humidity": ["high", "very high", "low", "low", "high"],
            ...    "Country_Rank": [1, 2, 3, 4, 5],
            ...    "Sales": ["very high", "high", "high", "medium", "very low"]
            ... })
            ...
            >>> id_col_check = IDColumnsDataCheck()
            >>> assert id_col_check.validate(df) == []

            However lowering the threshold will cause this column to be identified as an ID.

            >>> id_col_check = IDColumnsDataCheck()
            >>> id_col_check = IDColumnsDataCheck(id_threshold=0.95)
            >>> assert id_col_check.validate(df) == [
            ...     {
            ...         "message": "Columns 'Country_Rank' are 95.0% or more likely to be an ID column",
            ...         "data_check_name": "IDColumnsDataCheck",
            ...         "level": "warning",
            ...         "details": {"columns": ["Country_Rank"], "rows": None},
            ...         "code": "HAS_ID_COLUMN",
            ...         "action_options": [
            ...             {
            ...                 "code": "DROP_COL",
            ...                 "data_check_name": "IDColumnsDataCheck",
            ...                 "parameters": {},
            ...                 "metadata": {"columns": ["Country_Rank"], "rows": None}
            ...             }
            ...         ]
            ...     }
            ... ]

            If the first column of the dataframe has all unique values and is named either 'ID' or a name that ends with '_id', it is probably the primary key.
            The other ID columns should be dropped.

            >>> df = pd.DataFrame({
            ...     "sales_id": [0, 1, 2, 3, 4],
            ...     "customer_id": [123, 124, 125, 126, 127],
            ...     "Sales": [10, 42, 31, 51, 61]
            ... })
            ...
            >>> id_col_check = IDColumnsDataCheck()
            >>> assert id_col_check.validate(df) == [
            ...     {
            ...         "message": "The first column 'sales_id' is likely to be the primary key",
            ...         "data_check_name": "IDColumnsDataCheck",
            ...         "level": "warning",
            ...         "code": "HAS_ID_FIRST_COLUMN",
            ...         "details": {"columns": ["sales_id"], "rows": None},
            ...         "action_options": [
            ...             {
            ...                 "code": "SET_FIRST_COL_ID",
            ...                 "data_check_name": "IDColumnsDataCheck",
            ...                 "parameters": {},
            ...                 "metadata": {"columns": ["sales_id"], "rows": None}
            ...             }
            ...         ]
            ...    },
            ...    {
            ...        "message": "Columns 'customer_id' are 100.0% or more likely to be an ID column",
            ...         "data_check_name": "IDColumnsDataCheck",
            ...         "level": "warning",
            ...         "code": "HAS_ID_COLUMN",
            ...         "details": {"columns": ["customer_id"], "rows": None},
            ...         "action_options": [
            ...             {
            ...                 "code": "DROP_COL",
            ...                 "data_check_name": "IDColumnsDataCheck",
            ...                 "parameters": {},
            ...                 "metadata": {"columns": ["customer_id"], "rows": None}
            ...             }
            ...         ]
            ...    }
            ... ]
        """
        messages = []
        X = infer_feature_types(X)

        col_names = [col for col in X.columns]
        cols_named_id = [
            col for col in col_names if (str(col).lower() == "id")
        ]  # columns whose name is "id"
        id_cols = {col: 0.95 for col in cols_named_id}

        for dtypes in [["Double"], ["Integer", "IntegerNullable", "Categorical"]]:
            X_temp = X.ww.select(include=dtypes)
            check_all_unique = X_temp.nunique() == len(X_temp)
            cols_with_all_unique = check_all_unique[
                check_all_unique
            ].index.tolist()  # columns whose values are all unique

            # Temporary solution for downstream instances of integers being mapped to doubles.
            # Will be removed when resolved.
            if dtypes == ["Double"]:
                cols_with_all_unique = [
                    col
                    for col in cols_with_all_unique
                    if all(
                        X_temp[col].mod(1).eq(0)
                    )  # Parse out columns that contain all `integer` values
                ]

            id_cols.update(
                [
                    (col, 1.0) if col in id_cols else (col, 0.95)
                    for col in cols_with_all_unique
                ],
            )

        col_ends_with_id = [
            col for col in col_names if str(col).lower().endswith("_id")
        ]  # columns whose name ends with "_id"
        id_cols.update(
            [
                (col, 1.0) if str(col) in id_cols else (col, 0.95)
                for col in col_ends_with_id
            ],
        )

        id_cols_above_threshold = {
            key: value for key, value in id_cols.items() if value >= self.id_threshold
        }

        if id_cols_above_threshold:
            if (
                col_names[0] in id_cols_above_threshold
                and id_cols_above_threshold[col_names[0]] == 1.0
            ):
                del id_cols_above_threshold[col_names[0]]
                warning_msg = "The first column '{}' is likely to be the primary key"
                warning_msg = warning_msg.format(
                    col_names[0],
                )
                messages.append(
                    DataCheckWarning(
                        message=warning_msg,
                        data_check_name=self.name,
                        message_code=DataCheckMessageCode.HAS_ID_FIRST_COLUMN,
                        details={"columns": [col_names[0]]},
                        action_options=[
                            DataCheckActionOption(
                                DataCheckActionCode.SET_FIRST_COL_ID,
                                data_check_name=self.name,
                                metadata={"columns": [col_names[0]]},
                            ),
                        ],
                    ).to_dict(),
                )
            if id_cols_above_threshold:
                warning_msg = "Columns {} are {}% or more likely to be an ID column"
                warning_msg = warning_msg.format(
                    (", ").join(
                        ["'{}'".format(str(col)) for col in id_cols_above_threshold],
                    ),
                    self.id_threshold * 100,
                )
                messages.append(
                    DataCheckWarning(
                        message=warning_msg,
                        data_check_name=self.name,
                        message_code=DataCheckMessageCode.HAS_ID_COLUMN,
                        details={"columns": list(id_cols_above_threshold)},
                        action_options=[
                            DataCheckActionOption(
                                DataCheckActionCode.DROP_COL,
                                data_check_name=self.name,
                                metadata={"columns": list(id_cols_above_threshold)},
                            ),
                        ],
                    ).to_dict(),
                )

        return messages
