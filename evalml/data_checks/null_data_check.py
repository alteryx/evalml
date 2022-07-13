"""Data check that checks if there are any highly-null columns and rows in the input."""
from evalml.data_checks import (
    DataCheck,
    DataCheckActionCode,
    DataCheckActionOption,
    DataCheckMessageCode,
    DataCheckWarning,
)
from evalml.utils import infer_feature_types


class NullDataCheck(DataCheck):
    """Check if there are any highly-null numerical, boolean, categorical, natural language, and unknown columns and rows in the input.

    Args:
        pct_null_col_threshold(float): If the percentage of NaN values in an input feature exceeds this amount,
            that column will be considered highly-null. Defaults to 0.95.
        pct_moderately_null_col_threshold(float): If the percentage of NaN values in an input feature exceeds this amount
             but is less than the percentage specified in pct_null_col_threshold, that column will be considered
             moderately-null. Defaults to 0.20.
        pct_null_row_threshold(float): If the percentage of NaN values in an input row exceeds this amount,
            that row will be considered highly-null. Defaults to 0.95.
    """

    def __init__(
        self,
        pct_null_col_threshold=0.95,
        pct_moderately_null_col_threshold=0.20,
        pct_null_row_threshold=0.95,
    ):
        if not 0 <= pct_null_col_threshold <= 1:
            raise ValueError(
                "`pct_null_col_threshold` must be a float between 0 and 1, inclusive.",
            )
        if not 0 <= pct_moderately_null_col_threshold <= pct_null_col_threshold <= 1:
            raise ValueError(
                "`pct_moderately_null_col_threshold` must be a float between 0 and 1, inclusive, and must be less than or equal to `pct_null_col_threshold`.",
            )
        if not 0 <= pct_null_row_threshold <= 1:
            raise ValueError(
                "`pct_null_row_threshold` must be a float between 0 and 1, inclusive.",
            )

        self.pct_null_col_threshold = pct_null_col_threshold
        self.pct_moderately_null_col_threshold = pct_moderately_null_col_threshold
        self.pct_null_row_threshold = pct_null_row_threshold

    def validate(self, X, y=None):
        """Check if there are any highly-null columns or rows in the input.

        Args:
            X (pd.DataFrame, np.ndarray): Features.
            y (pd.Series, np.ndarray): Ignored. Defaults to None.

        Returns:
            dict: dict with a DataCheckWarning if there are any highly-null columns or rows.

        Examples:
            >>> import pandas as pd
            ...
            >>> class SeriesWrap():
            ...     def __init__(self, series):
            ...         self.series = series
            ...
            ...     def __eq__(self, series_2):
            ...         return all(self.series.eq(series_2.series))

            With pct_null_col_threshold set to 0.50, any column that has 50% or more of its observations set to null will be
            included in the warning, as well as the percentage of null values identified ("all_null": 1.0, "lots_of_null": 0.8).

            >>> df = pd.DataFrame({
            ...     "all_null": [None, pd.NA, None, None, None],
            ...     "lots_of_null": [None, None, None, None, 5],
            ...     "few_null": [1, 2, None, 2, 3],
            ...     "no_null": [1, 2, 3, 4, 5]
            ... })
            ...
            >>> highly_null_dc = NullDataCheck(pct_null_col_threshold=0.50)
            >>> assert highly_null_dc.validate(df) == [
            ...     {
            ...         "message": "Column(s) 'all_null', 'lots_of_null' are 50.0% or more null",
            ...         "data_check_name": "NullDataCheck",
            ...         "level": "warning",
            ...         "details": {
            ...             "columns": ["all_null", "lots_of_null"],
            ...             "rows": None,
            ...             "pct_null_rows": {"all_null": 1.0, "lots_of_null": 0.8}
            ...         },
            ...         "code": "HIGHLY_NULL_COLS",
            ...         "action_options": [
            ...             {
            ...                 "code": "DROP_COL",
            ...                 "data_check_name": "NullDataCheck",
            ...                 "parameters": {},
            ...                 "metadata": {"columns": ["all_null", "lots_of_null"], "rows": None}
            ...             }
            ...         ]
            ...     },
            ...     {
            ...         "message": "Column(s) 'few_null' have between 20.0% and 50.0% null values",
            ...         "data_check_name": "NullDataCheck",
            ...         "level": "warning",
            ...         "details": {"columns": ["few_null"], "rows": None},
            ...         "code": "COLS_WITH_NULL",
            ...         "action_options": [
            ...             {
            ...                 "code": "IMPUTE_COL",
            ...                 "data_check_name": "NullDataCheck",
            ...                 "metadata": {"columns": ["few_null"], "rows": None, "is_target": False},
            ...                 "parameters": {
            ...                     "impute_strategies": {
            ...                         "parameter_type": "column",
            ...                         "columns": {
            ...                             "few_null": {
            ...                                 "impute_strategy": {"categories": ["mean", "most_frequent"], "type": "category", "default_value": "mean"}
            ...                             }
            ...                         }
            ...                     }
            ...                 }
            ...             }
            ...         ]
            ...     }
            ... ]

            With pct_null_row_threshold set to 0.50, any row with 50% or more of its respective column values set to null will
            included in the warning, as well as the offending rows ("rows": [0, 1, 2, 3]).
            Since the default value for pct_null_col_threshold is 0.95, "all_null" is also included in the warnings since
            the percentage of null values in that row is over 95%.
            Since the default value for pct_moderately_null_col_threshold is 0.20, "few_null" is included as a "moderately null"
            column as it has a null column percentage of 20%.

            >>> highly_null_dc = NullDataCheck(pct_null_row_threshold=0.50)
            >>> validation_messages = highly_null_dc.validate(df)
            >>> validation_messages[0]["details"]["pct_null_cols"] = SeriesWrap(validation_messages[0]["details"]["pct_null_cols"])
            >>> highly_null_rows = SeriesWrap(pd.Series([0.5, 0.5, 0.75, 0.5]))
            >>> assert validation_messages == [
            ...     {
            ...         "message": "4 out of 5 rows are 50.0% or more null",
            ...         "data_check_name": "NullDataCheck",
            ...         "level": "warning",
            ...         "details": {
            ...             "columns": None,
            ...             "rows": [0, 1, 2, 3],
            ...             "pct_null_cols": highly_null_rows
            ...         },
            ...         "code": "HIGHLY_NULL_ROWS",
            ...         "action_options": [
            ...             {
            ...                 "code": "DROP_ROWS",
            ...                  "data_check_name": "NullDataCheck",
            ...                  "parameters": {},
            ...                  "metadata": {"columns": None, "rows": [0, 1, 2, 3]}
            ...              }
            ...         ]
            ...     },
            ...     {
            ...         "message": "Column(s) 'all_null' are 95.0% or more null",
            ...         "data_check_name": "NullDataCheck",
            ...         "level": "warning",
            ...         "details": {
            ...             "columns": ["all_null"],
            ...             "rows": None,
            ...             "pct_null_rows": {"all_null": 1.0}
            ...         },
            ...        "code": "HIGHLY_NULL_COLS",
            ...        "action_options": [
            ...             {
            ...                 "code": "DROP_COL",
            ...                 "data_check_name": "NullDataCheck",
            ...                 "metadata": {"columns": ["all_null"], "rows": None},
            ...                 "parameters": {}
            ...             }
            ...         ]
            ...     },
            ...     {
            ...         "message": "Column(s) 'lots_of_null', 'few_null' have between 20.0% and 95.0% null values",
            ...         "data_check_name": "NullDataCheck",
            ...         "level": "warning",
            ...         "details": {"columns": ["lots_of_null", "few_null"], "rows": None},
            ...         "code": "COLS_WITH_NULL",
            ...         "action_options": [
            ...             {
            ...                "code": "IMPUTE_COL",
            ...                "data_check_name": "NullDataCheck",
            ...                "metadata": {"columns": ["lots_of_null", "few_null"], "rows": None, "is_target": False},
            ...                "parameters": {
            ...                    "impute_strategies": {
            ...                        "parameter_type": "column",
            ...                        "columns": {
            ...                            "lots_of_null": {"impute_strategy": {"categories": ["mean", "most_frequent"], "type": "category", "default_value": "mean"}},
            ...                            "few_null": {"impute_strategy": {"categories": ["mean", "most_frequent"], "type": "category", "default_value": "mean"}}
            ...                        }
            ...                    }
            ...                }
            ...             }
            ...         ]
            ...     }
            ... ]

        """
        messages = []

        X = infer_feature_types(X)

        highly_null_rows = NullDataCheck.get_null_row_information(
            X,
            pct_null_row_threshold=self.pct_null_row_threshold,
        )
        if len(highly_null_rows) > 0:
            warning_msg = f"{len(highly_null_rows)} out of {len(X)} rows are {self.pct_null_row_threshold*100}% or more null"
            rows_to_drop = highly_null_rows.index.tolist()

            messages.append(
                DataCheckWarning(
                    message=warning_msg,
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.HIGHLY_NULL_ROWS,
                    details={
                        "rows": highly_null_rows.index.tolist(),
                        "pct_null_cols": highly_null_rows,
                    },
                    action_options=[
                        DataCheckActionOption(
                            DataCheckActionCode.DROP_ROWS,
                            data_check_name=self.name,
                            metadata={"rows": rows_to_drop},
                        ),
                    ],
                ).to_dict(),
            )

        highly_null_cols, _ = NullDataCheck.get_null_column_information(
            X,
            pct_null_col_threshold=self.pct_null_col_threshold,
        )

        X_to_check_for_any_null = X.ww.select(
            [
                "category",
                "boolean",
                "numeric",
                "IntegerNullable",
                "BooleanNullable",
            ],
        )

        cols_at_least_moderately_null, _ = NullDataCheck.get_null_column_information(
            X_to_check_for_any_null,
            pct_null_col_threshold=self.pct_moderately_null_col_threshold,
        )

        moderately_null_cols = [
            col for col in cols_at_least_moderately_null if col not in highly_null_cols
        ]

        warning_msg = "Column(s) {} are {}% or more null"
        if highly_null_cols:
            messages.append(
                DataCheckWarning(
                    message=warning_msg.format(
                        (", ").join(
                            ["'{}'".format(str(col)) for col in highly_null_cols],
                        ),
                        self.pct_null_col_threshold * 100,
                    ),
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.HIGHLY_NULL_COLS,
                    details={
                        "columns": list(highly_null_cols),
                        "pct_null_rows": highly_null_cols,
                    },
                    action_options=[
                        DataCheckActionOption(
                            DataCheckActionCode.DROP_COL,
                            data_check_name=self.name,
                            metadata={"columns": list(highly_null_cols)},
                        ),
                    ],
                ).to_dict(),
            )

        if moderately_null_cols:

            impute_strategies_dict = {}
            for col in moderately_null_cols:
                col_in_df = X.ww[col]
                categories = (
                    ["mean", "most_frequent"]
                    if col_in_df.ww.schema.is_numeric
                    else ["most_frequent"]
                )
                default_value = (
                    "mean" if col_in_df.ww.schema.is_numeric else "most_frequent"
                )
                impute_strategies_dict[col] = {
                    "impute_strategy": {
                        "categories": categories,
                        "type": "category",
                        "default_value": default_value,
                    },
                }

            messages.append(
                DataCheckWarning(
                    message="Column(s) {} have between {}% and {}% null values".format(
                        (", ").join(
                            ["'{}'".format(str(col)) for col in moderately_null_cols],
                        ),
                        self.pct_moderately_null_col_threshold * 100,
                        self.pct_null_col_threshold * 100,
                    ),
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.COLS_WITH_NULL,
                    details={
                        "columns": list(moderately_null_cols),
                    },
                    action_options=[
                        DataCheckActionOption(
                            DataCheckActionCode.IMPUTE_COL,
                            data_check_name=self.name,
                            parameters={
                                "impute_strategies": {
                                    "parameter_type": "column",
                                    "columns": impute_strategies_dict,
                                },
                            },
                            metadata={
                                "columns": list(moderately_null_cols),
                                "is_target": False,
                            },
                        ),
                    ],
                ).to_dict(),
            )
        return messages

    @staticmethod
    def get_null_column_information(X, pct_null_col_threshold=0.0):
        """Finds columns that are considered highly null (percentage null is greater than threshold) and returns dictionary mapping column name to percentage null and dictionary mapping column name to null indices.

        Args:
            X (pd.DataFrame): DataFrame to check for highly null columns.
            pct_null_col_threshold (float): Percentage threshold for a column to be considered null. Defaults to 0.0.

        Returns:
            tuple: Tuple containing: dictionary mapping column name to its null percentage and dictionary mapping column name to null indices in that column.
        """
        percent_null_cols = (X.isnull().mean()).to_dict()
        highly_null_cols = {
            key: value
            for key, value in percent_null_cols.items()
            if value >= pct_null_col_threshold and value != 0
        }
        highly_null_cols_indices = {
            col_: X[col_][X[col_].isnull()].index.tolist() for col_ in highly_null_cols
        }
        return highly_null_cols, highly_null_cols_indices

    @staticmethod
    def get_null_row_information(X, pct_null_row_threshold=0.0):
        """Finds rows that are considered highly null (percentage null is greater than threshold).

        Args:
            X (pd.DataFrame): DataFrame to check for highly null rows.
            pct_null_row_threshold (float): Percentage threshold for a row to be considered null. Defaults to 0.0.

        Returns:
            pd.Series: Series containing the percentage null for each row.
        """
        percent_null_rows = X.isnull().mean(axis=1)
        highly_null_rows = percent_null_rows[
            percent_null_rows >= pct_null_row_threshold
        ]
        return highly_null_rows
