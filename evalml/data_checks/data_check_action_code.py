"""Enum for data check action code."""
from enum import Enum

from evalml.utils import classproperty


class DataCheckActionCode(Enum):
    """Enum for data check action code."""

    DROP_COL = "drop_col"
    """Action code for dropping a column."""

    DROP_ROWS = "drop_rows"
    """Action code for dropping rows."""

    IMPUTE_COL = "impute_col"
    """Action code for imputing a column."""

    TRANSFORM_TARGET = "transform_target"
    """Action code for transforming the target data."""

    REGULARIZE_AND_IMPUTE_DATASET = "regularize_and_impute_dataset"
    """Action code for regularizing and imputing all features and target time series data."""

    SET_FIRST_COL_ID = "set_first_col_id"
    """Action code for setting the first column as an id column."""

    @classproperty
    def _all_values(cls):
        return {code.value.upper(): code for code in list(cls)}

    def __str__(self):
        """String representation of the DataCheckActionCode enum."""
        datacheck_action_code_dict = {
            DataCheckActionCode.DROP_COL.name: "drop_col",
            DataCheckActionCode.DROP_ROWS.name: "drop_rows",
            DataCheckActionCode.IMPUTE_COL.name: "impute_col",
            DataCheckActionCode.TRANSFORM_TARGET.name: "transform_target",
            DataCheckActionCode.REGULARIZE_AND_IMPUTE_DATASET.name: "regularize_and_impute_dataset",
            DataCheckActionCode.SET_FIRST_COL_ID.name: "set_first_col_id",
        }
        return datacheck_action_code_dict[self.name]
