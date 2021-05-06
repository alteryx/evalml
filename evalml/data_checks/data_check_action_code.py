from enum import Enum


class DataCheckActionCode(Enum):
    """Enum for data check action code."""

    DROP_COL = "drop_col"
    """Action code for dropping a column."""

    DROP_ROW = "drop_row"
    """Action code for dropping a row."""

    IMPUTE_COL = "impute_col"
    """Action code for imputing a column."""
