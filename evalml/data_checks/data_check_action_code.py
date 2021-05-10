from enum import Enum


class DataCheckActionCode(Enum):
    """Enum for data check action code."""

    DROP_COL = "drop_col"
    """Action code for dropping a column."""

    DROP_ROWS = "drop_rows"
    """Action code for dropping rows."""

    IMPUTE_COL = "impute_col"
    """Action code for imputing a column."""
