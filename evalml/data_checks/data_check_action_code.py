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

    @classproperty
    def _all_values(cls):
        return {code.value.upper(): code for code in list(cls)}
