from enum import Enum


class DataCheckMessageCode(Enum):
    """Enum for data check message code."""

    HIGHLY_NULL = "highly_null"
    """Message code for highly null columns."""

    HAS_ID_COLUMN = "has_id_column"
    """Message code for data that has ID columns."""

    TARGET_HAS_NULL = "target_has_null"
    """Message code for target data that has null values."""

    TARGET_UNSUPPORTED_TYPE = "target_unsupported_type"
    """Message code for target data that is of an unsupported type."""

    TARGET_BINARY_NOT_TWO_UNIQUE_VALUES = "target_binary_not_two_unique_values"
    """Message code for target data for a binary classification problem that does not have two unique values."""

    TARGET_BINARY_INVALID_VALUES = "target_binary_invalid_values"
    """Message code for target data for a binary classification problem with numerical values not equal to {0, 1}."""
