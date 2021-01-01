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

    HIGH_VARIANCE = "high_variance"
    """Message code for when high variance is detected for cross-validation."""

    TARGET_LEAKAGE = "target_leakage"
    """Message code for when target leakage is detected."""

    HAS_OUTLIERS = "has_outliers"
    """Message code for when outliers are detected."""

    CLASS_IMBALANCE_BELOW_THRESHOLD = "class_imbalance_below_threshold"
    """Message code for when balance in classes is less than the threshold."""

    CLASS_IMBALANCE_BELOW_FOLDS = "class_imbalance_below_folds"
    """Message code for when the number of values for each target is below 2 * number of CV folds."""

    NO_VARIANCE = "no_variance"
    """Message code for when data has no variance (1 unique value)."""

    NO_VARIANCE_WITH_NULL = "no_variance_with_null"
    """Message code for when data has one unique value and NaN values."""

    IS_MULTICOLLINEAR = "is_multicollinear"
    """Message code for when data is potentially multicollinear."""
