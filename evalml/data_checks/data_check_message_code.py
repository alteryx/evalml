from enum import Enum


class DataCheckMessageCode(Enum):
    """Enum for data check message code."""

    HIGHLY_NULL_COLS = "highly_null_cols"
    """Message code for highly null columns."""

    HIGHLY_NULL_ROWS = "highly_null_rows"
    """Message code for highly null rows."""

    HAS_ID_COLUMN = "has_id_column"
    """Message code for data that has ID columns."""

    TARGET_INCOMPATIBLE_OBJECTIVE = "target_incompatible_objective"
    """Message code for target data that has incompatible values for the specified objective"""

    TARGET_IS_NONE = "target_is_none"
    """Message code for when target is None."""

    TARGET_IS_EMPTY_OR_FULLY_NULL = "target_is_empty_or_fully_null"
    """Message code for target data that is empty or has all null values."""

    TARGET_HAS_NULL = "target_has_null"
    """Message code for target data that has null values."""

    TARGET_UNSUPPORTED_TYPE = "target_unsupported_type"
    """Message code for target data that is of an unsupported type."""

    TARGET_BINARY_NOT_TWO_UNIQUE_VALUES = "target_binary_not_two_unique_values"
    """Message code for target data for a binary classification problem that does not have two unique values."""

    TARGET_BINARY_INVALID_VALUES = "target_binary_invalid_values"
    """Message code for target data for a binary classification problem with numerical values not equal to {0, 1}."""

    TARGET_MULTICLASS_NOT_TWO_EXAMPLES_PER_CLASS = "target_multiclass_not_two_examples_per_class"
    """Message code for target data for a multi classification problem that does not have two examples per class."""

    TARGET_MULTICLASS_NOT_ENOUGH_CLASSES = "target_multiclass_not_enough_classes"
    """Message code for target data for a multi classification problem that does not have more than two unique classes."""

    TARGET_MULTICLASS_HIGH_UNIQUE_CLASS = "target_multiclass_high_unique_class_warning"
    """Message code for target data for a multi classification problem that has an abnormally large number of unique classes relative to the number of target values."""

    HIGH_VARIANCE = "high_variance"
    """Message code for when high variance is detected for cross-validation."""

    TARGET_LEAKAGE = "target_leakage"
    """Message code for when target leakage is detected."""

    HAS_OUTLIERS = "has_outliers"
    """Message code for when outliers are detected."""

    CLASS_IMBALANCE_BELOW_THRESHOLD = "class_imbalance_below_threshold"
    """Message code for when balance in classes is less than the threshold."""

    CLASS_IMBALANCE_SEVERE = "class_imbalance_severe"
    """Message code for when balance in classes is less than the threshold and minimum class is less than minimum number of accepted samples."""

    CLASS_IMBALANCE_BELOW_FOLDS = "class_imbalance_below_folds"
    """Message code for when the number of values for each target is below 2 * number of CV folds."""

    NO_VARIANCE = "no_variance"
    """Message code for when data has no variance (1 unique value)."""

    NO_VARIANCE_WITH_NULL = "no_variance_with_null"
    """Message code for when data has one unique value and NaN values."""

    IS_MULTICOLLINEAR = "is_multicollinear"
    """Message code for when data is potentially multicollinear."""

    NOT_UNIQUE_ENOUGH = "not_unique_enough"
    """Message code for when data does not possess enough unique values."""

    TOO_UNIQUE = "too_unique"
    """Message code for when data possesses too many unique values."""

    TOO_SPARSE = "too sparse"
    """Message code for when multiclass data has values that are too sparsely populated."""

    MISMATCHED_INDICES = "mismatched_indices"
    """Message code for when input target and features have mismatched indices."""

    MISMATCHED_INDICES_ORDER = "mismatched_indices_order"
    """Message code for when input target and features have mismatched indices order. The two inputs have the same index values, but shuffled."""

    MISMATCHED_LENGTHS = "mismatched_lengths"
    """Message code for when input target and features have different lengths."""

    DATETIME_HAS_NAN = "datetime_has_nan"
    """Message code for when input datetime columns contain NaN values."""

    NATURAL_LANGUAGE_HAS_NAN = "natural_language_has_nan"
    """Message code for when input natural language columns contain NaN values."""
