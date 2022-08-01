"""Enum for data check message code."""
from enum import Enum


class DataCheckMessageCode(Enum):
    """Enum for data check message code."""

    COLS_WITH_NULL = "cols_with_null"
    """Message code for columns with null values."""

    HIGHLY_NULL_COLS = "highly_null_cols"
    """Message code for highly null columns."""

    HIGHLY_NULL_ROWS = "highly_null_rows"
    """Message code for highly null rows."""

    HAS_ID_COLUMN = "has_id_column"
    """Message code for data that has ID columns."""

    HAS_ID_FIRST_COLUMN = "has_id_first_column"
    """Message code for data that has an ID column as the first column."""

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

    TARGET_UNSUPPORTED_TYPE_REGRESSION = "target_unsupported_type_regression"
    """Message code for target data that is incompatible with regression"""

    TARGET_UNSUPPORTED_PROBLEM_TYPE = "target_unsupported_problem_type"
    """Message code for target data that is being checked against an unsupported problem type."""

    TARGET_BINARY_NOT_TWO_UNIQUE_VALUES = "target_binary_not_two_unique_values"
    """Message code for target data for a binary classification problem that does not have two unique values."""

    TARGET_MULTICLASS_NOT_TWO_EXAMPLES_PER_CLASS = (
        "target_multiclass_not_two_examples_per_class"
    )
    """Message code for target data for a multi classification problem that does not have two examples per class."""

    TARGET_MULTICLASS_NOT_ENOUGH_CLASSES = "target_multiclass_not_enough_classes"
    """Message code for target data for a multi classification problem that does not have more than two unique classes."""

    TARGET_MULTICLASS_HIGH_UNIQUE_CLASS = "target_multiclass_high_unique_class_warning"
    """Message code for target data for a multi classification problem that has an abnormally large number of unique classes relative to the number of target values."""

    TARGET_LOGNORMAL_DISTRIBUTION = "target_lognormal_distribution"
    """Message code for target data with a lognormal distribution."""

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

    NO_VARIANCE_ZERO_UNIQUE = "no_variance_zero_unique"
    """Message code for when data has no variance (0 unique value)"""

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

    DATETIME_INFORMATION_NOT_FOUND = "datetime_information_not_found"
    """Message code for when datetime information can not be found or is in an unaccepted format."""

    DATETIME_NO_FREQUENCY_INFERRED = "datetime_no_frequency_inferred"
    """Message code for when no frequency can be inferred in the datetime values through Woodwork's infer_frequency."""

    DATETIME_HAS_UNEVEN_INTERVALS = "datetime_has_uneven_intervals"
    """Message code for when the datetime values have uneven intervals."""

    DATETIME_HAS_REDUNDANT_ROW = "datetime_has_redundant_row"
    """Message code for when datetime information has more than one row per datetime."""

    DATETIME_IS_MISSING_VALUES = "datetime_is_missing_values"
    """Message code for when datetime feature has values missing between the start and end dates."""

    DATETIME_HAS_MISALIGNED_VALUES = "datetime_has_misaligned_values"
    """Message code for when datetime information has values that are not aligned with the inferred frequency."""

    DATETIME_IS_NOT_MONOTONIC = "datetime_is_not_monotonic"
    """Message code for when the datetime values are not monotonically increasing."""

    TIMESERIES_PARAMETERS_NOT_COMPATIBLE_WITH_SPLIT = (
        "timeseries_parameters_not_compatible_with_split"
    )
    """Message code when the time series parameters are too large for the smallest data split."""

    TIMESERIES_TARGET_NOT_COMPATIBLE_WITH_SPLIT = (
        "timeseries_target_not_compatible_with_split"
    )
    """Message code when any training and validation split of the time series target doesn't contain all classes."""
