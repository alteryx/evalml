"""Exceptions used in EvalML."""
from enum import Enum


class MethodPropertyNotFoundError(Exception):
    """Exception to raise when a class is does not have an expected method or property."""

    pass


class PipelineNotFoundError(Exception):
    """An exception raised when a particular pipeline is not found in automl search results."""

    pass


class ObjectiveNotFoundError(Exception):
    """Exception to raise when specified objective does not exist."""

    pass


class MissingComponentError(Exception):
    """An exception raised when a component is not found in all_components()."""

    pass


class ComponentNotYetFittedError(Exception):
    """An exception to be raised when predict/predict_proba/transform is called on a component without fitting first."""

    pass


class PipelineNotYetFittedError(Exception):
    """An exception to be raised when predict/predict_proba/transform is called on a pipeline without fitting first."""

    pass


class AutoMLSearchException(Exception):
    """Exception raised when all pipelines in an automl batch return a score of NaN for the primary objective."""

    pass


class EnsembleMissingPipelinesError(Exception):
    """An exception raised when an ensemble is missing `estimators` (list) as a parameter."""

    pass


class PipelineScoreError(Exception):
    """An exception raised when a pipeline errors while scoring any objective in a list of objectives.

    Args:
        exceptions (dict): A dictionary mapping an objective name (str) to a tuple of the form (exception, traceback).
            All of the objectives that errored will be stored here.
        scored_successfully (dict): A dictionary mapping an objective name (str) to a score value. All of the objectives
            that did not error will be stored here.
    """

    def __init__(self, exceptions, scored_successfully):
        self.exceptions = exceptions
        self.scored_successfully = scored_successfully

        # Format the traceback message
        exception_list = []
        for objective, (exception, tb) in exceptions.items():
            exception_list.append(
                f"{objective} encountered {str(exception.__class__.__name__)} with message ({str(exception)}):\n"
            )
            exception_list.extend(tb)
        message = "\n".join(exception_list)

        self.message = message
        super().__init__(message)


class DataCheckInitError(Exception):
    """Exception raised when a data check can't initialize with the parameters given."""


class NullsInColumnWarning(UserWarning):
    """Warning thrown when there are null values in the column of interest."""


class ObjectiveCreationError(Exception):
    """Exception when get_objective tries to instantiate an objective and required args are not provided."""


class NoPositiveLabelException(Exception):
    """Exception when a particular classification label for the 'positive' class cannot be found in the column index or unique values."""


class ParameterNotUsedWarning(UserWarning):
    """Warning thrown when a pipeline parameter isn't used in a defined pipeline's component graph during initialization."""

    def __init__(self, components):
        self.components = components

        msg = f"Parameters for components {components} will not be used to instantiate the pipeline since they don't appear in the pipeline"
        super().__init__(msg)


class PartialDependenceErrorCode(Enum):
    """Enum identifying the type of error encountered in partial dependence."""

    TOO_MANY_FEATURES = "too_many_features"
    """too_many_features"""
    FEATURES_ARGUMENT_INCORRECT_TYPES = "features_argument_incorrect_types"
    """features_argument_incorrect_types"""
    UNFITTED_PIPELINE = "unfitted_pipeline"
    """unfitted_pipeline"""
    PIPELINE_IS_BASELINE = "pipeline_is_baseline"
    """pipeline_is_baseline"""
    TWO_WAY_REQUESTED_FOR_DATES = "two_way_requested_for_dates"
    """two_way_requested_for_dates"""
    FEATURE_IS_ALL_NANS = "feature_is_all_nans"
    """feature_is_all_nans"""
    FEATURE_IS_MOSTLY_ONE_VALUE = "feature_is_mostly_one_value"
    """feature_is_mostly_one_value"""
    COMPUTED_PERCENTILES_TOO_CLOSE = "computed_percentiles_too_close"
    """computed_percentiles_too_close"""
    INVALID_FEATURE_TYPE = "invalid_feature_type"
    """invalid_feature_type"""
    ICE_PLOT_REQUESTED_FOR_TWO_WAY_PLOT = (
        "ice_plot_requested_for_two_way_partial_dependence_plot"
    )
    """ice_plot_requested_for_two_way_partial_dependence_plot"""
    INVALID_CLASS_LABEL = "invalid_class_label_requested_for_plot"
    """invalid_class_label_requested_for_plot"""
    ALL_OTHER_ERRORS = "all_other_errors"
    """all_other_errors"""


class PartialDependenceError(ValueError):
    """Exception raised for all errors that partial dependence can raise."""

    def __init__(self, message, code):
        self.code = code
        super().__init__(message)
