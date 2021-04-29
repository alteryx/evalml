class MethodPropertyNotFoundError(Exception):
    """Exception to raise when a class is does not have an expected method or property."""
    pass


class PipelineNotFoundError(Exception):
    """An exception raised when a particular pipeline is not found in automl search results"""
    pass


class ObjectiveNotFoundError(Exception):
    """Exception to raise when specified objective does not exist."""
    pass


class MissingComponentError(Exception):
    """An exception raised when a component is not found in all_components()"""
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

    Arguments:
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
            exception_list.append(f"{objective} encountered {str(exception.__class__.__name__)} with message ({str(exception)}):\n")
            exception_list.extend(tb)
        message = "\n".join(exception_list)

        self.message = message
        super().__init__(message)


class DataCheckInitError(Exception):
    """Exception raised when a data check can't initialize with the parameters given."""


class NullsInColumnWarning(UserWarning):
    """Warning thrown when there are null values in the column of interest"""


class ObjectiveCreationError(Exception):
    """Exception when get_objective tries to instantiate an objective and required args are not provided."""


class NoPositiveLabelException(Exception):
    """Exception when a particular classification label for the 'positive' class cannot be found in the column index or unique values"""
