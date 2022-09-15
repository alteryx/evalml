"""Callbacks available to pass to AutoML."""
import logging

from evalml.exceptions import PipelineScoreError

logger = logging.getLogger(__name__)


def silent_error_callback(exception, traceback, automl, **kwargs):
    """No-op."""


def raise_error_callback(exception, traceback, automl, **kwargs):
    """Raises the exception thrown by the AutoMLSearch object.

    Also logs the exception as an error.

    Args:
        exception: Exception to log and raise.
        traceback: Exception traceback to log.
        automl: AutoMLSearch object.
        **kwargs: Other relevant keyword arguments to log.

    Raises:
        exception: Raises the input exception.
    """
    logger.error(f"AutoML search raised a fatal exception: {str(exception)}")
    logger.error("\n".join(traceback))
    automl.errors["Raised error"] = {
        "Traceback": traceback,
        "Exception": exception,
    }
    raise exception


def log_error_callback(exception, traceback, automl, **kwargs):
    """Logs the exception thrown as an error.

    Will not throw. This is the default behavior for AutoMLSearch.

    Args:
        exception: Exception to log.
        traceback: Exception traceback to log.
        automl: AutoMLSearch object.
        **kwargs: Other relevant keyword arguments to log.
    """
    fold_num = kwargs.get("fold_num")
    pipeline = kwargs.get("pipeline")
    trace = "\n".join(traceback) if traceback else ""
    if isinstance(exception, PipelineScoreError):
        logger.warning(
            f"\t\t\t{pipeline.name} fold {fold_num}: Encountered an error scoring the following objectives: {', '.join(exception.exceptions)}.",
        )
        logger.warning(
            f"\t\t\t{pipeline.name} fold {fold_num}: The scores for these objectives will be replaced with nan.",
        )
        trace += f"\n{exception.message}"
    else:
        logger.warning(f"\t\t\t{pipeline.name} fold {fold_num}: Encountered an error.")
        logger.warning(
            f"\t\t\t{pipeline.name} fold {fold_num}: All scores will be replaced with nan.",
        )
    logger.error(
        f"\t\t\tFold {fold_num}: Exception during automl search: {str(exception)}",
    )
    logger.error(f"\t\t\tFold {fold_num}: Parameters:\n\t{pipeline.parameters}")
    logger.error(f"\t\t\tFold {fold_num}: Traceback:\n{trace}")
    automl.errors[f"{pipeline.name}_fold_{fold_num}"] = {
        "Parameters": pipeline.parameters,
        "Traceback": trace,
        "Exception": exception,
    }
