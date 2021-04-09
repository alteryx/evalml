from evalml.exceptions import PipelineScoreError
from evalml.utils.logger import get_logger

logger = get_logger(__file__)


def silent_error_callback(exception, traceback, automl, **kwargs):
    """No-op."""


def raise_error_callback(exception, traceback, automl, **kwargs):
    """Raises the exception thrown by the AutoMLSearch object. Also logs the exception as an error."""
    logger.error(f'AutoML search raised a fatal exception: {str(exception)}')
    logger.error("\n".join(traceback))
    raise exception


def log_error_callback(exception, traceback, automl, **kwargs):
    """Logs the exception thrown as an error. Will not throw. This is the default behavior for AutoMLSearch."""
    fold_num = kwargs.get('fold_num')
    pipeline = kwargs.get('pipeline')
    trace = "\n".join(traceback) if traceback else ""
    if isinstance(exception, PipelineScoreError):
        logger.info(f"\t\t\tFold {fold_num}: Encountered an error scoring the following objectives: {', '.join(exception.exceptions)}.")
        logger.info(f"\t\t\tFold {fold_num}: The scores for these objectives will be replaced with nan.")
        trace += f"\n{exception.message}"
    else:
        logger.info(f"\t\t\tFold {fold_num}: Encountered an error.")
        logger.info(f"\t\t\tFold {fold_num}: All scores will be replaced with nan.")
    logger.info(f"\t\t\tFold {fold_num}: Please check {logger.handlers[1].baseFilename} for the current hyperparameters and stack trace.")
    logger.info(f"\t\t\tFold {fold_num}: Exception during automl search: {str(exception)}")
    logger.debug(f"\t\t\tFold {fold_num}: Hyperparameters:\n\t{pipeline.hyperparameters}")
    logger.debug(f"\t\t\tFold {fold_num}: Traceback:\n{trace}")
