import logging
import os
import sys
import time
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path

import tqdm


def get_logger(name):
    logger = logging.getLogger(name)
    if not len(logger.handlers):
        logger.setLevel(logging.DEBUG)
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)
        stdout_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stdout_handler)

        evalml_log_path_str = os.environ.get('EVALML_LOG_FILE', 'evalml_debug.log')
        evalml_log_path = Path(evalml_log_path_str)
        warning_msg = 'Continuing without logging to file. To change this, please set the EVALML_LOG_FILE environment variable to a valid file path with write permissions available. To disable debug logging, please set the EVALML_LOG_FILE environment variable to an empty value, or simply ignore this warning.'
        if len(evalml_log_path_str) == 0:
            return logger
        if evalml_log_path.is_dir() or not os.access(evalml_log_path.parent, os.W_OK):
            print(f'Warning: cannot write debug logs to path "{evalml_log_path}". ' + warning_msg)
            return logger
        try:
            date_fmt = "%m/%d/%Y %I:%M:%S %p"
            fmt = "%(asctime)s - %(levelname)s - %(filename)s: %(message)s"
            log_handler = RotatingFileHandler(filename=evalml_log_path)
            log_handler.setLevel(logging.DEBUG)
            log_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=date_fmt))
            logger.addHandler(log_handler)
        except Exception as e:
            logger.warning(f'Exception encountered while setting up debug log file at path {evalml_log_path}: {str(e)}')
            logger.warning(''.join(traceback.format_tb(sys.exc_info()[2])))
            logger.warning(warning_msg)
    return logger


def log_title(logger, title):
    logger.info("*" * (len(title) + 4))
    logger.info("* %s *" % title)
    logger.info("*" * (len(title) + 4))
    logger.info("")


def log_subtitle(logger, title, underline="="):
    logger.info("")
    logger.info("%s" % title)
    logger.info(underline * len(title))


def time_elapsed(start_time):
    """How much time has elapsed since the search started.

    Arguments:
        start_time (int): Time when search started.

    Returns:
        str: elapsed time formatted as a string [H:]MM:SS
    """
    return tqdm.std.tqdm.format_interval(time.time() - start_time)


def update_pipeline(logger, pipeline_name, current_iteration, max_pipelines, start_time):
    """Adds the next pipeline to be evaluated to the log along with how much time has elapsed.

    Arguments:
        logger (logging.Logger): Logger where we will record progress.
        pipeline_name (str): Name of next pipeline to be evaluated.
        current_iteration (int): How many pipelines have been evaluated during the search so far.
        max_pipelines (int, None): Max number of pipelines to search.
        start_time (int): Start time.

    Returns:
        None: logs progress to logger at info level.
    """
    if max_pipelines:
        status_update_format = "({current_iteration}/{max_pipelines}) {pipeline_name} Elapsed:{time_elapsed}"
        format_params = {'max_pipelines': max_pipelines, 'current_iteration': current_iteration}
    else:
        status_update_format = "{pipeline_name} Elapsed: {time_elapsed}"
        format_params = {}

    elapsed_time = time_elapsed(start_time)
    format_params.update({'pipeline_name': pipeline_name, 'time_elapsed': elapsed_time})
    logger.info(status_update_format.format(**format_params))
