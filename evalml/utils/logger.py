import logging
import sys
import time
from logging.handlers import RotatingFileHandler

import tqdm


def get_logger(name):
    logger = logging.getLogger(name)
    if not len(logger.handlers):
        date_fmt = "%m/%d/%Y %I:%M:%S %p"
        fmt = "%(asctime)s - %(levelname)s - %(filename)s: %(message)s"
        log_handler = RotatingFileHandler(filename="evalml_debug.log")
        log_handler.setLevel(logging.DEBUG)
        log_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=date_fmt))

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)
        stdout_handler.setFormatter(logging.Formatter('%(message)s'))

        logger.addHandler(stdout_handler)
        logger.addHandler(log_handler)
        logger.setLevel(logging.DEBUG)

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
