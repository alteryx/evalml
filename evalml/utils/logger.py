import logging
import os
import sys
import time
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path


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
    logger.info("\n" + "*" * (len(title) + 4))
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

    time_diff = time.time() - start_time
    # Source: tqdm.std.tqdm.format_interval
    mins, s = divmod(int(time_diff), 60)
    h, m = divmod(mins, 60)
    if h:
        return '{0:d}:{1:02d}:{2:02d}'.format(h, m, s)
    else:
        return '{0:02d}:{1:02d}'.format(m, s)
