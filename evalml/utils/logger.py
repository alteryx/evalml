import logging
import sys
import time


def get_logger(name):
    logger = logging.getLogger(name)
    if not len(logger.handlers):
        logger.setLevel(logging.DEBUG)
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)
        stdout_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stdout_handler)
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
        return "{0:d}:{1:02d}:{2:02d}".format(h, m, s)
    else:
        return "{0:02d}:{1:02d}".format(m, s)
