"""Logging functions."""
import logging
import sys
import time


def get_logger(name):
    """Get the logger with the associated name.

    Args:
        name (str): Name of the logger to get.

    Returns:
        The logger object with the associated name.
    """
    logger = logging.getLogger(name)
    if not len(logger.handlers):
        logger.setLevel(logging.DEBUG)
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)
        stdout_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stdout_handler)
    return logger


def log_title(logger, title):
    """Log with a title."""
    logger.info("\n" + "*" * (len(title) + 4))
    logger.info("* %s *" % title)
    logger.info("*" * (len(title) + 4))
    logger.info("")


def log_subtitle(logger, title, underline="="):
    """Log with a subtitle."""
    logger.info("")
    logger.info("%s" % title)
    logger.info(underline * len(title))


def time_elapsed(start_time):
    """How much time has elapsed since the search started.

    Args:
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


def log_batch_times(logger, batch_times):
    """Used to print out the batch times.

    Args:
        logger: the logger.
        batch_times: dict with (batch number, {pipeline name, pipeline time}).
    """
    log_title(logger, "Batch Time Stats")
    for batch_number in batch_times:
        subtitle = "Batch " + str(batch_number) + " time stats:"
        log_subtitle(logger, subtitle)
        for pipeline_name in batch_times[batch_number]:
            logger.info(
                "\n" + pipeline_name + ": " + batch_times[batch_number][pipeline_name],
            )
        logger.info("")
