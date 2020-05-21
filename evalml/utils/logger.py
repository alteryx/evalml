import logging
import sys
from logging.handlers import RotatingFileHandler


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
        logger.setLevel(logging.INFO)

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
