import logging
import sys
from logging.handlers import RotatingFileHandler


def get_logger(name):
    logger = logging.getLogger(name)
    if not len(logger.handlers):
        date_fmt = "%m/%d/%Y %I:%M:%S %p"
        # format='%(asctime)s,%(msecs)d %(levelname)-8s [name=%(name)s] [%(filename)s:%(lineno)d] %(message)s'
        fmt = "%(asctime)s %(name)s - %(levelname)s - [name=%(name)s]: %(message)s"

        log_handler = RotatingFileHandler(filename="debug.log")
        log_handler.setLevel(logging.DEBUG)
        log_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=date_fmt))

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)
        stdout_handler.setFormatter(logging.Formatter('%(message)s'))

        logger.addHandler(stdout_handler)
        logger.addHandler(log_handler)
        logger.setLevel(logging.INFO)

    return logger
