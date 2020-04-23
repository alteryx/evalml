import logging
import sys

from colorama import Style


class Logger:
    """Write log messages to stdout.

    Arguments:
        verbose (bool): If False, suppress log output. Default True.
    """

    def __init__(self, verbose=True):
        self.verbose = verbose
        logger = logging.getLogger('evalml')
        if not len(logger.handlers):
            out_handler = logging.StreamHandler(sys.stdout)
            date_fmt = '%m/%d/%Y %I:%M:%S %p'
            fmt = "%(message)s"
            # default_fmt = "%(asctime)s %(name)s - %(levelname)s: %(message)s"
            # err_fmt = "%(asctime)s %(name)s - %(levelname)s - %(module)s: %(lineno)d: %(message)s"
            # debug_fmt = "%(asctime)s %(name)s - %(levelname)s - %(module)s: %(lineno)d: %(message)s"
            out_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=date_fmt))
            logger.addHandler(out_handler)
            logger.setLevel('INFO')
        self.logger = logger

    def get_logger(self):
        return self.logger

    def warn(self):
        pass

    def error(self):
        pass

    def print(self):
        pass

    def log(self, msg, color=None, new_line=True):
        if not self.verbose:
            return

        if color:
            msg = color + msg + Style.RESET_ALL

        self.logger.info(msg)

    def log_title(self, title):
        self.log("*" * (len(title) + 4), color=Style.BRIGHT)
        self.log("* %s *" % title, color=Style.BRIGHT)
        self.log("*" * (len(title) + 4), color=Style.BRIGHT)
        self.log("")

    def log_subtitle(self, title, underline="=", color=None):
        self.log("")
        self.log("%s" % title, color=color)
        self.log(underline * len(title), color=color)
