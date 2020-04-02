from colorama import Style, Fore, Back

import logging
import sys


# Logging features we should have

# Support different log levels
# Include timestamp and file label in every log message
# All references to logger are static (this is already done I think)
# If AutoBase.__init__ has verbose=False, we should only log to file, except for fatal errors
# We should always log to a file in addition to stdout.
# Ability in code to enable/disable output to stdout/stderr
# Ability in code to configure verbosity level for stdout stream (or file stream)
# Ability in code to log to file instead of / in addition to stdout/stderr


def get_logger(logger_name):
    logger = logging.getLogger(logger_name)
    out_handler = logging.StreamHandler(sys.stdout)
    err_handler = logging.StreamHandler(sys.stdout)
    out_handler.setFormatter(LevelFormatter())
    # out_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=date_fmt))
    logger.setLevel('DEBUG')
    logger.addHandler(out_handler)
    return logger

def log(logger, msg, color=None):
    if color:
        msg = color + msg + Style.RESET_ALL

def log_title(logger, title):
    log(logger, "*" * (len(title) + 4), color=Fore.RED+Back.GREEN)
    log(logger, "* %s *" % title, color=Fore.RED)
    log(logger, "*" * (len(title) + 4), color=Fore.RED)
    log(logger, "")

def log_subtitle(logger, title, underline="=", color=None):
    log(logger, "")
    log(logger, "%s" % title, color=color)
    log(logger, underline * len(title), color=color)

# def log_subtitle(logger, title, underline="=", color=None):
#     self.log("")
#     self.log("%s" % title, color=color)
#     self.log(underline * len(title), color=color)



class LevelFormatter(logging.Formatter):

    date_fmt = '%m/%d/%Y %I:%M:%S %p'
    default_fmt = "%(asctime)s %(name)s - %(levelname)s: %(message)s"
    err_fmt  = "ERROR: %(message)s"
    debug_fmt  = "DEBUG: %(module)s: %(lineno)d: %(message)s"
    info_fmt = "%(message)s"

    def __init__(self):
        super().__init__(fmt=self.default_fmt, datefmt=self.date_fmt)

    def format(self, record):
        if record.levelno == logging.DEBUG:
            self._style._fmt = LevelFormatter.debug_fmt

        elif record.levelno == logging.INFO:
            self._style._fmt = LevelFormatter.info_fmt

        elif record.levelno == logging.ERROR:
            self._style._fmt = LevelFormatter.err_fmt

        result = logging.Formatter.format(self, record)
        return result




class Logger:

    def __init__(self, verbose=True):
        self.verbose = verbose

    def log(self, msg, color=None, new_line=True):
        if not self.verbose:
            return

        if color:
            msg = color + msg + Style.RESET_ALL

        if new_line:
            print(msg)
        else:
            print(msg, end="")

    def log_title(self, title):
        self.log("*" * (len(title) + 4), color=Style.BRIGHT)
        self.log("* %s *" % title, color=Style.BRIGHT)
        self.log("*" * (len(title) + 4), color=Style.BRIGHT)
        self.log("")

    def log_subtitle(self, title, underline="=", color=None):
        self.log("")
        self.log("%s" % title, color=color)
        self.log(underline * len(title), color=color)
