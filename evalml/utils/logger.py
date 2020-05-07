import io
import logging
import os
import sys
import traceback


class UpStackLogger(logging.Logger):
    """A custom logger class used to skip a stack frame when computing filename, line number, function name and stack information."""

    def findCaller(self, stack_info=False, stacklevel=2):
        """
        Find the stack frame of the caller so that we can note the source
        file name, line number and function name.

        This is taken directly from Python source code (https://github.com/python/cpython/blob/master/Lib/logging/__init__.py); the only modification is to the default stacklevel parameter value.
        """
        f = logging.currentframe()
        if f is not None:
            f = f.f_back
        orig_f = f
        while f and stacklevel > 1:
            f = f.f_back
            stacklevel -= 1
        if not f:
            f = orig_f
        rv = "(unknown file)", 0, "(unknown function)", None
        while hasattr(f, "f_code"):
            co = f.f_code
            filename = os.path.normcase(co.co_filename)
            if filename == logging._srcfile:
                f = f.f_back
                continue
            sinfo = None
            if stack_info:
                sio = io.StringIO()
                sio.write("Stack (most recent call last):\n")
                traceback.print_stack(f, file=sio)
                sinfo = sio.getvalue()
                if sinfo[-1] == "\n":
                    sinfo = sinfo[:-1]
                sio.close()
            rv = (co.co_filename, f.f_lineno, co.co_name, sinfo)
            break
        return rv


class Logger(logging.Logger):
    """Write log messages to stdout.

    Arguments:
        level (str): level of Logger. Valid options are "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL". Defaults to "INFO".
    """

    def __init__(self, level="INFO"):
        logging.setLoggerClass(UpStackLogger)
        logger = logging.getLogger("evalml")
        logging.setLoggerClass(logging.Logger)

        if not len(logger.handlers):
            out_handler = logging.StreamHandler(sys.stdout)
            date_fmt = "%m/%d/%Y %I:%M:%S %p"
            fmt = "%(asctime)s %(name)s - %(levelname)s - %(module)s: %(lineno)d: %(message)s"
            out_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=date_fmt))
            logger.addHandler(out_handler)

        self.level = level
        logger.setLevel(level)
        self.logger = logger

    def get_logger(self):
        """Gets underlying logger object."""
        return self.logger

    def warn(self, msg, stack_info=False):
        """Logs a warning message."""
        self.logger.warn(msg, stack_info=stack_info)

    def error(self, msg, stack_info=False):
        """Logs a error message."""
        self.logger.error(msg, stack_info=stack_info)

    def critical(self, msg, stack_info=False):
        """Logs a critical message."""
        self.logger.critical(msg, stack_info=stack_info)

    def print(self, msg, new_line=False, log=True):
        """Prints to console and optionally logs."""
        if new_line:
            print(msg)
        else:
            print(msg, end="")
        if log:
            self.logger.info(msg)

    def log(self, msg, print_stdout=False, new_line=True, level="INFO"):
        """Logs message."""
        if new_line:
            self.logger.info(f"{msg}\n")
        else:
            self.logger.info(msg)
        if print_stdout:
            self.print(msg, new_line=new_line, log=False)

    def log_title(self, title):
        self.logger.info("*" * (len(title) + 4))
        self.logger.info("* %s *" % title)
        self.logger.info("*" * (len(title) + 4))
        self.logger.info("")

    def log_subtitle(self, title, underline="="):
        self.logger.info("")
        self.logger.info("%s" % title)
        self.logger.info(underline * len(title))
