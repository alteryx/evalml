import io
import logging
import os
import sys
import traceback

from colorama import Style


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
                sio.write('Stack (most recent call last):\n')
                traceback.print_stack(f, file=sio)
                sinfo = sio.getvalue()
                if sinfo[-1] == '\n':
                    sinfo = sinfo[:-1]
                sio.close()
            rv = (co.co_filename, f.f_lineno, co.co_name, sinfo)
            break
        return rv


class Logger(logging.Logger):
    """Write log messages to stdout.

    Arguments:
        verbose (bool): If False, suppress log output. Default True.
    """

    def __init__(self, verbose=True):
        self.verbose = verbose
        logging.setLoggerClass(UpStackLogger)
        logger = logging.getLogger('evalml')
        logging.setLoggerClass(logging.Logger)

        if not len(logger.handlers):
            out_handler = logging.StreamHandler(sys.stdout)
            # date_fmt = '%m/%d/%Y %I:%M:%S %p'
            fmt = "%(module)s: %(lineno)d: %(message)s"

            # default_fmt = "%(asctime)s %(name)s - %(levelname)s: %(message)s"
            # err_fmt = "%(asctime)s %(name)s - %(levelname)s - %(module)s: %(lineno)d: %(message)s"
            # debug_fmt = "%(asctime)s %(name)s - %(levelname)s - %(module)s: %(lineno)d: %(message)s"
            out_handler.setFormatter(logging.Formatter(fmt=fmt))
            logger.addHandler(out_handler)
            logger.setLevel('INFO')
            # logger.addFilter(MyFilter())
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

        if new_line:
            self.logger.info(f"{msg}\n")
        else:
            self.logger.info(msg)

    def log_title(self, title):
        self.logger.info("*" * (len(title) + 4))
        self.logger.info("* %s *" % title)
        self.logger.info("*" * (len(title) + 4))
        self.logger.info("")

    def log_subtitle(self, title, underline="=", color=None):
        self.logger.info("")
        self.logger.info("%s" % title)
        self.logger.info(underline * len(title))
