from colorama import Style


class Logger:
    """Write log messages to stdout.

    Arguments:
        verbose (bool): If False, suppress log output. Default True.
    """
    def __init__(self, verbose=True):
        self._verbose = verbose

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, verbose):
        self._verbose = verbose

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
