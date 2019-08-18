from colorama import Style


class AutoBase:
    def _log(self, msg, color=None, new_line=True):
        if color:
            msg = color + msg + Style.RESET_ALL

        if new_line:
            print(msg)
        else:
            print(msg, end="")

    def _log_title(self, title):
        self._log("*" * (len(title) + 4), color=Style.BRIGHT)
        self._log("* %s *" % title, color=Style.BRIGHT)
        self._log("*" * (len(title) + 4), color=Style.BRIGHT)
        self._log("")

    def _log_subtitle(self, title, underline="=", color=None):
        self._log("%s" % title, color=color)
        self._log(underline * len(title), color=color)
