from .data_checks import DataChecks


class EmptyDataChecks(DataChecks):
    def __init__(self, data_checks=None):
        """
        An empty collection of data checks.

        Arguments:
            data_checks (list (DataCheck)): Ignored.
        """
        self.data_checks = []
