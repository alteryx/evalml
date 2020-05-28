from .data_checks import DataChecks
from .highly_null_data_check import HighlyNullDataCheck


class DefaultDataChecks(DataChecks):
    def __init__(self, data_checks=None):
        """
        A collection of basic data checks.

        Arguments:
            data_checks (list (DataCheck)): Ignored.
        """
        self.data_checks = [HighlyNullDataCheck()]
