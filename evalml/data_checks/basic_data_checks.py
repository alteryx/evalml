from .data_checks import DataChecks
from .detect_highly_null_data_check import DetectHighlyNullDataCheck


class BasicDataChecks(DataChecks):
    def __init__(self, data_checks=None):
        """
        A collection of data checks.

        Arguments:
            data_checks (list (DataCheck)): Ignored.
        """
        self.data_checks = [DetectHighlyNullDataCheck()]
