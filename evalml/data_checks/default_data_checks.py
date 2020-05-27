from .data_checks import DataChecks
from .detect_highly_null_data_check import DetectHighlyNullDataCheck
from .invalid_targets_data_check import InvalidTargetsDataCheck


class DefaultDataChecks(DataChecks):
    def __init__(self, data_checks=None):
        """
        A collection of basic data checks.

        Arguments:
            data_checks (list (DataCheck)): Ignored.
        """
        self.data_checks = [DetectHighlyNullDataCheck(), InvalidTargetsDataCheck()]
