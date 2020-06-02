from .data_checks import DataChecks
from .highly_null_data_check import HighlyNullDataCheck
from .id_columns_data_check import IDColumnsDataCheck
from .invalid_targets_data_check import InvalidTargetDataCheck
from .label_leakage_data_check import LabelLeakageDataCheck


class DefaultDataChecks(DataChecks):
    """A collection of basic data checks that is used by AutoML by default. Includes HighlyNullDataCheck, IDColumnsDataCheck, and LabelLeakageDataCheck."""

    def __init__(self, data_checks=None):
        """
        A collection of basic data checks.

        Arguments:
            data_checks (list (DataCheck)): Ignored.
        """
        self.data_checks = [HighlyNullDataCheck(), IDColumnsDataCheck(), LabelLeakageDataCheck(), InvalidTargetDataCheck()]
