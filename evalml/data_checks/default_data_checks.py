from .highly_null_data_check import HighlyNullDataCheck
from .id_columns_data_check import IDColumnsDataCheck
from .invalid_targets_data_check import InvalidTargetDataCheck
from .label_leakage_data_check import LabelLeakageDataCheck
from .no_variance_data_check import NoVarianceDataCheck


DefaultDataChecks = [HighlyNullDataCheck, IDColumnsDataCheck,
                     LabelLeakageDataCheck, InvalidTargetDataCheck,
                     NoVarianceDataCheck]
