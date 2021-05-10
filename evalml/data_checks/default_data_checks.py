from .class_imbalance_data_check import ClassImbalanceDataCheck
from .data_checks import DataChecks
from .datetime_nan_data_check import DateTimeNaNDataCheck
from .highly_null_data_check import HighlyNullDataCheck
from .id_columns_data_check import IDColumnsDataCheck
from .invalid_targets_data_check import InvalidTargetDataCheck
from .natural_language_nan_data_check import NaturalLanguageNaNDataCheck
from .no_variance_data_check import NoVarianceDataCheck
from .target_leakage_data_check import TargetLeakageDataCheck

from evalml.problem_types import ProblemTypes, handle_problem_types


class DefaultDataChecks(DataChecks):
    """A collection of basic data checks that is used by AutoML by default.
    Includes:

        - `HighlyNullDataCheck`
        - `HighlyNullRowsDataCheck`
        - `IDColumnsDataCheck`
        - `TargetLeakageDataCheck`
        - `InvalidTargetDataCheck`
        - `NoVarianceDataCheck`
        - `ClassImbalanceDataCheck` (for classification problem types)
        - `DateTimeNaNDataCheck`
        - `NaturalLanguageNaNDataCheck`

    """

    _DEFAULT_DATA_CHECK_CLASSES = [HighlyNullDataCheck, IDColumnsDataCheck,
                                   TargetLeakageDataCheck, InvalidTargetDataCheck, NoVarianceDataCheck,
                                   NaturalLanguageNaNDataCheck, DateTimeNaNDataCheck]

    def __init__(self, problem_type, objective, n_splits=3):
        """
        A collection of basic data checks.

        Arguments:
            problem_type (str): The problem type that is being validated. Can be regression, binary, or multiclass.
            objective (str or ObjectiveBase): Name or instance of the objective class.
            n_splits (int): The number of splits as determined by the data splitter being used.
        """
        if handle_problem_types(problem_type) in [ProblemTypes.REGRESSION, ProblemTypes.TIME_SERIES_REGRESSION]:
            super().__init__(self._DEFAULT_DATA_CHECK_CLASSES,
                             data_check_params={"InvalidTargetDataCheck": {"problem_type": problem_type,
                                                                           "objective": objective}})
        else:
            super().__init__(self._DEFAULT_DATA_CHECK_CLASSES + [ClassImbalanceDataCheck],
                             data_check_params={"InvalidTargetDataCheck": {"problem_type": problem_type,
                                                                           "objective": objective},
                                                "ClassImbalanceDataCheck": {"num_cv_folds": n_splits}})
