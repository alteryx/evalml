"""Data check that checks whether the time series parameters are compatible with the data size."""
from evalml.data_checks import DataCheck, DataCheckError, DataCheckMessageCode
from evalml.utils.gen_utils import (
    are_ts_parameters_valid_for_split,
    contains_all_ts_parameters,
)


class TimeSeriesParametersDataCheck(DataCheck):
    """Checks whether the time series parameters are compatible with data splitting.

    If `gap + max_delay + forecast_horizon > X.shape[0] // (n_splits + 1)`

    then the feature engineering window is larger than the smallest split. This will cause the
    pipeline to create features from data that does not exist, which will cause errors.

    Args:
        problem_configuration (dict): Dict containing problem_configuration parameters.
        n_splits (int): Number of time series splits.
    """

    def __init__(self, problem_configuration, n_splits):
        is_valid, msg = contains_all_ts_parameters(problem_configuration)
        if not is_valid:
            raise ValueError(msg)

        self.gap = problem_configuration["gap"]
        self.forecast_horizon = problem_configuration["forecast_horizon"]
        self.max_delay = problem_configuration["max_delay"]
        self.n_splits = n_splits

    def validate(self, X, y=None):
        """Check if the time series parameters are compatible with data splitting.

        Args:
            X (pd.DataFrame, np.ndarray): Features.
            y (pd.Series, np.ndarray): Ignored. Defaults to None.

        Returns:
            dict: dict with a DataCheckError if parameters are too big for the split sizes.

        """
        results = {"warnings": [], "errors": [], "actions": []}

        validation = are_ts_parameters_valid_for_split(
            gap=self.gap,
            max_delay=self.max_delay,
            forecast_horizon=self.forecast_horizon,
            n_splits=self.n_splits,
            n_obs=X.shape[0],
        )
        if not validation.is_valid:
            results["errors"].append(
                DataCheckError(
                    message=validation.msg,
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.TIMESERIES_PARAMETERS_NOT_COMPATIBLE_WITH_SPLIT,
                    details={
                        "max_window_size": validation.max_window_size,
                        "min_split_size": validation.smallest_split_size,
                    },
                ).to_dict()
            )
        return results
