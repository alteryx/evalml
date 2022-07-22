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

        Examples:
            >>> import pandas as pd

            The time series parameters have to be compatible with the data passed. If the window size (gap + max_delay +
            forecast_horizon) is greater than or equal to the split size, then an error will be raised.

            >>> X = pd.DataFrame({
            ...    "dates": pd.date_range("1/1/21", periods=100),
            ...    "first": [i for i in range(100)],
            ... })
            >>> y = pd.Series([i for i in range(100)])
            ...
            >>> problem_config = {"gap": 7, "max_delay": 2, "forecast_horizon": 12, "time_index": "dates"}
            >>> ts_parameters_check = TimeSeriesParametersDataCheck(problem_configuration=problem_config, n_splits=7)
            >>> assert ts_parameters_check.validate(X, y) == [
            ...     {
            ...         "message": "Since the data has 100 observations, n_splits=7, and a forecast horizon of 12, the smallest "
            ...                    "split would have 16 observations. Since 21 (gap + max_delay + forecast_horizon)"
            ...                    " >= 16, then at least one of the splits would be empty by the time it reaches "
            ...                    "the pipeline. Please use a smaller number of splits, reduce one or more these "
            ...                    "parameters, or collect more data.",
            ...         "data_check_name": "TimeSeriesParametersDataCheck",
            ...         "level": "error",
            ...         "code": "TIMESERIES_PARAMETERS_NOT_COMPATIBLE_WITH_SPLIT",
            ...         "details": {
            ...             "columns": None,
            ...             "rows": None,
            ...             "max_window_size": 21,
            ...             "min_split_size": 16,
            ...             "n_obs": 100,
            ...             "n_splits": 7
            ...         },
            ...         "action_options": []
            ...     }
            ... ]

        """
        messages = []
        validation = are_ts_parameters_valid_for_split(
            gap=self.gap,
            max_delay=self.max_delay,
            forecast_horizon=self.forecast_horizon,
            n_splits=self.n_splits,
            n_obs=X.shape[0],
        )
        if not validation.is_valid:
            messages.append(
                DataCheckError(
                    message=validation.msg,
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.TIMESERIES_PARAMETERS_NOT_COMPATIBLE_WITH_SPLIT,
                    details={
                        "max_window_size": validation.max_window_size,
                        "min_split_size": validation.smallest_split_size,
                        "n_obs": validation.n_obs,
                        "n_splits": validation.n_splits,
                    },
                ).to_dict(),
            )
        return messages
