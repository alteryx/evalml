"""Data check that checks whether the time series parameters are compatible with the data size."""
from evalml.data_checks import DataCheck, DataCheckError, DataCheckMessageCode
from evalml.problem_types import handle_problem_types
from evalml.utils.gen_utils import (
    contains_all_ts_parameters,
    is_ts_valid_for_split,
)


class TimeSeriesParametersDataCheck(DataCheck):
    """Checks whether the time series parameters and target data are compatible with data splitting.

    If `gap + max_delay + forecast_horizon > X.shape[0] // (n_splits + 1)`

    then the feature engineering window is larger than the smallest split. This will cause the
    pipeline to create features from data that does not exist, which will cause errors.
    If the target data in the first split doesn't have representation from all classes (for time
    series classification problems) this will prevent the estimators from training on all potential
    outcomes which will cause errors during prediction.

    Args:
        problem_configuration (dict): Dict containing problem_configuration parameters.
        problem_type (str or ProblemTypes): Problem type.
        n_splits (int): Number of time series splits.
    """

    def __init__(self, problem_configuration, problem_type, n_splits):
        is_valid, msg = contains_all_ts_parameters(problem_configuration)
        if not is_valid:
            raise ValueError(msg)

        self.gap = problem_configuration["gap"]
        self.forecast_horizon = problem_configuration["forecast_horizon"]
        self.max_delay = problem_configuration["max_delay"]
        self.problem_type = handle_problem_types(problem_type)
        self.n_splits = n_splits

    def validate(self, X, y=None):
        """Check if the time series parameters are compatible with data splitting.

        Args:
            X (pd.DataFrame, np.ndarray): Features.
            y (pd.Series, np.ndarray): Defaults to None.

        Returns:
            dict: dict with a DataCheckError if parameters are too big for the split sizes or if target
            values have inadequate representation in the first split (for time series classification).

        Example:
            >>> import pandas as pd
            ...
            >>> X = pd.DataFrame(pd.date_range("1/1/21", periods=100), columns=["dates"])
            >>> y = pd.Series([0 if i < 25 else 1 for i in range(100)])
            ...
            >>> problem_config = {"gap": 1, "max_delay": 23, "forecast_horizon": 1, "date_index": "dates"}
            >>> ts_params_check = TimeSeriesParametersDataCheck(problem_configuration=problem_config, problem_type="time series binary", n_splits=3)
            >>> assert ts_params_check.validate(X, y) == {
            ...     "errors": [{'message': 'Since the data has 100 observations and n_splits=3, '
            ...                            'the smallest split would have 25 observations. Since '
            ...                            '25 (gap + max_delay + forecast_horizon) >= 25, then at '
            ...                            'least one of the splits would be empty by the time it '
            ...                            'reaches the pipeline. Please use a smaller number of '
            ...                            'splits, reduce one or more of these parameters, or '
            ...                            'collect more data.',
            ...                 'data_check_name': 'TimeSeriesParametersDataCheck',
            ...                 'level': 'error',
            ...                 'details': {'columns': None, 'rows': None, 'max_window_size': 25, 'min_split_size': 25},
            ...                 'code': 'TIMESERIES_PARAMETERS_NOT_COMPATIBLE_WITH_SPLIT'},
            ...                 {'message': 'Since the data has 100 observations and n_splits=3, '
            ...                             'the smallest split would have 25 observations. Time '
            ...                             'Series Binary and Time Series Multiclass problem types '
            ...                             'require every training and validation split to have at '
            ...                             'least one instance of all the target classes. '
            ...                             'The following splits are invalid: [1, 2, 3]',
            ...                  'data_check_name': 'TimeSeriesParametersDataCheck',
            ...                  'level': 'error',
            ...                  'details': {'columns': None, 'rows': None, 'invalid_splits': [1, 2, 3]},
            ...                  'code': 'TIMESERIES_TARGET_NOT_COMPATIBLE_WITH_SPLIT'}],
            ...     "warnings": [],
            ...     "actions": []}

        """
        results = {"warnings": [], "errors": [], "actions": []}

        validation = is_ts_valid_for_split(
            gap=self.gap,
            max_delay=self.max_delay,
            forecast_horizon=self.forecast_horizon,
            n_splits=self.n_splits,
            n_obs=X.shape[0],
            problem_type=self.problem_type,
            y=y,
        )
        if not validation.is_valid:
            if "(gap + max_delay + forecast_horizon)" in validation.msg:
                results["errors"].append(
                    DataCheckError(
                        message=validation.msg[: validation.msg.index("data.") + 5],
                        data_check_name=self.name,
                        message_code=DataCheckMessageCode.TIMESERIES_PARAMETERS_NOT_COMPATIBLE_WITH_SPLIT,
                        details={
                            "max_window_size": validation.max_window_size,
                            "min_split_size": validation.smallest_split_size,
                        },
                    ).to_dict()
                )
            if validation.invalid_splits:
                results["errors"].append(
                    DataCheckError(
                        message=validation.msg[
                            : validation.msg.index("observations.") + 14
                        ]
                        + validation.msg[validation.msg.index("Time Series Binary") :],
                        data_check_name=self.name,
                        message_code=DataCheckMessageCode.TIMESERIES_TARGET_NOT_COMPATIBLE_WITH_SPLIT,
                        details={
                            "invalid_splits": validation.invalid_splits,
                        },
                    ).to_dict()
                )
        return results
