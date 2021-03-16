import numpy as np
import pandas as pd

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes
from evalml.utils import (
    _convert_woodwork_types_wrapper,
    infer_feature_types,
    pad_with_nans
)


class TimeSeriesBaselineEstimator(Estimator):
    """Time series estimator that predicts using the naive forecasting approach.

    This is useful as a simple baseline estimator for time series problems
    """
    name = "Time Series Baseline Estimator"
    hyperparameter_ranges = {}
    model_family = ModelFamily.BASELINE
    supported_problem_types = [ProblemTypes.TIME_SERIES_REGRESSION, ProblemTypes.TIME_SERIES_BINARY,
                               ProblemTypes.TIME_SERIES_MULTICLASS]
    predict_uses_y = True

    def __init__(self, gap=1, random_seed=0, **kwargs):
        """Baseline time series estimator that predicts using the naive forecasting approach.

        Arguments:
            gap (int): Gap between prediction date and target date and must be a positive integer. If gap is 0, target date will be shifted ahead by 1 time period.
            random_state (None, int): Deprecated - use random_seed instead.
            random_seed (int): Seed for the random number generator. Defaults to 0.

        """

        self._prediction_value = None
        self._num_features = None
        self.gap = gap

        if gap < 0:
            raise ValueError(f'gap value must be a positive integer. {gap} was provided.')

        parameters = {"gap": gap}
        parameters.update(kwargs)
        super().__init__(parameters=parameters,
                         component_obj=None,
                         random_seed=random_seed)

    def fit(self, X, y=None):
        if X is None:
            X = pd.DataFrame()
        X = infer_feature_types(X)
        self._num_features = X.shape[1]
        return self

    def predict(self, X, y=None):
        if y is None:
            raise ValueError("Cannot predict Time Series Baseline Estimator if y is None")
        y = infer_feature_types(y)
        y = _convert_woodwork_types_wrapper(y.to_series())

        if self.gap == 0:
            y = y.shift(periods=1)

        return infer_feature_types(y)

    def predict_proba(self, X, y=None):
        if y is None:
            raise ValueError("Cannot predict Time Series Baseline Estimator if y is None")
        y = infer_feature_types(y)
        y = _convert_woodwork_types_wrapper(y.to_series())
        preds = self.predict(X, y).to_series().dropna(axis=0, how='any').astype('int')
        proba_arr = np.zeros((len(preds), y.max() + 1))
        proba_arr[np.arange(len(preds)), preds] = 1
        padded = pad_with_nans(pd.DataFrame(proba_arr), len(y) - len(preds))
        return infer_feature_types(padded)

    @property
    def feature_importance(self):
        """Returns importance associated with each feature.

        Since baseline estimators do not use input features to calculate predictions, returns an array of zeroes.

        Returns:
            np.ndarray (float): an array of zeroes

        """
        return np.zeros(self._num_features)
