import numpy as np
import pandas as pd

from evalml.pipelines import TimeSeriesRegressionPipeline
from evalml.utils.gen_utils import (
    _convert_to_woodwork_structure,
    _convert_woodwork_types_wrapper,
    drop_rows_with_nans,
    pad_with_nans
)


class TimeSeriesBaselineRegressionPipeline(TimeSeriesRegressionPipeline):
    """Baseline Pipeline for time series regression problems."""
    _name = "Time Series Baseline Regression Pipeline"
    component_graph = ["Time Series Baseline Regressor"]

    def fit(self, X, y):
        """Fit a time series regression pipeline.

        Arguments:
            X (ww.DataTable, pd.DataFrame, np.ndarray, or None): The input training data of shape [n_samples, n_features]
            y (ww.DataColumn, pd.Series, np.ndarray): The target training targets of length [n_samples]

        Returns:
            self
        """
        if X is None:
            X = pd.DataFrame()

        X = _convert_to_woodwork_structure(X)
        y = _convert_to_woodwork_structure(y)
        X = _convert_woodwork_types_wrapper(X.to_dataframe())
        y = _convert_woodwork_types_wrapper(y.to_series())

        X_t = self._compute_features_during_fit(X, y)
        if X_t.empty:
            X_t = pd.DataFrame(np.zeros(len(y)))

        y_shifted = y.shift(-self.gap)
        X_t, y_shifted = drop_rows_with_nans(X_t, y_shifted)
        self.estimator.fit(X_t, y_shifted)
        return self

    def predict(self, X, y, objective=None):
        """Make predictions using selected features.

        Arguments:
            X (ww.DataTable, pd.DataFrame, np.ndarray, or None): Data of shape [n_samples, n_features]
            y (ww.DataColumn, pd.Series, np.ndarray): The target training targets of length [n_samples]. y is required for this pipeline.
            objective (Object or string): The objective to use to make predictions

        Returns:
            pd.Series: Predicted values.
        """
        if X is None:
            X = pd.DataFrame()
        X = _convert_to_woodwork_structure(X)
        y = _convert_to_woodwork_structure(y)
        X = _convert_woodwork_types_wrapper(X.to_dataframe())
        y = _convert_woodwork_types_wrapper(y.to_series())

        predictions = self.estimator.predict(None, y)
        return pad_with_nans(predictions, max(0, X.shape[0] - predictions.shape[0]))
