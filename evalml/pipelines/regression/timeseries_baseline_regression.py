import pandas as pd

from evalml.pipelines import TimeSeriesRegressionPipeline
from evalml.utils.gen_utils import (
    _convert_to_woodwork_structure,
    _convert_woodwork_types_wrapper,
    pad_with_nans
)


class TimeSeriesBaselineRegressionPipeline(TimeSeriesRegressionPipeline):
    """Baseline Pipeline for time series regression problems."""
    _name = "Time Series Baseline Regression Pipeline"
    component_graph = ["Time Series Baseline Regressor"]

    def predict(self, X, y, objective=None):
        """Make predictions using selected features.

        Arguments:
            X (ww.DataTable, pd.DataFrame, or np.ndarray): Data of shape [n_samples, n_features]
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
