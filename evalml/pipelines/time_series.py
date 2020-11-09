import pandas as pd

from evalml.objectives import get_objective
from evalml.pipelines.regression_pipeline import RegressionPipeline
from evalml.problem_types import ProblemTypes
from evalml.utils.gen_utils import any_values_are_nan, drop_nan, pad_with_nans


class TimeSeriesRegressionPipeline(RegressionPipeline):
    problem_type = ProblemTypes.TIME_SERIES_REGRESSION

    def __init__(self, parameters, gap, max_delay, random_state=0):
        super().__init__(parameters, random_state)
        self.gap = gap
        self.max_delay = max_delay

    def fit(self, X, y=None):
        """Fit a time series regression pipeline.

        Arguments:
            X (ww.DataTable, pd.DataFrame or np.ndarray): The input training data of shape [n_samples, n_features]
            y (ww.DataColumn, pd.Series, np.ndarray): The target training targets of length [n_samples]

        Returns:
            self
        """
        if y is None:
            assert isinstance(X, pd.Series), "When modeling a single time series, the data must be a series."
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if y is None:
            X_t, _ = self._compute_features_during_fit(X, y=None)
            y = X.squeeze()
        else:
            y = pd.Series(y)
            X_t, _ = self._compute_features_during_fit(X, y)

        y_shifted = y.shift(-self.gap)
        X_t, y_shifted = drop_nan(X_t, y_shifted)
        self.estimator.fit(X_t, y_shifted)
        return self

    def predict(self, X, y=None, objective=None):
        """Make predictions using selected features.

        Arguments:
            X (ww.DataTable, pd.DataFrame, or np.ndarray): Data of shape [n_samples, n_features]
            y (ww.DataColumn, pd.Series, np.ndarray, None): The target training targets of length [n_samples]
            objective (Object or string): The objective to use to make predictions

        Returns:
            pd.Series: Predicted values.
        """
        features = self.compute_estimator_features(X, y)
        predictions = self.estimator.predict(features.dropna(axis=0, how="any"))
        if any_values_are_nan(features):
            return pad_with_nans(predictions, self.max_delay)
        else:
            return predictions

    def score(self, X, y, objectives):
        """Evaluate model performance on current and additional objectives.

        Arguments:
            X (ww.DataTable, pd.DataFrame or np.ndarray): Data of shape [n_samples, n_features]
            y (pd.Series, ww.DataColumn): True labels of length [n_samples]
            objectives (list): Non-empty list of objectives to score on

        Returns:
            dict: Ordered dictionary of objective scores
        """
        y_predicted = self.predict(X, y)
        if y is None:
            y = X.copy()
        y_shifted = y.shift(-self.gap)
        objectives = [get_objective(o, return_instance=True) for o in objectives]
        y_shifted, y_predicted = drop_nan(y_shifted, y_predicted)
        return self._score_all_objectives(X, y_shifted,
                                          y_predicted,
                                          y_pred_proba=None,
                                          objectives=objectives)
