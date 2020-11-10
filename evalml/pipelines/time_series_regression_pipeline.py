import pandas as pd

from evalml.objectives import get_objective
from evalml.pipelines.regression_pipeline import RegressionPipeline
from evalml.problem_types import ProblemTypes
from evalml.utils.gen_utils import any_values_are_nan, drop_nan, pad_with_nans


class TimeSeriesRegressionPipeline(RegressionPipeline):
    """Pipeline base class for time series regression problems."""

    problem_type = ProblemTypes.TIME_SERIES_REGRESSION

    def __init__(self, parameters, random_state=0):
        """Machine learning pipeline for time series regression problems made out of transformers and a classifier.

        Required Class Variables:
            component_graph (list): List of components in order. Accepts strings or ComponentBase subclasses in the list

        Arguments:
            parameters (dict): Dictionary with component names as keys and dictionary of that component's parameters as values.
                 An empty dictionary {} implies using all default values for component parameters. Pipeline-level
                 parameters such as gap and max_delay must be specified as a dictionary with the "pipeline" key.
            random_state (int, np.random.RandomState): The random seed/state. Defaults to 0.
        """
        if "pipeline" not in parameters:
            raise ValueError("gap and max_delay parameters cannot be omitted from the parameters dict. "
                             "Please specify them as a dictionary with the key 'pipeline'.")
        pipeline_params = parameters.pop("pipeline")
        self.gap = pipeline_params['gap']
        self.max_delay = pipeline_params['max_delay']
        super().__init__(parameters, random_state)

    def fit(self, X, y=None):
        """Fit a time series regression pipeline.

        Arguments:
            X (ww.DataTable, pd.DataFrame or np.ndarray): The input training data of shape [n_samples, n_features]
            y (ww.DataColumn, pd.Series, np.ndarray): The target training targets of length [n_samples]

        Returns:
            self
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if X.empty:
            X_t = self._compute_features_during_fit(X=None, y=y)
        else:
            y = pd.Series(y)
            X_t = self._compute_features_during_fit(X, y)

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
        return pad_with_nans(predictions, self.max_delay if any_values_are_nan(features) else 0)

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
        y_shifted = y.shift(-self.gap)
        objectives = [get_objective(o, return_instance=True) for o in objectives]
        y_shifted, y_predicted = drop_nan(y_shifted, y_predicted)
        return self._score_all_objectives(X, y_shifted,
                                          y_predicted,
                                          y_pred_proba=None,
                                          objectives=objectives)
