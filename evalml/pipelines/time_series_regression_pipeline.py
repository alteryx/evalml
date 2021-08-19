from evalml.pipelines.time_series_pipeline_base import TimeSeriesPipelineBase
from evalml.problem_types import ProblemTypes
from evalml.utils import (
    drop_rows_with_nans,
    infer_feature_types,
    pad_with_nans,
)


class TimeSeriesRegressionPipeline(TimeSeriesPipelineBase):
    """Pipeline base class for time series regression problems.

    Arguments:
        component_graph (list or dict): List of components in order. Accepts strings or ComponentBase subclasses in the list.
            Note that when duplicate components are specified in a list, the duplicate component names will be modified with the
            component's index in the list. For example, the component graph
            [Imputer, One Hot Encoder, Imputer, Logistic Regression Classifier] will have names
            ["Imputer", "One Hot Encoder", "Imputer_2", "Logistic Regression Classifier"]
        parameters (dict): Dictionary with component names as keys and dictionary of that component's parameters as values.
             An empty dictionary {} implies using all default values for component parameters. Pipeline-level
             parameters such as date_index, gap, and max_delay must be specified with the "pipeline" key. For example:
             Pipeline(parameters={"pipeline": {"date_index": "Date", "max_delay": 4, "gap": 2}}).
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    problem_type = ProblemTypes.TIME_SERIES_REGRESSION
    """ProblemTypes.TIME_SERIES_REGRESSION"""

    def predict(self, X, y=None, objective=None):
        """Make predictions using selected features.

        Arguments:
            X (pd.DataFrame, or np.ndarray): Data of shape [n_samples, n_features].
            y (pd.Series, np.ndarray, None): The target training targets of length [n_samples].
            objective (Object or string): The objective to use to make predictions.

        Returns:
            pd.Series: Predicted values.
        """
        if self.estimator is None:
            raise ValueError(
                "Cannot call predict() on a component graph because the final component is not an Estimator."
            )
        X, y = self._convert_to_woodwork(X, y)
        features = self.compute_estimator_features(X, y)
        features_no_nan, y = drop_rows_with_nans(features, y)
        predictions = self._estimator_predict(features_no_nan, y)
        predictions.index = y.index
        predictions = self.inverse_transform(predictions)
        predictions = predictions.rename(self.input_target_name)
        padded = pad_with_nans(
            predictions, max(0, features.shape[0] - predictions.shape[0])
        )
        return infer_feature_types(padded)

    def score(self, X, y, objectives):
        """Evaluate model performance on current and additional objectives.

        Arguments:
            X (pd.DataFrame or np.ndarray): Data of shape [n_samples, n_features].
            y (pd.Series): True labels of length [n_samples].
            objectives (list): Non-empty list of objectives to score on.

        Returns:
            dict: Ordered dictionary of objective scores.
        """
        X, y = self._convert_to_woodwork(X, y)
        objectives = self.create_objectives(objectives)
        y_predicted = self.predict(X, y)
        y_shifted = y.shift(-self.gap)
        y_shifted, y_predicted = drop_rows_with_nans(y_shifted, y_predicted)
        return self._score_all_objectives(
            X, y_shifted, y_predicted, y_pred_proba=None, objectives=objectives
        )
