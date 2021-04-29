import pandas as pd

from evalml.pipelines.pipeline_meta import TimeSeriesPipelineBaseMeta
from evalml.pipelines.regression_pipeline import RegressionPipeline
from evalml.problem_types import ProblemTypes
from evalml.utils import (
    _convert_woodwork_types_wrapper,
    drop_rows_with_nans,
    infer_feature_types,
    pad_with_nans
)


class TimeSeriesRegressionPipeline(RegressionPipeline, metaclass=TimeSeriesPipelineBaseMeta):
    """Pipeline base class for time series regression problems."""

    problem_type = ProblemTypes.TIME_SERIES_REGRESSION

    def __init__(self, component_graph, parameters=None, custom_name=None, custom_hyperparameters=None, random_seed=0):
        """Machine learning pipeline for time series regression problems made out of transformers and a classifier.

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
        if "pipeline" not in parameters:
            raise ValueError("date_index, gap, and max_delay parameters cannot be omitted from the parameters dict. "
                             "Please specify them as a dictionary with the key 'pipeline'.")
        pipeline_params = parameters["pipeline"]
        self.date_index = pipeline_params['date_index']
        self.gap = pipeline_params['gap']
        self.max_delay = pipeline_params['max_delay']
        super().__init__(component_graph,
                         custom_name=custom_name,
                         parameters=parameters,
                         custom_hyperparameters=custom_hyperparameters,
                         random_seed=random_seed)

    def fit(self, X, y):
        """Fit a time series regression pipeline.

        Arguments:
            X (ww.DataTable, pd.DataFrame or np.ndarray): The input training data of shape [n_samples, n_features]
            y (ww.DataColumn, pd.Series, np.ndarray): The target training targets of length [n_samples]

        Returns:
            self
        """
        if X is None:
            X = pd.DataFrame()

        X = infer_feature_types(X)
        y = infer_feature_types(y)
        X = _convert_woodwork_types_wrapper(X.to_dataframe())
        y = _convert_woodwork_types_wrapper(y.to_series())
        X_t = self._compute_features_during_fit(X, y)
        X_t = X_t.to_dataframe()

        y_shifted = y.shift(-self.gap)
        X_t, y_shifted = drop_rows_with_nans(X_t, y_shifted)
        self.estimator.fit(X_t, y_shifted)
        self.input_feature_names = self._component_graph.input_feature_names

        return self

    def predict(self, X, y=None, objective=None):
        """Make predictions using selected features.

        Arguments:
            X (ww.DataTable, pd.DataFrame, or np.ndarray): Data of shape [n_samples, n_features]
            y (ww.DataColumn, pd.Series, np.ndarray, None): The target training targets of length [n_samples]
            objective (Object or string): The objective to use to make predictions

        Returns:
            ww.DataColumn: Predicted values.
        """
        if X is None:
            X = pd.DataFrame()
        X = infer_feature_types(X)
        y = infer_feature_types(y)
        X = _convert_woodwork_types_wrapper(X.to_dataframe())
        y = _convert_woodwork_types_wrapper(y.to_series())
        features = self.compute_estimator_features(X, y)
        features = _convert_woodwork_types_wrapper(features.to_dataframe())
        features_no_nan, y = drop_rows_with_nans(features, y)
        y_arg = None
        if self.estimator.predict_uses_y:
            y_arg = y
        predictions = self.estimator.predict(features_no_nan, y_arg).to_series()
        predictions = predictions.rename(self.input_target_name)
        padded = pad_with_nans(predictions, max(0, features.shape[0] - predictions.shape[0]))
        return infer_feature_types(padded)

    def score(self, X, y, objectives):
        """Evaluate model performance on current and additional objectives.

        Arguments:
            X (ww.DataTable, pd.DataFrame or np.ndarray): Data of shape [n_samples, n_features]
            y (pd.Series, ww.DataColumn): True labels of length [n_samples]
            objectives (list): Non-empty list of objectives to score on

        Returns:
            dict: Ordered dictionary of objective scores
        """
        # Only converting X for the call to _score_all_objectives
        if X is None:
            X = pd.DataFrame()
        X = infer_feature_types(X)
        X = _convert_woodwork_types_wrapper(X.to_dataframe())
        y = infer_feature_types(y)
        y = _convert_woodwork_types_wrapper(y.to_series())

        y_predicted = self.predict(X, y)
        y_predicted = _convert_woodwork_types_wrapper(y_predicted.to_series())

        y_shifted = y.shift(-self.gap)
        objectives = self.create_objectives(objectives)
        y_shifted, y_predicted = drop_rows_with_nans(y_shifted, y_predicted)
        return self._score_all_objectives(X, y_shifted,
                                          y_predicted,
                                          y_pred_proba=None,
                                          objectives=objectives)
