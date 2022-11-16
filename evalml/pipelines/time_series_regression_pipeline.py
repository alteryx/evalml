"""Pipeline base class for time series regression problems."""
import pandas as pd
from woodwork.statistics_utils import infer_frequency

from evalml.pipelines.time_series_pipeline_base import TimeSeriesPipelineBase
from evalml.problem_types import ProblemTypes
from evalml.utils.woodwork_utils import infer_feature_types


class TimeSeriesRegressionPipeline(TimeSeriesPipelineBase):
    """Pipeline base class for time series regression problems.

    Args:
        component_graph (ComponentGraph, list, dict): ComponentGraph instance, list of components in order, or dictionary of components.
            Accepts strings or ComponentBase subclasses in the list.
            Note that when duplicate components are specified in a list, the duplicate component names will be modified with the
            component's index in the list. For example, the component graph
            [Imputer, One Hot Encoder, Imputer, Logistic Regression Classifier] will have names
            ["Imputer", "One Hot Encoder", "Imputer_2", "Logistic Regression Classifier"]
        parameters (dict): Dictionary with component names as keys and dictionary of that component's parameters as values.
             An empty dictionary {} implies using all default values for component parameters. Pipeline-level
             parameters such as time_index, gap, and max_delay must be specified with the "pipeline" key. For example:
             Pipeline(parameters={"pipeline": {"time_index": "Date", "max_delay": 4, "gap": 2}}).
        random_seed (int): Seed for the random number generator. Defaults to 0.

    Example:
        >>> pipeline = TimeSeriesRegressionPipeline(component_graph=["Simple Imputer", "Linear Regressor"],
        ...                                                       parameters={"Linear Regressor": {"normalize": True},
        ...                                                                   "pipeline": {"gap": 1, "max_delay": 1, "forecast_horizon": 1, "time_index": "date"}},
        ...                                                       custom_name="My TimeSeriesRegression Pipeline")
        ...
        >>> assert pipeline.custom_name == "My TimeSeriesRegression Pipeline"
        >>> assert pipeline.component_graph.component_dict.keys() == {'Simple Imputer', 'Linear Regressor'}

        The pipeline parameters will be chosen from the default parameters for every component, unless specific parameters
        were passed in as they were above.

        >>> assert pipeline.parameters == {
        ...     'Simple Imputer': {'impute_strategy': 'most_frequent', 'fill_value': None},
        ...     'Linear Regressor': {'fit_intercept': True, 'normalize': True, 'n_jobs': -1},
        ...     'pipeline': {'gap': 1, 'max_delay': 1, 'forecast_horizon': 1, 'time_index': "date"}}
    """

    problem_type = ProblemTypes.TIME_SERIES_REGRESSION
    """ProblemTypes.TIME_SERIES_REGRESSION"""

    def fit(self, X, y):
        """Fit a time series pipeline.

        Args:
            X (pd.DataFrame or np.ndarray): The input training data of shape [n_samples, n_features].
            y (pd.Series, np.ndarray): The target training targets of length [n_samples].

        Returns:
            self

        Raises:
            ValueError: If the target is not numeric.
        """
        X, y = self._convert_to_woodwork(X, y)
        self.frequency = infer_frequency(X[self.time_index])

        if "numeric" not in y.ww.semantic_tags:
            raise ValueError(
                "Time Series Regression pipeline can only handle numeric target data!",
            )

        X, y = self._drop_time_index(X, y)
        self._fit(X, y)
        return self

    def score(self, X, y, objectives, X_train=None, y_train=None):
        """Evaluate model performance on current and additional objectives.

        Args:
            X (pd.DataFrame or np.ndarray): Data of shape [n_samples, n_features].
            y (pd.Series): True labels of length [n_samples].
            objectives (list): Non-empty list of objectives to score on.
            X_train (pd.DataFrame, np.ndarray): Data the pipeline was trained on of shape [n_samples_train, n_feautures].
            y_train (pd.Series, np.ndarray): Targets used to train the pipeline of shape [n_samples_train].

        Returns:
            dict: Ordered dictionary of objective scores.
        """
        X, y = self._convert_to_woodwork(X, y)
        X_train, y_train = self._convert_to_woodwork(X_train, y_train)
        objectives = self.create_objectives(objectives)
        y_predicted = self.predict_in_sample(X, y, X_train, y_train)
        return self._score_all_objectives(
            X,
            y,
            y_predicted,
            y_pred_proba=None,
            objectives=objectives,
        )

    def get_forecast_period(self, X):
        """Generates all possible forecasting time points based on latest data point in X.

        Args:
            X (pd.DataFrame, np.ndarray): Data the pipeline was trained on of shape [n_samples_train, n_feautures].

        Raises:
            ValueError: If pipeline is not trained.

        Returns:
            pd.Series: Datetime periods out to `forecast_horizon + gap`.

        Example:
            >>> X = pd.DataFrame({'date': pd.date_range(start='1-1-2022', periods=10, freq='D'), 'feature': range(10, 20)})
            >>> y = pd.Series(range(0, 10), name='target')
            >>> gap = 1
            >>> forecast_horizon = 2
            >>> pipeline = TimeSeriesRegressionPipeline(component_graph=["Linear Regressor"],
            ...                                         parameters={"Linear Regressor": {"normalize": True},
            ...                                                     "pipeline": {"gap": gap, "max_delay": 1, "forecast_horizon": forecast_horizon, "time_index": "date"}},
            ...                                        )
            >>> pipeline.fit(X, y)
            pipeline = TimeSeriesRegressionPipeline(component_graph={'Linear Regressor': ['Linear Regressor', 'X', 'y']}, parameters={'Linear Regressor':{'fit_intercept': True, 'normalize': True, 'n_jobs': -1}, 'pipeline':{'gap': 1, 'max_delay': 1, 'forecast_horizon': 2, 'time_index': 'date'}}, random_seed=0)
            >>> dates = pipeline.get_forecast_period(X)
            >>> expected = pd.Series(pd.date_range(start='2022-01-11', periods=(gap + forecast_horizon), freq='D'), name='date', index=[10, 11, 12])
            >>> assert dates.equals(expected)
        """
        if not self._is_fitted:
            raise ValueError("Pipeline must be fitted before getting forecast.")

        X = infer_feature_types(X)

        # Generate prediction periods
        first_date = X.iloc[-1][self.time_index]
        predicted_date_range = pd.Series(
            pd.date_range(
                start=first_date,
                periods=self.forecast_horizon
                + self.gap
                + 1,  # Add additional period to account for dropping first date row
                freq=self.frequency,
            ),
        )

        # Generate numerical index
        first_idx = len(X) - 1 if not isinstance(X.index.dtype, int) else X.index[-1]
        num_idx = pd.Series(range(first_idx, first_idx + predicted_date_range.size))
        predicted_date_range.index = num_idx

        predicted_date_range = predicted_date_range.drop(predicted_date_range.index[0])
        predicted_date_range.name = self.time_index
        return predicted_date_range

    def get_forecast_predictions(self, X, y):
        """Generates all possible forecasting predictions based on last period of X.

        Args:
            X (pd.DataFrame, np.ndarray): Data the pipeline was trained on of shape [n_samples_train, n_feautures].
            y (pd.Series, np.ndarray): Targets used to train the pipeline of shape [n_samples_train].

        Returns:
            Predictions out to `forecast_horizon + gap` periods.
        """
        X, y = self._convert_to_woodwork(X, y)
        pred_dates = pd.DataFrame(self.get_forecast_period(X))
        preds = self.predict(pred_dates, objective=None, X_train=X, y_train=y)
        return preds
