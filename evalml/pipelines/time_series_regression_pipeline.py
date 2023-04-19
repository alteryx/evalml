"""Pipeline base class for time series regression problems."""
import numpy as np
import pandas as pd
from woodwork.statistics_utils import infer_frequency

from evalml.model_family import ModelFamily
from evalml.pipelines.components import STLDecomposer
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
        ...                                                       parameters={"Simple Imputer": {"impute_strategy": "mean"},
        ...                                                                   "pipeline": {"gap": 1, "max_delay": 1, "forecast_horizon": 1, "time_index": "date"}},
        ...                                                       custom_name="My TimeSeriesRegression Pipeline")
        ...
        >>> assert pipeline.custom_name == "My TimeSeriesRegression Pipeline"
        >>> assert pipeline.component_graph.component_dict.keys() == {'Simple Imputer', 'Linear Regressor'}

        The pipeline parameters will be chosen from the default parameters for every component, unless specific parameters
        were passed in as they were above.

        >>> assert pipeline.parameters == {
        ...     'Simple Imputer': {'impute_strategy': 'mean', 'fill_value': None},
        ...     'Linear Regressor': {'fit_intercept': True, 'n_jobs': -1},
        ...     'pipeline': {'gap': 1, 'max_delay': 1, 'forecast_horizon': 1, 'time_index': "date"}}
    """

    problem_type = ProblemTypes.TIME_SERIES_REGRESSION

    NO_PREDS_PI_ESTIMATORS = [
        ModelFamily.ARIMA,
        ModelFamily.EXPONENTIAL_SMOOTHING,
        ModelFamily.PROPHET,
    ]

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
            ...                                         parameters={"Simple Imputer": {"impute_strategy": "mean"},
            ...                                                     "pipeline": {"gap": gap, "max_delay": 1, "forecast_horizon": forecast_horizon, "time_index": "date"}},
            ...                                        )
            >>> pipeline.fit(X, y)
            pipeline = TimeSeriesRegressionPipeline(component_graph={'Linear Regressor': ['Linear Regressor', 'X', 'y']}, parameters={'Linear Regressor':{'fit_intercept': True, 'n_jobs': -1}, 'pipeline':{'gap': 1, 'max_delay': 1, 'forecast_horizon': 2, 'time_index': 'date'}}, random_seed=0)
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

    def get_prediction_intervals(
        self,
        X,
        y=None,
        X_train=None,
        y_train=None,
        coverage=None,
    ):
        """Find the prediction intervals using the fitted regressor.

        This function takes the predictions of the fitted estimator and calculates the rolling standard deviation across
        all predictions using a window size of 5. The lower and upper predictions are determined by taking the percent
        point (quantile) function of the lower tail probability at each bound multiplied by the rolling standard deviation.

        Certain estimators (Extra Trees Estimator, XGBoost Estimator, Prophet Estimator, ARIMA, and
        Exponential Smoothing estimator) utilize a different methodology to calculate prediction intervals.
        See the docs for these estimators to learn more.

        Args:
            X (pd.DataFrame): Data of shape [n_samples, n_features].
            y (pd.Series): Target data.
            X_train (pd.DataFrame, np.ndarray): Data the pipeline was trained on of shape [n_samples_train, n_features].
            y_train (pd.Series, np.ndarray): Targets used to train the pipeline of shape [n_samples_train].
            coverage (list[float]): A list of floats between the values 0 and 1 that the upper and lower bounds of the
                prediction interval should be calculated for.

        Returns:
            dict: Prediction intervals, keys are in the format {coverage}_lower or {coverage}_upper.

        Raises:
            MethodPropertyNotFoundError: If the estimator does not support Time Series Regression as a problem type.
        """
        X_no_datetime, y_no_datetime = self._drop_time_index(X, y)
        estimator_input = self.transform_all_but_final(
            X_no_datetime,
            y_no_datetime,
            X_train=X_train,
            y_train=y_train,
        )
        has_stl = STLDecomposer.name in list(
            self.component_graph.component_instances.keys(),
        )
        if coverage is None:
            coverage = [0.95]

        if self.estimator.model_family in self.NO_PREDS_PI_ESTIMATORS and has_stl:
            pred_intervals = self.estimator.get_prediction_intervals(
                X=estimator_input,
                y=y,
                coverage=coverage,
            )
            trans_pred_intervals = {}
            residuals = self.estimator.predict(
                estimator_input,
            )  # Get residual values
            trend_pred_intervals = self.get_component(
                "STL Decomposer",
            ).get_trend_prediction_intervals(y, coverage=coverage)
            for key, orig_pi_values in pred_intervals.items():
                trans_pred_intervals[key] = pd.Series(
                    (orig_pi_values.values - residuals.values)
                    + trend_pred_intervals[key].values
                    + y.values,
                    index=orig_pi_values.index,
                )
            return trans_pred_intervals
        else:
            future_vals = self.predict(
                X=X,
                X_train=X_train,
                y_train=y_train,
            )

            predictions_train = self.predict_in_sample(
                X=X_train,
                y=y_train,
                X_train=X_train,
                y_train=y_train,
                calculating_residuals=True,
            )
            if self.component_graph.has_dfs:
                predictions_train.index = y_train.index
            residuals = y_train - predictions_train
            std_residual = np.sqrt(np.sum(residuals**2) / len(residuals))

            res_dict = {}
            cov_to_mult = {0.75: 1.15, 0.85: 1.44, 0.95: 1.96}
            for cov in coverage:
                lower = []
                upper = []
                multiplier = cov_to_mult[cov]
                for counter, val in enumerate(future_vals):
                    factor = multiplier * std_residual * np.sqrt(counter + 1)
                    lower.append(val - factor)
                    upper.append(val + factor)

                res_dict[f"{cov}_lower"] = pd.Series(
                    lower,
                    name=f"{cov}_lower",
                    index=future_vals.index,
                )
                res_dict[f"{cov}_upper"] = pd.Series(
                    upper,
                    name=f"{cov}_upper",
                    index=future_vals.index,
                )
            return res_dict
