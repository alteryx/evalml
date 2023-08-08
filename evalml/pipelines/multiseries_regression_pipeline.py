"""Pipeline base class for time series regression problems."""
from woodwork.statistics_utils import infer_frequency

from evalml.pipelines.time_series_regression_pipeline import (
    TimeSeriesRegressionPipeline,
)
from evalml.problem_types import ProblemTypes


class MultiseriesRegressionPipeline(TimeSeriesRegressionPipeline):
    """Pipeline base class for multiseries time series regression problems.

    Args:
        component_graph (ComponentGraph, list, dict): ComponentGraph instance, list of components in order, or dictionary of components.
        parameters (dict): Dictionary with component names as keys and dictionary of that component's parameters as values.
             An empty dictionary {} implies using all default values for component parameters. Pipeline-level
             parameters such as time_index, gap, and max_delay must be specified with the "pipeline" key. For example:
             Pipeline(parameters={"pipeline": {"time_index": "Date", "max_delay": 4, "gap": 2}}).
        custom_name (str): Custom name for the pipeline. Defaults to None.
        random_seed (int): Seed for the random number generator. Defaults to 0.

    """

    problem_type = ProblemTypes.TIME_SERIES_REGRESSION

    """ProblemTypes.TIME_SERIES_REGRESSION"""

    def __init__(
        self,
        component_graph,
        parameters=None,
        custom_name=None,
        random_seed=0,
    ):
        if not parameters or "pipeline" not in parameters:
            raise ValueError(
                "time_index, gap, max_delay, and forecast_horizon parameters cannot be omitted from the parameters dict. "
                "Please specify them as a dictionary with the key 'pipeline'.",
            )
        if "series_id" not in parameters["pipeline"]:
            raise ValueError(
                "series_id must be defined for multiseries time series pipelines. Please specify it as a key in the pipeline "
                "parameters dict.",
            )
        self.series_id = parameters["pipeline"]["series_id"]
        super().__init__(
            component_graph,
            custom_name=custom_name,
            parameters=parameters,
            random_seed=random_seed,
        )

    def fit(self, X, y):
        """Fit a multiseries time series pipeline.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series): The target training targets of length [n_samples*n_series].

        Returns:
            self

        Raises:
            ValueError: If the target is not numeric.
        """
        self._fit(X, y)
        return self

    def _fit(self, X, y):
        from evalml.pipelines.utils import unstack_multiseries

        self.input_target_name = y.name
        X_unstacked, y_unstacked = unstack_multiseries(
            X,
            y,
            self.series_id,
            self.time_index,
            self.input_target_name,
        )
        self.frequency = infer_frequency(X_unstacked[self.time_index])

        self.component_graph.fit(X_unstacked, y_unstacked)
        self.input_feature_names = self.component_graph.input_feature_names

    def predict_in_sample(
        self,
        X,
        y,
        X_train,
        y_train,
        objective=None,
        calculating_residuals=False,
    ):
        """Predict on future data where the target is known, e.g. cross validation.

        Args:
            X (pd.DataFrame or np.ndarray): Future data of shape [n_samples, n_features]
            y (pd.Series, np.ndarray): Future target of shape [n_samples]
            X_train (pd.DataFrame, np.ndarray): Data the pipeline was trained on of shape [n_samples_train, n_feautures]
            y_train (pd.Series, np.ndarray): Targets used to train the pipeline of shape [n_samples_train]
            objective (ObjectiveBase, str, None): Objective used to threshold predicted probabilities, optional.
            calculating_residuals (bool): Whether we're calling predict_in_sample to calculate the residuals.  This means
                the X and y arguments are not future data, but actually the train data.

        Returns:
            pd.Series: Estimated labels.

        Raises:
            ValueError: If final component is not an Estimator.
        """
        from evalml.pipelines.utils import stack_data, unstack_multiseries

        X_unstacked, y_unstacked = unstack_multiseries(
            X,
            y,
            self.series_id,
            self.time_index,
            self.input_target_name,
        )
        X_train_unstacked, y_train_unstacked = unstack_multiseries(
            X_train,
            y_train,
            self.series_id,
            self.time_index,
            self.input_target_name,
        )
        unstacked_predictions = super().predict_in_sample(
            X_unstacked,
            y_unstacked,
            X_train_unstacked,
            y_train_unstacked,
            objective,
            calculating_residuals,
        )
        stacked_predictions = stack_data(unstacked_predictions)

        # Index will start at the unstacked index, so we need to reset it to the original index
        stacked_predictions.index = X.index
        return stacked_predictions
