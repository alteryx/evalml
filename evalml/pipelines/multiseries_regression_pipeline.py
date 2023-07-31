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
        self.frequency = infer_frequency(X[self.time_index])
        self._fit(X, y)
        return self

    def _fit(self, X, y):
        self.input_target_name = y.name

        from evalml.pipelines.utils import unstack_multiseries

        X_unstacked, y_unstacked = unstack_multiseries(
            X,
            y,
            self.series_id,
            self.time_index,
            self.input_target_name,
        )

        self.component_graph.fit(X_unstacked, y_unstacked)
        self.input_feature_names = self.component_graph.input_feature_names
