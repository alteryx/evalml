import pandas as pd

from evalml.pipelines import PipelineBase
from evalml.pipelines.pipeline_meta import PipelineBaseMeta
from evalml.utils import drop_rows_with_nans, infer_feature_types


class TimeSeriesPipelineBase(PipelineBase, metaclass=PipelineBaseMeta):

    """Pipeline base class for time series problems.

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

    def __init__(
        self,
        component_graph,
        parameters=None,
        custom_name=None,
        random_seed=0,
    ):
        if not parameters or "pipeline" not in parameters:
            raise ValueError(
                "date_index, gap, and max_delay parameters cannot be omitted from the parameters dict. "
                "Please specify them as a dictionary with the key 'pipeline'."
            )
        pipeline_params = parameters["pipeline"]
        self.date_index = pipeline_params["date_index"]
        self.gap = pipeline_params["gap"]
        self.max_delay = pipeline_params["max_delay"]
        super().__init__(
            component_graph,
            custom_name=custom_name,
            parameters=parameters,
            random_seed=random_seed,
        )

    @staticmethod
    def _convert_to_woodwork(X, y):
        if X is None:
            X = pd.DataFrame()
        X = infer_feature_types(X)
        y = infer_feature_types(y)
        return X, y

    def fit(self, X, y):
        """Fit a time series pipeline.

        Arguments:
            X (pd.DataFrame or np.ndarray): The input training data of shape [n_samples, n_features].
            y (pd.Series, np.ndarray): The target training targets of length [n_samples].

        Returns:
            self
        """
        X, y = self._convert_to_woodwork(X, y)
        self._fit(X, y)
        return self

    def _fit(self, X, y):
        self.input_target_name = y.name
        X_t = self.component_graph.fit_features(X, y)
        y_shifted = y.shift(-self.gap)
        X_t, y_shifted = drop_rows_with_nans(X_t, y_shifted)

        if self.estimator is not None:
            self.estimator.fit(X_t, y_shifted)
        else:
            self.component_graph.get_last_component().fit(X_t, y)

        self.input_feature_names = self.component_graph.input_feature_names

    def _estimator_predict(self, features, y):
        """Get estimator predictions.

        This helper passes y as an argument if needed by the estimator.
        """
        y_arg = None
        if self.estimator.predict_uses_y:
            y_arg = y
        return self.estimator.predict(features, y=y_arg)
