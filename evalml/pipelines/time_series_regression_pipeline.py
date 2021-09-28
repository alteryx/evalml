"""Pipeline base class for time series regression problems."""
from evalml.pipelines.time_series_pipeline_base import TimeSeriesPipelineBase
from evalml.problem_types import ProblemTypes


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
             parameters such as date_index, gap, and max_delay must be specified with the "pipeline" key. For example:
             Pipeline(parameters={"pipeline": {"date_index": "Date", "max_delay": 4, "gap": 2}}).
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    problem_type = ProblemTypes.TIME_SERIES_REGRESSION
    """ProblemTypes.TIME_SERIES_REGRESSION"""

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
            X, y, y_predicted, y_pred_proba=None, objectives=objectives
        )
