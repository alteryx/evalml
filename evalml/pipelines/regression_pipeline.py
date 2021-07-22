from evalml.pipelines import PipelineBase
from evalml.problem_types import ProblemTypes
from evalml.utils import infer_feature_types


class RegressionPipeline(PipelineBase):
    """Pipeline subclass for all regression pipelines.

    Arguments:
        component_graph (list or dict): List of components in order. Accepts strings or ComponentBase subclasses in the list.
            Note that when duplicate components are specified in a list, the duplicate component names will be modified with the
            component's index in the list. For example, the component graph
            [Imputer, One Hot Encoder, Imputer, Logistic Regression Classifier] will have names
            ["Imputer", "One Hot Encoder", "Imputer_2", "Logistic Regression Classifier"]
        parameters (dict): Dictionary with component names as keys and dictionary of that component's parameters as values.
             An empty dictionary or None implies using all default values for component parameters. Defaults to None.
        custom_name (str): Custom name for the pipeline. Defaults to None.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    problem_type = ProblemTypes.REGRESSION
    """ProblemTypes.REGRESSION"""

    def fit(self, X, y):
        """Build a regression model.

        Arguments:
            X (pd.DataFrame or np.ndarray): The input training data of shape [n_samples, n_features]
            y (pd.Series, np.ndarray): The target training data of length [n_samples]

        Returns:
            self
        """
        X = infer_feature_types(X)
        y = infer_feature_types(y)
        if "numeric" not in y.ww.semantic_tags:
            raise ValueError(f"Regression pipeline can only handle numeric target data")

        self._fit(X, y)
        return self

    def score(self, X, y, objectives):
        """Evaluate model performance on current and additional objectives

        Arguments:
            X (pd.DataFrame, or np.ndarray): Data of shape [n_samples, n_features]
            y (pd.Series, or np.ndarray): True values of length [n_samples]
            objectives (list): Non-empty list of objectives to score on

        Returns:
            dict: Ordered dictionary of objective scores
        """
        objectives = self.create_objectives(objectives)
        y_predicted = self.predict(X)
        return self._score_all_objectives(
            X, y, y_predicted, y_pred_proba=None, objectives=objectives
        )

    def predict(self, X, objective=None):
        X = infer_feature_types(X)
        predictions = self.component_graph.predict(X)
        predictions = self.inverse_transform(predictions)
        predictions.name = self.input_target_name
        return infer_feature_types(predictions)
