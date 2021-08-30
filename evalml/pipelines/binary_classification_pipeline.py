from .binary_classification_pipeline_mixin import (
    BinaryClassificationPipelineMixin,
)

from evalml.objectives import get_objective
from evalml.pipelines.classification_pipeline import ClassificationPipeline
from evalml.problem_types import ProblemTypes
from evalml.utils import infer_feature_types


class BinaryClassificationPipeline(
    BinaryClassificationPipelineMixin, ClassificationPipeline
):
    """Pipeline subclass for all binary classification pipelines.

    Parameters
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

    problem_type = ProblemTypes.BINARY
    """ProblemTypes.BINARY"""

    def _predict(self, X, objective=None):
        """Make predictions using selected features.

        Parameters
            X (pd.DataFrame): Data of shape [n_samples, n_features]
            objective (Object or string): The objective to use to make predictions

        Returns:
            pd.Series: Estimated labels
        """

        if objective is not None:
            objective = get_objective(objective, return_instance=True)
            if not objective.is_defined_for_problem_type(self.problem_type):
                raise ValueError(
                    "You can only use a binary classification objective to make predictions for a binary classification pipeline."
                )

        if self.threshold is None:
            return self.component_graph.predict(X)
        ypred_proba = self.predict_proba(X)
        predictions = self._predict_with_objective(X, ypred_proba, objective)
        return infer_feature_types(predictions)

    def predict_proba(self, X):
        """Make probability estimates for labels. Assumes that the column at index 1 represents the positive label case.

        Parameters
            X (pd.DataFrame or np.ndarray): Data of shape [n_samples, n_features]

        Returns:
            pd.Series: Probability estimates
        """
        return super().predict_proba(X)

    @staticmethod
    def _score(X, y, predictions, objective):
        """Given data, model predictions or predicted probabilities computed on the data, and an objective, evaluate and return the objective score."""
        if predictions.ndim > 1:
            predictions = predictions.iloc[:, 1]
        return ClassificationPipeline._score(X, y, predictions, objective)
