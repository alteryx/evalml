
import pandas as pd

from evalml.objectives import get_objective
from evalml.pipelines import PipelineBase
from evalml.problem_types import ProblemTypes


class RegressionPipeline(PipelineBase):
    """Pipeline subclass for all regression pipelines."""
    problem_type = ProblemTypes.REGRESSION

    def score(self, X, y, objectives):
        """Evaluate model performance on current and additional objectives

        Args:
            X (pd.DataFrame or np.array) : data of shape [n_samples, n_features]
            y (pd.Series) : true labels of length [n_samples]
            objectives (list): Non-empty list of objectives to score on

        Returns:
            dict: ordered dictionary of objective scores
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        def is_objective_suitable_for_regression(objective):
            if objective.score_needs_proba:
                raise ValueError(f"Objective `{objective.name}` is not suited for regression problems.")

        objectives = [get_objective(o) for o in objectives]
        y_predicted = self.predict(X)
        return self._score(X, y, y_predicted, predicted_probabilities=None, objectives=objectives,
                           is_objective_suitable=is_objective_suitable_for_regression)
