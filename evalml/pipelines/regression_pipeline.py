
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

        objectives = [get_objective(o) for o in objectives]
        y_predicted = self.predict(X)
        return self._score_all_objectives(X, y, y_predicted, y_pred_proba=None, objectives=objectives)
