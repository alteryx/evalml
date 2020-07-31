
import pandas as pd

from evalml.objectives import get_objective
from evalml.pipelines import PipelineBase
from evalml.problem_types import ProblemTypes
from evalml.utils.gen_utils import numeric_dtypes


class RegressionPipeline(PipelineBase):
    """Pipeline subclass for all regression pipelines."""
    problem_type = ProblemTypes.REGRESSION

    def fit(self, X, y):
        """Build a regression model.

        Arguments:
            X (pd.DataFrame or np.array): the input training data of shape [n_samples, n_features]

            y (pd.Series): the target training labels of length [n_samples]

        Returns:
            self

        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        valid_target_data_types = numeric_dtypes
        if y.dtype not in valid_target_data_types:
            raise ValueError(f"Regression pipeline cannot handle targets with dtype: {y.dtype}")
        self._fit(X, y)
        return self

    def score(self, X, y, objectives):
        """Evaluate model performance on current and additional objectives

        Args:
            X (pd.DataFrame or np.array): data of shape [n_samples, n_features]
            y (pd.Series): true labels of length [n_samples]
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
