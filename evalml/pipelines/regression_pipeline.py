
import numpy as np
import pandas as pd

from evalml.objectives import get_objective
from evalml.pipelines import PipelineBase
from evalml.problem_types import ProblemTypes
from evalml.utils.gen_utils import (
    _convert_to_woodwork_structure,
    _convert_woodwork_types_wrapper,
    numeric_dtypes
)


class RegressionPipeline(PipelineBase):
    """Pipeline subclass for all regression pipelines."""
    problem_type = ProblemTypes.REGRESSION

    def fit(self, X, y):
        """Build a regression model.

        Arguments:
            X (pd.DataFrame or np.array): The input training data of shape [n_samples, n_features]
            y (pd.Series): The target training data of length [n_samples]

        Returns:
            self

        """
        X = _convert_to_woodwork_structure(X)
        y = _convert_to_woodwork_structure(y)
        X = _convert_woodwork_types_wrapper(X.to_pandas())
        y = _convert_woodwork_types_wrapper(y.to_pandas())
        if y.dtype not in numeric_dtypes:
            raise ValueError(f"Regression pipeline cannot handle targets with dtype: {y.dtype}")
        self._fit(X, y)
        return self

    def score(self, X, y, objectives):
        """Evaluate model performance on current and additional objectives

        Arguments:
            X (pd.DataFrame or np.array): Data of shape [n_samples, n_features]
            y (pd.Series): True values of length [n_samples]
            objectives (list): Non-empty list of objectives to score on

        Returns:
            dict: Ordered dictionary of objective scores
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        objectives = [get_objective(o, return_instance=True) for o in objectives]
        y_predicted = self.predict(X)
        return self._score_all_objectives(X, y, y_predicted, y_pred_proba=None, objectives=objectives)
