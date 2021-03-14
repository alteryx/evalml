
from evalml.pipelines import PipelineBase
from evalml.problem_types import ProblemTypes
from evalml.utils import _convert_woodwork_types_wrapper, infer_feature_types


class RegressionPipeline(PipelineBase):
    """Pipeline subclass for all regression pipelines."""
    problem_type = ProblemTypes.REGRESSION

    def fit(self, X, y):
        """Build a regression model.

        Arguments:
            X (ww.DataTable, pd.DataFrame or np.ndarray): The input training data of shape [n_samples, n_features]
            y (ww.DataColumn, pd.Series, np.ndarray): The target training data of length [n_samples]

        Returns:
            self
        """
        X = infer_feature_types(X)
        y = infer_feature_types(y)
        if "numeric" not in y.semantic_tags:
            raise ValueError(f"Regression pipeline can only handle numeric target data")
        y = _convert_woodwork_types_wrapper(y.to_series())

        self._fit(X, y)
        return self

    def score(self, X, y, objectives):
        """Evaluate model performance on current and additional objectives

        Arguments:
            X (ww.DataTable, pd.DataFrame, or np.ndarray): Data of shape [n_samples, n_features]
            y (ww.DataColumn, pd.Series, or np.ndarray): True values of length [n_samples]
            objectives (list): Non-empty list of objectives to score on

        Returns:
            dict: Ordered dictionary of objective scores
        """
        objectives = self.create_objectives(objectives)
        y_predicted = _convert_woodwork_types_wrapper(self.predict(X).to_series())
        return self._score_all_objectives(X, y, y_predicted, y_pred_proba=None, objectives=objectives)
