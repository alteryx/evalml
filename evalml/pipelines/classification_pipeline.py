

import pandas as pd

from evalml.pipelines import PipelineBase


class ClassificationPipeline(PipelineBase):

    component_graph = []

    def predict_proba(self, X):
        """Make probability estimates for labels.

        Arguments:
            X (pd.DataFrame or np.array) : data of shape [n_samples, n_features]

        Returns:
            pd.DataFrame : probability estimates
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X = self._transform(X)
        proba = self.estimator.predict_proba(X)
        return proba
