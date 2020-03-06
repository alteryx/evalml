
import pandas as pd

from .pipeline_plots import PipelinePlots

from evalml.pipelines import PipelineBase


class ClassificationPipeline(PipelineBase):

    plot = PipelinePlots

    def predict_proba(self, X):
        """Make probability estimates for labels.

        Args:
            X (pd.DataFrame or np.array) : data of shape [n_samples, n_features]

        Returns:
            pd.DataFrame : probability estimates
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X = self._transform(X)
        proba = self.estimator.predict_proba(X)
        return proba
