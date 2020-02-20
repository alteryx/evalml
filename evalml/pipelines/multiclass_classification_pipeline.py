
import pandas as pd

from .pipeline_plots import PipelinePlots

from evalml.pipelines import PipelineBase


class MulticlassClassificationPipeline(PipelineBase):

    # Necessary for "Plotting" documentation, since Sphinx does not work well with instance attributes.
    plot = PipelinePlots
    threshold_selection_split = False  # primary difference between binary and multiclass

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
