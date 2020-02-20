from collections import OrderedDict

import pandas as pd
from sklearn.model_selection import train_test_split

from .components import Estimator, handle_component
from .pipeline_plots import PipelinePlots

from evalml.objectives import get_objective
from evalml.pipelines import PipelineBase
from evalml.utils import Logger


class MulticlassClassificationPipeline(PipelineBase):

    # Necessary for "Plotting" documentation, since Sphinx does not work well with instance attributes.
    plot = PipelinePlots
    threshold_selection_split = False

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

        if proba.shape[1] <= 2:
            raise ValueError("Expected more than two classes for multiclass problem.")
        return proba
