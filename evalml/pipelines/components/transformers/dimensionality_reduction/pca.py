import numpy as np
import pandas as pd
from sklearn.decomposition import PCA as SkPCA
from skopt.space import Real

from evalml.pipelines.components.transformers import Transformer


class PCA(Transformer):
    """Reduces the number of features by using Principal Component Analysis"""
    name = 'PCA'
    hyperparameter_ranges = {
        "variance": Real(0.01, 1)}

    def __init__(self, variance=0.95, random_state=0, **kwargs):
        """Initalizes an transformer that imputes missing data according to the specified imputation strategy."

        Arguments:
            variance (float): the percentage of the original data variance that should be preserved when reducing the
                              number of features.
        """
        parameters = {"variance": variance}
        parameters.update(kwargs)
        pca = SkPCA(n_components=variance,
                    **kwargs)
        super().__init__(parameters=parameters,
                         component_obj=pca,
                         random_state=random_state)
