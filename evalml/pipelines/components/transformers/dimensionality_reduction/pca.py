import pandas as pd
from sklearn.decomposition import PCA as SkPCA
from skopt.space import Real

from evalml.pipelines.components.transformers import Transformer
from evalml.utils.gen_utils import (
    _convert_to_woodwork_structure,
    _convert_woodwork_types_wrapper,
    is_all_numeric
)


class PCA(Transformer):
    """Reduces the number of features by using Principal Component Analysis"""
    name = 'PCA Transformer'
    hyperparameter_ranges = {
        "variance": Real(0.25, 1)}

    def __init__(self, variance=0.95, n_components=None, random_state=0, **kwargs):
        """Initalizes an transformer that reduces the number of features using PCA."

        Arguments:
            variance (float): the percentage of the original data variance that should be preserved when reducing the
                              number of features.
            n_components (int): the number of features to maintain after computing SVD. Defaults to None, but will override
                                variance variable if set.
        """
        parameters = {"variance": variance,
                      "n_components": n_components}
        parameters.update(kwargs)
        if n_components:
            pca = SkPCA(n_components=n_components, **kwargs)
        else:
            pca = SkPCA(n_components=variance, **kwargs)
        super().__init__(parameters=parameters,
                         component_obj=pca,
                         random_state=random_state)

    def fit(self, X, y=None):
        X = _convert_to_woodwork_structure(X)
        X = _convert_woodwork_types_wrapper(X.to_dataframe())
        if not is_all_numeric(X):
            raise ValueError("PCA input must be all numeric")

        self._component_obj.fit(X)
        return self

    def transform(self, X, y=None):
        X = _convert_to_woodwork_structure(X)
        X = _convert_woodwork_types_wrapper(X.to_dataframe())
        if not is_all_numeric(X):
            raise ValueError("PCA input must be all numeric")

        X_t = self._component_obj.transform(X)
        return pd.DataFrame(X_t, index=X.index, columns=[f"component_{i}" for i in range(X_t.shape[1])])

    def fit_transform(self, X, y=None):
        X = _convert_to_woodwork_structure(X)
        X = _convert_woodwork_types_wrapper(X.to_dataframe())
        if not is_all_numeric(X):
            raise ValueError("PCA input must be all numeric")

        X_t = self._component_obj.fit_transform(X, y)
        return pd.DataFrame(X_t, index=X.index, columns=[f"component_{i}" for i in range(X_t.shape[1])])
