import pandas as pd
from sklearn.decomposition import PCA as SkPCA
from skopt.space import Real

from evalml.pipelines.components.transformers import Transformer
from evalml.utils import (
    _convert_woodwork_types_wrapper,
    _retain_custom_types_and_initalize_woodwork,
    infer_feature_types,
    is_all_numeric
)


class PCA(Transformer):
    """Reduces the number of features by using Principal Component Analysis"""
    name = 'PCA Transformer'
    hyperparameter_ranges = {
        "variance": Real(0.25, 1)}

    def __init__(self, variance=0.95, n_components=None, random_seed=0, **kwargs):
        """Initalizes an transformer that reduces the number of features using PCA."

        Arguments:
            variance (float): The percentage of the original data variance that should be preserved when reducing the
                              number of features.
            n_components (int): The number of features to maintain after computing SVD. Defaults to None, but will override
                                variance variable if set.
            random_seed (int): Seed for the random number generator. Defaults to 0.
        """
        parameters = {"variance": variance,
                      "n_components": n_components}
        parameters.update(kwargs)
        if n_components:
            pca = SkPCA(n_components=n_components, random_state=random_seed, **kwargs)
        else:
            pca = SkPCA(n_components=variance, random_state=random_seed, **kwargs)
        super().__init__(parameters=parameters,
                         component_obj=pca,
                         random_seed=random_seed)

    def fit(self, X, y=None):
        X = infer_feature_types(X)
        if not is_all_numeric(X):
            raise ValueError("PCA input must be all numeric")
        X = _convert_woodwork_types_wrapper(X.to_dataframe())
        self._component_obj.fit(X)
        return self

    def transform(self, X, y=None):
        X_ww = infer_feature_types(X)
        if not is_all_numeric(X_ww):
            raise ValueError("PCA input must be all numeric")
        X = _convert_woodwork_types_wrapper(X_ww.to_dataframe())
        X_t = self._component_obj.transform(X)
        X_t = pd.DataFrame(X_t, index=X.index, columns=[f"component_{i}" for i in range(X_t.shape[1])])
        return _retain_custom_types_and_initalize_woodwork(X_ww, X_t)

    def fit_transform(self, X, y=None):
        X_ww = infer_feature_types(X)
        if not is_all_numeric(X_ww):
            raise ValueError("PCA input must be all numeric")
        X = _convert_woodwork_types_wrapper(X_ww.to_dataframe())
        X_t = self._component_obj.fit_transform(X, y)
        X_t = pd.DataFrame(X_t, index=X.index, columns=[f"component_{i}" for i in range(X_t.shape[1])])
        return _retain_custom_types_and_initalize_woodwork(X_ww, X_t)
