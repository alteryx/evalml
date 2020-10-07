import pandas as pd
from sklearn.decomposition import PCA as SkPCA
from skopt.space import Real

from evalml.pipelines.components.transformers import Transformer
from evalml.utils.gen_utils import is_all_numeric


class PCA(Transformer):
    """Reduces the number of features by using Principal Component Analysis"""
    name = 'PCA Transformer'
    hyperparameter_ranges = {
        "variance": Real(0.25, 1)}

    def __init__(self, variance=0.95, n_components=None, random_state=0, **kwargs):
        """Initalizes an transformer that imputes missing data according to the specified imputation strategy."

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
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not is_all_numeric(X):
            raise ValueError("PCA input must be numeric")

        self._component_obj.fit(X)
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not is_all_numeric(X):
            raise ValueError("PCA input must be numeric")

        X_t = self._component_obj.transform(X)
        return pd.DataFrame(X_t, index=X.index)

    def fit_transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not is_all_numeric(X):
            raise ValueError("PCA input must be numeric")

        X_t = self._component_obj.fit_transform(X, y)
        return pd.DataFrame(X_t, index=X.index)
