import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SkLDA

from evalml.pipelines.components.transformers import Transformer
from evalml.utils import (
    _convert_woodwork_types_wrapper,
    _retain_custom_types_and_initalize_woodwork,
    infer_feature_types,
    is_all_numeric
)


class LinearDiscriminantAnalysis(Transformer):
    """Reduces the number of features by using Linear Discriminant Analysis"""
    name = 'Linear Discriminant Analysis Transformer'
    hyperparameter_ranges = {}

    def __init__(self, n_components=None, random_seed=0, **kwargs):
        """Initalizes an transformer that reduces the number of features using linear discriminant analysis."

        Arguments:
            n_components (int): The number of features to maintain after computation. Defaults to None.
            random_seed (int): Seed for the random number generator. Defaults to 0.
        """
        if n_components and n_components < 1:
            raise ValueError("Invalid number of compponents for Linear Discriminant Analysis")
        parameters = {"n_components": n_components}
        parameters.update(kwargs)
        lda = SkLDA(n_components=n_components, **kwargs)
        super().__init__(parameters=parameters,
                         component_obj=lda,
                         random_seed=random_seed)

    def fit(self, X, y):
        X = infer_feature_types(X)
        if not is_all_numeric(X):
            raise ValueError("LDA input must be all numeric")
        y = infer_feature_types(y)
        X = _convert_woodwork_types_wrapper(X.to_dataframe())
        y = _convert_woodwork_types_wrapper(y.to_series())
        n_features = X.shape[1]
        n_classes = y.nunique()
        n_components = self.parameters['n_components']
        if n_components is not None and n_components > min(n_classes, n_features):
            raise ValueError(f"n_components value {n_components} is too large")

        self._component_obj.fit(X, y)
        return self

    def transform(self, X, y=None):
        X_ww = infer_feature_types(X)
        if not is_all_numeric(X_ww):
            raise ValueError("LDA input must be all numeric")
        X = _convert_woodwork_types_wrapper(X_ww.to_dataframe())
        X_t = self._component_obj.transform(X)
        X_t = pd.DataFrame(X_t, index=X.index, columns=[f"component_{i}" for i in range(X_t.shape[1])])
        return _retain_custom_types_and_initalize_woodwork(X_ww, X_t)

    def fit_transform(self, X, y=None):
        X_ww = infer_feature_types(X)
        if not is_all_numeric(X_ww):
            raise ValueError("LDA input must be all numeric")
        y = infer_feature_types(y)
        X = _convert_woodwork_types_wrapper(X_ww.to_dataframe())
        y = _convert_woodwork_types_wrapper(y.to_series())

        X_t = self._component_obj.fit_transform(X, y)
        X_t = pd.DataFrame(X_t, index=X.index, columns=[f"component_{i}" for i in range(X_t.shape[1])])
        return _retain_custom_types_and_initalize_woodwork(X_ww, X_t)
