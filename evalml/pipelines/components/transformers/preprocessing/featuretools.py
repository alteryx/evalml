from featuretools import EntitySet, calculate_feature_matrix, dfs

from evalml.pipelines.components.transformers.transformer import Transformer
from evalml.utils import (
    _retain_custom_types_and_initalize_woodwork,
    infer_feature_types
)


class DFSTransformer(Transformer):
    """Featuretools DFS component that generates features for pd.DataFrames"""
    name = "DFS Transformer"
    hyperparameter_ranges = {}

    def __init__(self, index='index', random_seed=0, **kwargs):
        """Allows for featuretools to be used in EvalML.

        Arguments:
            index (string): The name of the column that contains the indices. If no column with this name exists,
                then featuretools.EntitySet() creates a column with this name to serve as the index column. Defaults to 'index'
            random_seed (int): Seed for the random number generator
        """
        parameters = {"index": index}
        if not isinstance(index, str):
            raise TypeError(f"Index provided must be string, got {type(index)}")

        self.index = index
        self.features = None
        parameters.update(kwargs)
        super().__init__(parameters=parameters,
                         random_seed=random_seed)

    def _make_entity_set(self, X):
        """Helper method that creates and returns the entity set given the input data"""
        ft_es = EntitySet()
        if self.index not in X.columns:
            es = ft_es.entity_from_dataframe(entity_id="X", dataframe=X, index=self.index, make_index=True)
        else:
            es = ft_es.entity_from_dataframe(entity_id="X", dataframe=X, index=self.index)
        return es

    def fit(self, X, y=None):
        """Fits the DFSTransformer Transformer component.

        Arguments:
            X (pd.DataFrame, np.array): The input data to transform, of shape [n_samples, n_features]
            y (pd.Series, np.ndarray, optional): The target training data of length [n_samples]

        Returns:
            self
        """
        X_ww = infer_feature_types(X)
        X_ww = X_ww.ww.rename({col: str(col) for col in X_ww.columns})
        es = self._make_entity_set(X_ww)
        self.features = dfs(entityset=es,
                            target_entity='X',
                            features_only=True,
                            max_depth=1)
        return self

    def transform(self, X, y=None):
        """Computes the feature matrix for the input X using featuretools' dfs algorithm.

        Arguments:
            X (pd.DataFrame or np.ndarray): The input training data to transform. Has shape [n_samples, n_features]
            y (pd.Series, optional): Ignored.

        Returns:
            pd.DataFrame: Feature matrix
        """
        X_ww = infer_feature_types(X)
        X_ww = X_ww.ww.rename({col: str(col) for col in X_ww.columns})
        es = self._make_entity_set(X_ww)
        feature_matrix = calculate_feature_matrix(features=self.features, entityset=es)
        return _retain_custom_types_and_initalize_woodwork(X_ww.ww.logical_types, feature_matrix)
