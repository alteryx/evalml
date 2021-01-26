from featuretools import EntitySet, calculate_feature_matrix, dfs

from evalml.pipelines.components.transformers.transformer import Transformer
from evalml.utils.gen_utils import (
    _convert_to_woodwork_structure,
    _convert_woodwork_types_wrapper
)


class DFSTransformer(Transformer):
    """Featuretools DFS component that generates features for ww.DataTables and pd.DataFrames"""
    name = "DFS Transformer"
    hyperparameter_ranges = {}

    def __init__(self, index='index', random_state=0, **kwargs):
        """Allows for featuretools to be used in EvalML

        Arguments:
            index (string): The name of the column that contains the indices. If no column with this name exists,
                then featuretools.EntitySet() creates a column with this name to serve as the index column. Defaults to 'index'
            random_state (int): Seed for the random number generator
        """
        parameters = {"index": index}
        if not isinstance(index, str):
            raise TypeError(f"Index provided must be string, got {type(index)}")

        self.index = index
        self.features = None
        parameters.update(kwargs)
        super().__init__(parameters=parameters,
                         random_state=random_state)

    def _make_entity_set(self, X):
        """helper method that creates and returns the entity set given the datatable X
        """
        ft_es = EntitySet()
        if self.index not in X.columns:
            es = ft_es.entity_from_dataframe(entity_id="X", dataframe=X, index=self.index, make_index=True)
        else:
            es = ft_es.entity_from_dataframe(entity_id="X", dataframe=X, index=self.index)
        return es

    def fit(self, X, y=None):
        """Fits the DFSTransformer Transformer component

        Arguments:
            X (ww.DataTable, pd.DataFrame, np.array): The input data to transform, of shape [n_samples, n_features]
            y (ww.DataColumn, pd.Series, np.ndarray, optional): The target training data of length [n_samples]
        """
        X = _convert_to_woodwork_structure(X)
        X = _convert_woodwork_types_wrapper(X.to_dataframe())
        X.columns = X.columns.astype(str)
        es = self._make_entity_set(X)
        self.features = dfs(entityset=es,
                            target_entity='X',
                            features_only=True)
        return self

    def transform(self, X, y=None):
        """Computes the feature matrix for the input X using featuretools' dfs algorithm.

        Arguments:
            X (ww.DataTable, pd.DataFrame or np.ndarray): The input training data to transform. Has shape [n_samples, n_features]
        """
        X = _convert_to_woodwork_structure(X)
        X = _convert_woodwork_types_wrapper(X.to_dataframe())
        X.columns = X.columns.astype(str)
        es = self._make_entity_set(X)
        feature_matrix = calculate_feature_matrix(features=self.features, entityset=es)
        return feature_matrix
