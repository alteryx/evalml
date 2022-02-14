"""Featuretools DFS component that generates features for the input features."""
from featuretools import EntitySet, calculate_feature_matrix, dfs
from featuretools.feature_base import IdentityFeature

from evalml.pipelines.components.transformers.transformer import Transformer
from evalml.utils import infer_feature_types


class DFSTransformer(Transformer):
    """Featuretools DFS component that generates features for the input features.

    Args:
        index (string): The name of the column that contains the indices. If no column with this name exists,
            then featuretools.EntitySet() creates a column with this name to serve as the index column. Defaults to 'index'.
        random_seed (int): Seed for the random number generator. Defaults to 0.
        features (list)[FeatureBase]: List of features to run DFS on. Defaults to None. Features will only be computed if the columns used by the feature exist in the input and if the feature itself is not in input.
    """

    name = "DFS Transformer"
    hyperparameter_ranges = {}
    """{}"""

    def __init__(self, index="index", features=None, random_seed=0, **kwargs):
        parameters = {"index": index}
        if not isinstance(index, str):
            raise TypeError(f"Index provided must be string, got {type(index)}")

        self.index = index
        self.features = features
        self._passed_in_features = True if features else None
        parameters.update(kwargs)
        super().__init__(parameters=parameters, random_seed=random_seed)

    def _make_entity_set(self, X):
        """Helper method that creates and returns the entity set given the input data."""
        ft_es = EntitySet()
        # TODO: This delete was introduced for compatibility with Featuretools 1.0.0.  This should
        # be removed after Featuretools handles unnamed dataframes being passed to this function.
        del X.ww
        if self.index not in X.columns:
            es = ft_es.add_dataframe(
                dataframe=X, dataframe_name="X", index=self.index, make_index=True
            )
        else:
            es = ft_es.add_dataframe(dataframe=X, dataframe_name="X", index=self.index)
        return es

    def _filter_features(self, X):
        features_to_use = []
        X_columns_set = set(X.columns)
        for feature in self.features:
            input_cols = [f.column_name for f in feature.base_features]

            # If feature is an identity feature and the column doesn't exist, skip feature
            if (
                isinstance(feature, IdentityFeature)
                and feature.column_name not in X.columns
            ):
                continue

            # If feature's required columns doesn't exist, skip feature
            if not set(input_cols).issubset(X_columns_set):
                continue

            # If feature's transformed columns already exist, skip feature
            if not isinstance(feature, IdentityFeature) and set(
                feature.get_feature_names()
            ).intersection(X_columns_set):
                continue

            features_to_use.append(feature)
        return features_to_use

    def fit(self, X, y=None):
        """Fits the DFSTransformer Transformer component.

        Args:
            X (pd.DataFrame, np.array): The input data to transform, of shape [n_samples, n_features].
            y (pd.Series): The target training data of length [n_samples].

        Returns:
            self
        """
        if not self._passed_in_features:
            X_ww = infer_feature_types(X)
            X_ww = X_ww.ww.rename({col: str(col) for col in X_ww.columns})
            es = self._make_entity_set(X_ww)
            self.features = dfs(
                entityset=es, target_dataframe_name="X", features_only=True, max_depth=1
            )
        return self

    def transform(self, X, y=None):
        """Computes the feature matrix for the input X using featuretools' dfs algorithm.

        Args:
            X (pd.DataFrame or np.ndarray): The input training data to transform. Has shape [n_samples, n_features]
            y (pd.Series, optional): Ignored.

        Returns:
            pd.DataFrame: Feature matrix
        """
        X_ww = infer_feature_types(X)
        X_ww = X_ww.ww.rename({col: str(col) for col in X_ww.columns})

        features_to_use = (
            self._filter_features(X) if self._passed_in_features else self.features
        )
        all_identity = all([isinstance(f, IdentityFeature) for f in features_to_use])
        if not features_to_use or (all_identity and self._passed_in_features):
            return X_ww
        es = self._make_entity_set(X_ww)
        feature_matrix = calculate_feature_matrix(
            features=features_to_use, entityset=es
        )
        typed_columns = set(X_ww.columns).intersection(set(feature_matrix.columns))
        feature_matrix.ww.init(schema=X_ww.ww.schema.get_subset_schema(typed_columns))
        return feature_matrix
