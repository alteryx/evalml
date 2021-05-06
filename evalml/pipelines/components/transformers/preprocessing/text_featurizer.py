import string

import featuretools as ft
import nlp_primitives
import pandas as pd

from evalml.pipelines.components.transformers.preprocessing import (
    LSA,
    TextTransformer
)
from evalml.utils import (
    _convert_woodwork_types_wrapper,
    _retain_custom_types_and_initalize_woodwork,
    infer_feature_types
)


class TextFeaturizer(TextTransformer):
    """Transformer that can automatically featurize text columns."""
    name = "Text Featurization Component"
    hyperparameter_ranges = {}

    def __init__(self, random_seed=0, **kwargs):
        """Extracts features from text columns using featuretools' nlp_primitives

        Arguments:
            random_seed (int): Seed for the random number generator. Defaults to 0.
        """
        self._trans = [nlp_primitives.DiversityScore,
                       nlp_primitives.MeanCharactersPerWord,
                       nlp_primitives.PolarityScore]
        self._features = None
        self._lsa = LSA(random_seed=random_seed)
        self._primitives_provenance = {}
        super().__init__(random_seed=random_seed,
                         **kwargs)

    def _clean_text(self, X):
        """Remove all non-alphanum chars other than spaces, and make lowercase"""

        def normalize(text):
            text = text.translate(str.maketrans('', '', string.punctuation))
            return text.lower()

        for col_name in X.columns:
            # we assume non-str values will have been filtered out prior to calling TextFeaturizer. casting to str is a safeguard.
            col = X[col_name].astype(str)
            X[col_name] = col.apply(normalize)
        return X

    def _make_entity_set(self, X, text_columns):
        X_text = X[text_columns]
        X_text = self._clean_text(X_text)

        # featuretools expects str-type column names
        X_text.rename(columns=str, inplace=True)
        all_text_variable_types = {col_name: 'natural_language' for col_name in X_text.columns}

        es = ft.EntitySet()
        es.entity_from_dataframe(entity_id='X', dataframe=X_text, index='index', make_index=True,
                                 variable_types=all_text_variable_types)
        return es

    def fit(self, X, y=None):
        """Fits component to data

        Arguments:
            X (ww.DataTable, pd.DataFrame or np.ndarray): The input training data of shape [n_samples, n_features]
            y (ww.DataColumn, pd.Series, np.ndarray, optional): The target training data of length [n_samples]

        Returns:
            self
        """
        X = infer_feature_types(X)
        self._text_columns = self._get_text_columns(X)
        if len(self._text_columns) == 0:
            return self

        self._lsa.fit(X)

        X = _convert_woodwork_types_wrapper(X.to_dataframe())
        es = self._make_entity_set(X, self._text_columns)
        self._features = ft.dfs(entityset=es,
                                target_entity='X',
                                trans_primitives=self._trans,
                                max_depth=1,
                                features_only=True)
        return self

    @staticmethod
    def _get_primitives_provenance(features):
        provenance = {}
        for feature in features:
            input_col = feature.base_features[0].get_name()
            # Return a copy because `get_feature_names` returns a reference to the names
            output_features = [name for name in feature.get_feature_names()]
            if input_col not in provenance:
                provenance[input_col] = output_features
            else:
                provenance[input_col] += output_features
        return provenance

    def transform(self, X, y=None):
        """Transforms data X by creating new features using existing text columns

        Arguments:
            X (ww.DataTable, pd.DataFrame): The data to transform.
            y (ww.DataColumn, pd.Series, optional): Ignored.

        Returns:
            ww.DataTable: Transformed X
        """
        X_ww = infer_feature_types(X)
        if self._features is None or len(self._features) == 0:
            return X_ww
        X = _convert_woodwork_types_wrapper(X_ww.to_dataframe())
        es = self._make_entity_set(X, self._text_columns)
        X_nlp_primitives = ft.calculate_feature_matrix(features=self._features, entityset=es)
        if X_nlp_primitives.isnull().any().any():
            X_nlp_primitives.fillna(0, inplace=True)

        X_lsa = self._lsa.transform(X[self._text_columns]).to_dataframe()
        X_nlp_primitives.set_index(X.index, inplace=True)
        X_t = pd.concat([X.drop(self._text_columns, axis=1), X_nlp_primitives, X_lsa], axis=1)
        return _retain_custom_types_and_initalize_woodwork(X_ww, X_t)

    def _get_feature_provenance(self):
        if not self._text_columns:
            return {}
        provenance = self._get_primitives_provenance(self._features)
        for col, lsa_features in self._lsa._get_feature_provenance().items():
            if col in provenance:
                provenance[col] += lsa_features
        return provenance
