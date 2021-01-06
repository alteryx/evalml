import string

import featuretools as ft
import nlp_primitives
import pandas as pd

from evalml.pipelines.components.transformers.preprocessing import (
    LSA,
    TextTransformer
)
from evalml.utils.gen_utils import (
    _convert_to_woodwork_structure,
    _convert_woodwork_types_wrapper
)


class TextFeaturizer(TextTransformer):
    """Transformer that can automatically featurize text columns."""
    name = "Text Featurization Component"
    hyperparameter_ranges = {}

    def __init__(self, random_state=0, **kwargs):
        """Extracts features from text columns using featuretools' nlp_primitives

        Arguments:
            random_state (int, np.random.RandomState): Seed for the random number generator.

        """
        self._trans = [nlp_primitives.DiversityScore,
                       nlp_primitives.MeanCharactersPerWord,
                       nlp_primitives.PolarityScore]
        self._features = None
        self._lsa = LSA(random_state=random_state)
        super().__init__(random_state=random_state,
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
        all_text_variable_types = {col_name: 'text' for col_name in X_text.columns}

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
        X = _convert_to_woodwork_structure(X)
        self._text_columns = self._get_text_columns(X)
        if len(self._text_columns) == 0:
            return self

        self._lsa.fit(X)

        X = _convert_woodwork_types_wrapper(X.to_dataframe())
        es = self._make_entity_set(X, self._text_columns)
        self._features = ft.dfs(entityset=es,
                                target_entity='X',
                                trans_primitives=self._trans,
                                features_only=True)
        return self

    def transform(self, X, y=None):
        """Transforms data X by creating new features using existing text columns

        Arguments:
            X (ww.DataTable, pd.DataFrame or np.ndarray): The data to transform.
            y (ww.DataColumn, pd.Series, np.ndarray, optional): Ignored.

        Returns:
            pd.DataFrame: Transformed X
        """
        X = _convert_to_woodwork_structure(X)
        X = _convert_woodwork_types_wrapper(X.to_dataframe())
        if self._features is None or len(self._features) == 0:
            return X

        es = self._make_entity_set(X, self._text_columns)
        X_nlp_primitives = ft.calculate_feature_matrix(features=self._features, entityset=es)
        if X_nlp_primitives.isnull().any().any():
            X_nlp_primitives.fillna(0, inplace=True)

        X_lsa = self._lsa.transform(X[self._text_columns])
        X_nlp_primitives.set_index(X.index, inplace=True)
        return pd.concat([X.drop(self._text_columns, axis=1), X_nlp_primitives, X_lsa], axis=1)
