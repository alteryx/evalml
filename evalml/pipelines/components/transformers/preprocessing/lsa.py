import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

from evalml.pipelines.components.transformers.preprocessing import (
    TextTransformer
)
from evalml.utils.gen_utils import (
    _convert_to_woodwork_structure,
    _convert_woodwork_types_wrapper
)


class LSA(TextTransformer):
    """Transformer to calculate the Latent Semantic Analysis Values of text input"""
    name = "LSA Transformer"
    hyperparameter_ranges = {}

    def __init__(self, text_columns=None, random_state=0, **kwargs):
        """Creates a transformer to perform TF-IDF transformation and Singular Value Decomposition for text columns.

        Arguments:
            text_columns (list): List of feature names which should be treated as text features.
            random_state (int): Seed for the random number generator. Defaults to 0.
        """
        self._lsa_pipeline = make_pipeline(TfidfVectorizer(), TruncatedSVD(random_state=random_state))
        self._provenance = {}
        super().__init__(text_columns=text_columns,
                         random_state=random_state,
                         **kwargs)

    def fit(self, X, y=None):
        if len(self._all_text_columns) == 0:
            return self
        X = _convert_to_woodwork_structure(X)
        X = _convert_woodwork_types_wrapper(X.to_dataframe())
        text_columns = self._get_text_columns(X)
        corpus = X[text_columns].values.flatten()
        # we assume non-str values will have been filtered out prior to calling LSA.fit. this is a safeguard.
        corpus = corpus.astype(str)
        self._lsa_pipeline.fit(corpus)
        return self

    def transform(self, X, y=None):
        """Transforms data X by applying the LSA pipeline.

        Arguments:
            X (ww.DataTable, pd.DataFrame): Data to transform
            y (ww.DataColumn, pd.Series, optional): Ignored.

        Returns:
            ww.DataTable: Transformed X. The original column is removed and replaced with two columns of the
                          format `LSA(original_column_name)[feature_number]`, where `feature_number` is 0 or 1.
        """
        X = _convert_to_woodwork_structure(X)
        if len(self._all_text_columns) == 0:
            return X

        X = _convert_woodwork_types_wrapper(X.to_dataframe())
        X_t = X.copy()
        text_columns = self._get_text_columns(X)
        provenance = {}
        for col in text_columns:
            transformed = self._lsa_pipeline.transform(X[col])
            X_t['LSA({})[0]'.format(col)] = pd.Series(transformed[:, 0], index=X.index)
            X_t['LSA({})[1]'.format(col)] = pd.Series(transformed[:, 1], index=X.index)
            provenance[col] = ['LSA({})[0]'.format(col), 'LSA({})[1]'.format(col)]
        self._provenance = provenance
        X_t = X_t.drop(columns=text_columns)
        return _convert_to_woodwork_structure(X_t)

    def _get_feature_provenance(self):
        return self._provenance
