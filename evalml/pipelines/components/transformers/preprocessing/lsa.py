"""Transformer to calculate the Latent Semantic Analysis Values of text input."""
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

from evalml.pipelines.components.transformers.preprocessing import TextTransformer
from evalml.utils import infer_feature_types


class LSA(TextTransformer):
    """Transformer to calculate the Latent Semantic Analysis Values of text input.

    Args:
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "LSA Transformer"
    hyperparameter_ranges = {}
    """{}"""

    def __init__(self, random_seed=0, **kwargs):
        self._lsa_pipeline = make_pipeline(
            TfidfVectorizer(),
            TruncatedSVD(random_state=random_seed),
        )
        self._provenance = {}
        super().__init__(random_seed=random_seed, **kwargs)

    def fit(self, X, y=None):
        """Fits the input data.

        Args:
            X (pd.DataFrame): The data to transform.
            y (pd.Series, optional): Ignored.

        Returns:
            self
        """
        X = infer_feature_types(X)
        self._text_columns = self._get_text_columns(X)

        if len(self._text_columns) == 0:
            return self
        corpus = X[self._text_columns].values.flatten()
        # we assume non-str values will have been filtered out prior to calling LSA.fit. this is a safeguard.
        corpus = corpus.astype(str)
        self._lsa_pipeline.fit(corpus)
        return self

    def transform(self, X, y=None):
        """Transforms data X by applying the LSA pipeline.

        Args:
            X (pd.DataFrame): The data to transform.
            y (pd.Series, optional): Ignored.

        Returns:
            pd.DataFrame: Transformed X. The original column is removed and replaced with two columns of the
                          format `LSA(original_column_name)[feature_number]`, where `feature_number` is 0 or 1.
        """
        X_ww = infer_feature_types(X)
        if len(self._text_columns) == 0:
            return X_ww

        provenance = {}
        for col in self._text_columns:
            transformed = self._lsa_pipeline.transform(X_ww[col])
            X_ww.ww["LSA({})[0]".format(col)] = pd.Series(
                transformed[:, 0],
                index=X_ww.index,
            )
            X_ww.ww["LSA({})[1]".format(col)] = pd.Series(
                transformed[:, 1],
                index=X_ww.index,
            )
            provenance[col] = ["LSA({})[0]".format(col), "LSA({})[1]".format(col)]
        self._provenance = provenance

        X_t = X_ww.ww.drop(columns=self._text_columns)
        return X_t

    def _get_feature_provenance(self):
        return self._provenance
