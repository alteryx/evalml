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

    def __init__(self, random_state=0, **kwargs):
        """Creates a transformer to perform TF-IDF transformation and Singular Value Decomposition for text columns.

        Arguments:
            random_state (int): Seed for the random number generator. Defaults to 0.
        """
        self._lsa_pipeline = make_pipeline(TfidfVectorizer(), TruncatedSVD(random_state=random_state))
        super().__init__(random_state=random_state,
                         **kwargs)

    def fit(self, X, y=None):
        """Fits component to data by applying the LSA pipeline.

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
        X = _convert_woodwork_types_wrapper(X.to_dataframe())
        corpus = X[self._text_columns].values.flatten()
        # we assume non-str values will have been filtered out prior to calling LSA.fit. this is a safeguard.
        corpus = corpus.astype(str)
        self._lsa_pipeline.fit(corpus)
        return self

    def transform(self, X, y=None):
        """Transforms data X by applying the LSA pipeline.

        Arguments:
            X (ww.DataTable, pd.DataFrame): The data to transform.
            y (ww.DataColumn, pd.Series, optional): Ignored.

        Returns:
            ww.DataTable: Transformed X. The original column is removed and replaced with two columns of the
                          format `LSA(original_column_name)[feature_number]`, where `feature_number` is 0 or 1.
        """
        X = _convert_to_woodwork_structure(X)
<<<<<<< HEAD
        X = _convert_woodwork_types_wrapper(X.to_dataframe())
        if len(self._text_columns) == 0:
=======
        if len(self._all_text_columns) == 0:
>>>>>>> main
            return X

        X = _convert_woodwork_types_wrapper(X.to_dataframe())
        X_t = X.copy()
        for col in self._text_columns:
            transformed = self._lsa_pipeline.transform(X[col])
            X_t['LSA({})[0]'.format(col)] = pd.Series(transformed[:, 0], index=X.index)
            X_t['LSA({})[1]'.format(col)] = pd.Series(transformed[:, 1], index=X.index)
<<<<<<< HEAD

        X_t = X_t.drop(columns=self._text_columns)
        return X_t
=======
        X_t = X_t.drop(columns=text_columns)
        return _convert_to_woodwork_structure(X_t)
>>>>>>> main
