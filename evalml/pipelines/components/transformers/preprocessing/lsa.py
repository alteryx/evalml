import logging

import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

from evalml.pipelines.components.transformers import Transformer

logger = logging.getLogger()


class LSA(Transformer):
    """Transformer to calculate the Latent Semantic Analysis Values of text input"""
    name = "LSA Transformer"
    hyperparameter_ranges = {}

    def __init__(self, text_columns=None, random_state=0, **kwargs):
        """Creates a transformer to perform TF-IDF transformation and Singular Value Decomposition for text columns.

        Arguments:
            text_columns (list): list of feature names which should be treated as text features.
            random_state (int, np.random.RandomState): Seed for the random number generator.
        """
        parameters = {'text_columns': text_columns}
        parameters.update(kwargs)

        self._all_text_columns = text_columns or []
        self._lsa_pipeline = make_pipeline(TfidfVectorizer(), TruncatedSVD(random_state=random_state))
        super().__init__(parameters=parameters,
                         component_obj=None,
                         random_state=random_state)

    def _get_text_columns(self, X):
        """Returns the ordered list of columns names in the input which have been designated as text columns."""
        columns = []
        missing_columns = []
        for col_name in self._all_text_columns:
            if col_name in X.columns:
                columns.append(col_name)
            else:
                missing_columns.append(col_name)
        if len(columns) == 0:
            raise AttributeError("None of the provided text column names match the columns in the given DataFrame")
        if len(columns) < len(self._all_text_columns):
            logger.warn("Columns {} were not found in the given DataFrame, ignoring".format(missing_columns))
        return columns

    def fit(self, X, y=None):
        if len(self._all_text_columns) == 0:
            return self
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        text_columns = self._get_text_columns(X)
        corpus = X[text_columns].values.flatten()
        # we assume non-str values will have been filtered out prior to calling LSA.fit. this is a safeguard.
        corpus = corpus.astype(str)
        self._lsa_pipeline.fit(corpus)
        return self

    def transform(self, X, y=None):
        """Transforms data X by applying the LSA pipeline.
        Arguments:
            X (pd.DataFrame): Data to transform
            y (pd.Series, optional): Targets
        Returns:
            pd.DataFrame: Transformed X. The original column is removed and replaced with two columns of the
                          format `LSA(original_column_name)[feature_number]`, where `feature_number` is 0 or 1.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if len(self._all_text_columns) == 0:
            return X

        X_t = X.copy()
        text_columns = self._get_text_columns(X)
        for col in text_columns:
            transformed = self._lsa_pipeline.transform(X[col])

            X_t['LSA({})[0]'.format(col)] = pd.Series(transformed[:, 0])
            X_t['LSA({})[1]'.format(col)] = pd.Series(transformed[:, 1])
        X_t = X_t.drop(columns=text_columns)
        return X_t
