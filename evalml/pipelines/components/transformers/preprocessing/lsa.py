import warnings

import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

from evalml.pipelines.components.transformers import Transformer


class LSA(Transformer):
    """Transformer to calculate the Latent Semantic Analysis Values of text input"""
    name = "LSA Transformer"
    hyperparameter_ranges = {}

    def __init__(self, text_columns=None, random_state=0, **kwargs):
        """Creates a transformer to perform TF-IDF transformation and Singular Value Decomposition for text columns.

        Arguments:
            text_colums (list): list of `pd.DataFrame` column names that contain text.
            random_state (int, np.random.RandomState): Seed for the random number generator.
        """
        parameters = {'text_columns': text_columns}
        text_columns = text_columns or []
        parameters.update(kwargs)

        self.text_col_names = [str(col_name) for col_name in text_columns]
        self._lsa_pipeline = make_pipeline(TfidfVectorizer(), TruncatedSVD(random_state=random_state))
        super().__init__(parameters=parameters,
                         component_obj=None,
                         random_state=random_state)

    def _verify_col_names(self, col_names):
        missing_cols = [col for col in self.text_col_names if col not in col_names]

        if len(missing_cols) > 0:
            if len(missing_cols) == len(self.text_col_names):
                raise RuntimeError("None of the provided text column names match the columns in the given DataFrame")
            for col in missing_cols:
                self.text_col_names.remove(col)
            warnings.warn("Columns {} were not found in the given DataFrame, ignoring".format(missing_cols), RuntimeWarning)

    def fit(self, X, y=None):
        if len(self.text_col_names) == 0:
            warnings.warn("No text columns were given to LSA, component has no effect", RuntimeWarning)
            return self
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X).rename(columns=str)
        self._verify_col_names(X.columns)

        corpus = []
        for col in self.text_col_names:
            corpus.extend(X[col].values.tolist())

        self._lsa_pipeline.fit(corpus)
        return self

    def transform(self, X, y=None):
        """Transforms data X by applying the LSA pipeline.
        Arguments:
            X (pd.DataFrame): Data to transform
            y (pd.Series, optional): Targets
        Returns:
            pd.DataFrame: Transformed X
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X_t = X

        for col in self.text_col_names:
            try:
                transformed = self._lsa_pipeline.transform(X[col])
                X_t = X_t.drop(labels=col, axis=1)
            except KeyError:
                transformed = self._lsa_pipeline.transform(X[int(col)])
                X_t = X_t.drop(labels=int(col), axis=1)

            X_t['LSA({})[0]'.format(col)] = pd.Series(transformed[:, 0])
            X_t['LSA({})[1]'.format(col)] = pd.Series(transformed[:, 1])
        return X_t
