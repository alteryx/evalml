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
        """Initalizes an transformer to perform TF-IDF transformation and Singular Value Decomposition.

        Arguments:
            training_corpus(iterable): The collection of documents to fit this component on. Any iterable
            that yields str or unicode objects can be passed in, the simplest format being a 1-dimensional
            list, numpy array, or pandas Series. If no document is passed in, the component will be trained
            on (nltk's brown sentence corpus.)[https://www.nltk.org/book/ch02.html#brown-corpus].
            random_state(int): A seed for the random state.
        """
        text_columns = text_columns or []
        parameters = {'text_columns': text_columns}
        parameters.update(kwargs)

        if len(text_columns) == 0:
            warnings.warn("No text columns were given to LSA, component will have no effect", RuntimeWarning)
        for i, col_name in enumerate(text_columns):
            if not isinstance(col_name, str):
                text_columns[i] = str(col_name)
        self.text_col_names = text_columns
        self.lsa_pipeline = make_pipeline(TfidfVectorizer(), TruncatedSVD(random_state=random_state))
        super().__init__(parameters=parameters,
                         component_obj=None,
                         random_state=random_state)

    def _verify_col_names(self, col_names):
        missing_cols = []
        for col in self.text_col_names:
            if col not in col_names:
                missing_cols.append(col)

        if len(missing_cols) > 0:
            if len(missing_cols) == len(self.text_col_names):
                raise RuntimeError("None of the provided text column names match the columns in the given DataFrame")
            for col in missing_cols:
                self.text_col_names.remove(col)
            warnings.warn("Columns {} were not found in the given DataFrame, ignoring".format(missing_cols), RuntimeWarning)

    def fit(self, X, y=None):
        if len(self.text_col_names) == 0:
            return self
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X).rename(columns=str)
        self._verify_col_names(X.columns)

        corpus = []
        for col in self.text_col_names:
            corpus.extend(X[col].values.tolist())

        self.lsa_pipeline.fit(corpus)
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
                transformed = self.lsa_pipeline.transform(X[col])
                X_t = X_t.drop(labels=col, axis=1)
            except KeyError:
                transformed = self.lsa_pipeline.transform(X[int(col)])
                X_t = X_t.drop(labels=int(col), axis=1)

            X_t['LSA({})[0]'.format(col)] = pd.Series(transformed[:, 0])
            X_t['LSA({})[1]'.format(col)] = pd.Series(transformed[:, 1])
        return X_t
