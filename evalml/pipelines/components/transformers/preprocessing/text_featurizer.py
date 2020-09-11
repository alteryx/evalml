import string

import pandas as pd

from evalml.pipelines.components.transformers.preprocessing import (
    LSA,
    TextTransformer
)
from evalml.utils import import_or_raise


class TextFeaturizer(TextTransformer):
    """Transformer that can automatically featurize text columns."""
    name = "Text Featurization Component"
    hyperparameter_ranges = {}

    def __init__(self, text_columns=None, random_state=0, **kwargs):
        """Extracts features from text columns using featuretools' nlp_primitives

        Arguments:
            text_columns (list): list of feature names which should be treated as text features.
            random_state (int, np.random.RandomState): Seed for the random number generator.

        """
        self._ft = import_or_raise("featuretools", error_msg="Package featuretools is not installed. Please install using `pip install featuretools[nlp_primitives].`")
        self._nlp_primitives = import_or_raise("nlp_primitives", error_msg="Package nlp_primitives is not installed. Please install using `pip install featuretools[nlp_primitives].`")
        self._trans = [self._nlp_primitives.DiversityScore,
                       self._nlp_primitives.MeanCharactersPerWord,
                       self._nlp_primitives.PolarityScore]
        self._features = None
        self._lsa = LSA(text_columns=text_columns, random_state=random_state)
        super().__init__(text_columns=text_columns,
                         random_state=random_state,
                         **kwargs)

    def _clean_text(self, X):
        """Remove all non-alphanum chars other than spaces, and make lowercase"""

        def normalize(text):
            text = text.translate(str.maketrans('', '', string.punctuation))
            return text  # .lower()

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

        es = self._ft.EntitySet()
        es.entity_from_dataframe(entity_id='X', dataframe=X_text, index='index', make_index=True,
                                 variable_types=all_text_variable_types)

        variable_types = es.entities[0].variable_types
        for col in text_columns:
            if variable_types[str(col)] is not self._ft.variable_types.variable.Text:
                raise ValueError("Column '{}' is not a text column, cannot apply TextFeaturizer component".format(col))
        return es

    def fit(self, X, y=None):
        """Fits component to data

        Arguments:
            X (pd.DataFrame or np.array): the input training data of shape [n_samples, n_features]
            y (pd.Series, optional): the target training labels of length [n_samples]

        Returns:
            self
        """
        if len(self._all_text_columns) == 0:
            return self
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        text_columns = self._get_text_columns(X)
        es = self._make_entity_set(X, text_columns)
        self._features = self._ft.dfs(entityset=es,
                                      target_entity='X',
                                      trans_primitives=self._trans,
                                      features_only=True)
        self._lsa.fit(X)
        return self

    def transform(self, X, y=None):
        """Transforms data X by creating new features using existing text columns

        Arguments:
            X (pd.DataFrame): Data to transform
            y (pd.Series, optional): Input Labels
        Returns:
            pd.DataFrame: Transformed X
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if self._features is None or len(self._features) == 0:
            return X

        text_columns = self._get_text_columns(X)
        es = self._make_entity_set(X, text_columns)
        X_nlp_primitives = self._ft.calculate_feature_matrix(features=self._features, entityset=es)
        if X_nlp_primitives.isnull().any().any():
            X_nlp_primitives.fillna(0, inplace=True)

        X_lsa = self._lsa.transform(X[text_columns])

        return pd.concat([X.drop(text_columns, axis=1), X_nlp_primitives, X_lsa], axis=1)
