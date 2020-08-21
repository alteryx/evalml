import logging
import string

import pandas as pd

from evalml.pipelines.components.transformers import Transformer
from evalml.pipelines.components.transformers.preprocessing import LSA
from evalml.utils import import_or_raise

logger = logging.getLogger()


class TextFeaturizer(Transformer):
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
                       self._nlp_primitives.PartOfSpeechCount,
                       self._nlp_primitives.PolarityScore]

        parameters = {'text_columns': text_columns}
        parameters.update(kwargs)

        self._features = None
        text_columns = text_columns or []
        self._lsa = LSA(text_columns=text_columns, random_state=random_state)
        self._text_col_names = text_columns
        super().__init__(parameters=parameters,
                         component_obj=None,
                         random_state=random_state)

    def _clean_text(self, X):
        """Remove all non-alphanum chars other than spaces, and make lowercase"""

        def normalize(text):
            text = text.translate(str.maketrans('', '', string.punctuation))
            return text.lower()

        for text_col in self._text_col_names:
            # we assume non-str values will have been filtered out prior to calling TextFeaturizer. casting to str is a safeguard.
            col = X[text_col].astype(str)
            X[text_col] = col.apply(normalize)
        return X

    def _verify_col_names(self, col_names):
        missing_cols = [col for col in self._text_col_names if col not in col_names]

        if len(missing_cols) > 0:
            if len(missing_cols) == len(self._text_col_names):
                raise RuntimeError("None of the provided text column names match the columns in the given DataFrame")
            for col in missing_cols:
                self._text_col_names.remove(col)
            logger.warn("Columns {} were not found in the given DataFrame, ignoring".format(missing_cols))

    def _make_entity_set(self, X):
        self._verify_col_names(X.columns)
        X_text = X[self._text_col_names]
        X_text = self._clean_text(X_text)
        X_text.rename(columns=str, inplace=True)

        all_text_variable_types = {col_name: 'text' for col_name in self._text_col_names}
        es = self._ft.EntitySet()
        es.entity_from_dataframe(entity_id='X', dataframe=X_text, index='index', make_index=True,
                                 variable_types=all_text_variable_types)

        variable_types = es.entities[0].variable_types
        for col in self._text_col_names:
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
        if len(self._text_col_names) == 0:
            return self
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        es = self._make_entity_set(X)
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

        es = self._make_entity_set(X)
        X_nlp_primitives = self._ft.calculate_feature_matrix(features=self._features, entityset=es)
        if X_nlp_primitives.isnull().any().any():
            X_nlp_primitives.fillna(0, inplace=True)
        X_nlp_primitives.reindex(X.index)

        X_lsa = self._lsa.transform(X[self._text_col_names])

        return pd.concat([X.drop(self._text_col_names, axis=1), X_nlp_primitives, X_lsa], axis=1)
