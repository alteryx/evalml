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

        parameters = {'text_columns': text_columns}
        text_columns = text_columns or []
        parameters.update(kwargs)

        self._features = None
        self._lsa = LSA(text_columns=text_columns, random_state=random_state)
        self._text_col_names = text_columns
        super().__init__(parameters=parameters,
                         component_obj=None,
                         random_state=random_state)

    def _clean_text(self, X):

        def normalize(text):
            text = text.translate(str.maketrans('', '', string.punctuation))
            return text.lower()

        for text_col in self._text_col_names:
            X[text_col] = X[text_col].apply(normalize)
        return X

    def _verify_col_names(self, col_names):
        missing_cols = [col for col in self._text_col_names if col not in col_names]

        if len(missing_cols) > 0:
            if len(missing_cols) == len(self._text_col_names):
                raise RuntimeError("None of the provided text column names match the columns in the given DataFrame")
            for col in missing_cols:
                self._text_col_names.remove(col)
            logger.warn("Columns {} were not found in the given DataFrame, ignoring".format(missing_cols))

    def _verify_col_types(self, entity_set):
        var_types = entity_set.entities[0].variable_types
        for col in self._text_col_names:
            if var_types[str(col)] is not self._ft.variable_types.variable.Text:
                raise ValueError("Column {} is not a text column, cannot apply TextFeaturizer component".format(col))

    def fit(self, X, y=None):
        if len(self._text_col_names) == 0:
            self._features = []
            return self
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self._verify_col_names(X.columns)
        X_text = X[self._text_col_names]
        X_text['index'] = range(len(X_text))

        es = self._ft.EntitySet()
        es = es.entity_from_dataframe(entity_id='X', dataframe=X_text.rename(columns=str), index='index')
        self._verify_col_types(es)
        es.df = self._clean_text(X)

        trans = [self._nlp_primitives.DiversityScore,
                 self._nlp_primitives.MeanCharactersPerWord,
                 self._nlp_primitives.PartOfSpeechCount,
                 self._nlp_primitives.PolarityScore]

        self._lsa.fit(X)
        self._features = self._ft.dfs(entityset=es,
                                      target_entity='X',
                                      trans_primitives=trans,
                                      features_only=True)
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
        self._verify_col_names(X.columns)

        X_text = X[self._text_col_names]
        X_lsa = self._lsa.transform(X_text)

        X_text['index'] = range(len(X_text))
        X_t = X.drop(self._text_col_names, axis=1)

        es = self._ft.EntitySet()
        es = es.entity_from_dataframe(entity_id='X', dataframe=X_text.rename(columns=str), index='index')
        self._verify_col_types(es)
        es.df = self._clean_text(X)

        feature_matrix = self._ft.calculate_feature_matrix(features=self._features,
                                                           entityset=es,
                                                           verbose=True)
        X_t = pd.concat([X_t, feature_matrix.reindex(X.index), X_lsa], axis=1)
        return X_t
