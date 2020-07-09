import string
import warnings

import featuretools as ft
import pandas as pd
from nlp_primitives import (
    LSA,
    DiversityScore,
    MeanCharactersPerWord,
    PartOfSpeechCount,
    PolarityScore
)

from evalml.pipelines.components.transformers import Transformer


class TextFeaturization(Transformer):
    """Transformer that can automatically featurize text columns."""
    name = "Text Featurization Component"
    hyperparameter_ranges = {}

    def __init__(self, text_columns=[], random_state=0, **kwargs):
        """Extracts features from text columns using featuretools' nlp_primitives

        Arguments:
            text_colums (list): list of `pd.DataFrame` column names that contain text.
            random_state (int, np.random.RandomState): Seed for the random number generator.

        """
        parameters = {}
        parameters.update(kwargs)

        if len(text_columns) == 0:
            warnings.warn("No text columns were given to TextFeaturization, component will have no effect", RuntimeWarning)
        for i, col_name in enumerate(text_columns):
            if not isinstance(col_name, str):
                text_columns[i] = str(col_name)
        self.text_col_names = text_columns
        self._features = None
        super().__init__(parameters=parameters,
                         component_obj=None,
                         random_state=random_state)

    @property
    def features(self):
        return self._features

    def _clean_text(self, X):

        def normalize(text):
            text = text.translate(str.maketrans('', '', string.punctuation))
            return text.lower()

        for text_col in self.text_col_names:
            X[text_col] = X[text_col].apply(normalize)
        return X

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

    def _verify_col_types(self, entity_set):
        var_types = entity_set.entities[0].variable_types
        for col in self.text_col_names:
            if var_types[col] is not ft.variable_types.variable.Text:
                raise ValueError("Column {} is not a text column, cannot apply TextFeaturization component".format(col))

    def fit(self, X, y=None):
        if len(self.text_col_names) == 0:
            self._features = []
            return self
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X).rename(columns=str)
        self._verify_col_names(X.columns)
        X_text = X[self.text_col_names]
        X_text['index'] = range(len(X_text))

        es = ft.EntitySet()
        es = es.entity_from_dataframe(entity_id='X', dataframe=X_text, index='index')
        self._verify_col_types(es)
        es.df = self._clean_text(X)

        trans = [DiversityScore,
                 LSA,
                 MeanCharactersPerWord,
                 PartOfSpeechCount,
                 PolarityScore]

        self._features = ft.dfs(entityset=es,
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
        if self._features is None:
            raise RuntimeError(f"You must fit {self.name} before calling transform!")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if len(self._features) == 0:
            return X
        X = X.rename(columns=str)
        self._verify_col_names(X.columns)

        X_text = X[self.text_col_names]
        X_text['index'] = range(len(X_text))
        X_t = X.drop(self.text_col_names, axis=1)

        es = ft.EntitySet()
        es = es.entity_from_dataframe(entity_id='X', dataframe=X_text, index='index')
        self._verify_col_types(es)
        es.df = self._clean_text(X)

        feature_matrix = ft.calculate_feature_matrix(features=self._features,
                                                     entityset=es,
                                                     verbose=True)
        X_t = pd.concat([X_t, feature_matrix.reindex(X.index)], axis=1)
        return X_t
