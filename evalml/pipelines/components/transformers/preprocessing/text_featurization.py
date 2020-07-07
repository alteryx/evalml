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

        for col_name in text_columns:
            if not isinstance(col_name, str):
                raise ValueError("Column names must be of object type")
        self.text_col_names = text_columns
        self._features = None
        super().__init__(parameters=parameters,
                         component_obj=None,
                         random_state=random_state)

    @property
    def features(self):
        return self._features

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
            self._features = []
            return self
        self._verify_col_names(X.columns)
        X_text = X[self.text_col_names]
        X_text['index'] = range(len(X_text))

        es = ft.EntitySet()
        es = es.entity_from_dataframe(entity_id='X', dataframe=X_text, index='index')

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
        if len(self._features) == 0:
            return X
        self._verify_col_names(X.columns)

        X_text = X[self.text_col_names]
        X_text['index'] = range(len(X_text))
        X_t = X.drop(self.text_col_names, axis=1)

        es = ft.EntitySet()
        es = es.entity_from_dataframe(entity_id='X', dataframe=X_text, index='index')

        feature_matrix = ft.calculate_feature_matrix(features=self._features,
                                                     entityset=es,
                                                     verbose=True)
        X_t = pd.concat([X_t, feature_matrix.reindex(X.index)], axis=1)
        return X_t
