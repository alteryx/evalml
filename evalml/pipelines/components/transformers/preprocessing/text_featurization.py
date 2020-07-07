import featuretools as ft
from nlp_primitives import (
    DiversityScore,
    LSA,
    MeanCharactersPerWord,
    PartOfSpeechCount,
    PolarityScore
)
import pandas as pd

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

        self.text_col_names = text_columns
        self._features = None
        super().__init__(parameters=parameters,
                         component_obj=None,
                         random_state=random_state)

    def get_features(self):
        return self._features

    def fit(self, X, y=None):
        if len(self.text_col_names) == 0:
            self._features = []
            return self
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
        """Transforms data X by creating new features using existing DateTime columns, and then dropping those DateTime columns

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

        X_text = X[self.text_col_names]
        X_text['index'] = range(len(X_text))
        X_t = X.drop(self.text_col_names, axis=1)

        es = ft.EntitySet()
        es = es.entity_from_dataframe(entity_id='X', dataframe=X_text, index='index')

        feature_matrix = ft.calculate_feature_matrix(features=self._features,
                                                     entityset=es,
                                                     verbose=True)
        X_t = pd.concat([X_t.reindex(feature_matrix.index), feature_matrix], axis=1)
        return X_t
