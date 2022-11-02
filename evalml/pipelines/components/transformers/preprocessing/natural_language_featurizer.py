"""Transformer that can automatically featurize text columns using featuretools' nlp_primitives."""
import string

import featuretools as ft
from featuretools.primitives import (
    DiversityScore,
    MeanCharactersPerWord,
    NumCharacters,
    NumWords,
    PolarityScore,
)

from evalml.pipelines.components.transformers.preprocessing import LSA, TextTransformer
from evalml.utils import infer_feature_types


class NaturalLanguageFeaturizer(TextTransformer):
    """Transformer that can automatically featurize text columns using featuretools' nlp_primitives.

    Since models cannot handle non-numeric data, any text must be broken down into features that
    provide useful information about that text. This component splits each text column into
    several informative features: Diversity Score, Mean Characters per Word, Polarity Score,
    LSA (Latent Semantic Analysis), Number of Characters, and Number of Words.
    Calling transform on this component will replace any text columns in the given dataset with these numeric columns.

    Args:
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Natural Language Featurizer"
    hyperparameter_ranges = {}
    """{}"""

    def __init__(self, random_seed=0, **kwargs):
        self._trans = [
            NumWords,
            NumCharacters,
            DiversityScore,
            MeanCharactersPerWord,
            PolarityScore,
        ]
        self._features = None
        self._lsa = LSA(random_seed=random_seed)
        self._primitives_provenance = {}
        super().__init__(random_seed=random_seed, **kwargs)

    def _clean_text(self, X):
        """Remove all non-alphanum chars other than spaces, and make lowercase."""

        def normalize(text):
            text = text.translate(str.maketrans("", "", string.punctuation))
            return text.lower()

        for col_name in X.columns:
            # we assume non-str values will have been filtered out prior to calling NaturalLanguageFeaturizer. casting to str is a safeguard.
            X[col_name].fillna("", inplace=True)
            col = X[col_name].astype(str)
            X[col_name] = col.apply(normalize)
        return X

    def _make_entity_set(self, X, text_columns):
        X_text = X[text_columns]
        X_text = self._clean_text(X_text)

        # featuretools expects str-type column names
        X_text.rename(columns=str, inplace=True)
        all_text_logical_types = {
            col_name: "natural_language" for col_name in X_text.columns
        }

        es = ft.EntitySet()
        es.add_dataframe(
            dataframe_name="X",
            dataframe=X_text,
            index="index",
            make_index=True,
            logical_types=all_text_logical_types,
        )
        return es

    def fit(self, X, y=None):
        """Fits component to data.

        Args:
            X (pd.DataFrame or np.ndarray): The input training data of shape [n_samples, n_features]
            y (pd.Series): The target training data of length [n_samples]

        Returns:
            self
        """
        X = infer_feature_types(X)
        self._text_columns = self._get_text_columns(X)
        if len(self._text_columns) == 0:
            return self

        self._lsa.fit(X)

        es = self._make_entity_set(X, self._text_columns)
        self._features = ft.dfs(
            entityset=es,
            target_dataframe_name="X",
            trans_primitives=self._trans,
            max_depth=1,
            features_only=True,
        )
        return self

    @staticmethod
    def _get_primitives_provenance(features):
        provenance = {}
        for feature in features:
            input_col = feature.base_features[0].get_name()
            # Return a copy because `get_feature_names` returns a reference to the names
            output_features = [name for name in feature.get_feature_names()]
            if input_col not in provenance:
                provenance[input_col] = output_features
            else:
                provenance[input_col] += output_features
        return provenance

    def transform(self, X, y=None):
        """Transforms data X by creating new features using existing text columns.

        Args:
            X (pd.DataFrame): The data to transform.
            y (pd.Series, optional): Ignored.

        Returns:
            pd.DataFrame: Transformed X
        """
        X_ww = infer_feature_types(X)
        if self._features is None or len(self._features) == 0:
            return X_ww

        es = self._make_entity_set(X_ww, self._text_columns)
        nan_mask = X[self._text_columns].isna()
        any_nans = nan_mask.any().any()
        X_nlp_primitives = ft.calculate_feature_matrix(
            features=self._features,
            entityset=es,
        )
        if X_nlp_primitives.isnull().any().any():
            X_nlp_primitives.fillna(0, inplace=True)

        X_ww_altered = infer_feature_types(
            X_ww.ww[self._text_columns].fillna(""),
            {s: "NaturalLanguage" for s in self._text_columns},
        )
        X_lsa = self._lsa.transform(X_ww_altered)
        X_nlp_primitives.set_index(X_ww.index, inplace=True)
        if any_nans:
            primitive_features = self._get_primitives_provenance(self._features)
            for column, derived_features in primitive_features.items():
                X_nlp_primitives.loc[nan_mask[column], derived_features] = None

            lsa_features = self._lsa._get_feature_provenance()
            for column, derived_features in lsa_features.items():
                X_lsa.loc[nan_mask[column], derived_features] = None
        X_lsa.ww.init(logical_types={col: "Double" for col in X_lsa.columns})
        X_nlp_primitives.ww.init(
            logical_types={col: "Double" for col in X_nlp_primitives.columns},
        )
        X_ww = X_ww.ww.drop(self._text_columns)
        for col in X_nlp_primitives:
            X_ww.ww[col] = X_nlp_primitives[col]
        for col in X_lsa:
            X_ww.ww[col] = X_lsa[col]
        return X_ww

    def _get_feature_provenance(self):
        if not self._text_columns:
            return {}
        provenance = self._get_primitives_provenance(self._features)
        for col, lsa_features in self._lsa._get_feature_provenance().items():
            if col in provenance:
                provenance[col] += lsa_features
        return provenance
