
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder as SKOneHotEncoder

from .encoder import CategoricalEncoder


class OneHotEncoder(CategoricalEncoder):

    """One-hot encoder to encode non-numeric data"""
    name = 'One Hot Encoder'
    hyperparameter_ranges = {}

    def __init__(self, top_n=10, categories="auto", drop=None, handle_unknown="ignore", random_state=0):
        """Initalizes self."""
        parameters = {"top_n": top_n,
                      "categories": categories}
        self.encoder = None
        self.drop = drop
        self.handle_unknown = handle_unknown
        super().__init__(parameters=parameters,
                         component_obj=None,
                         random_state=random_state)

    def _get_cat_cols(self, X):
        """Get names of 'object' or 'categorical' columns in the DataFrame."""
        obj_cols = []
        for idx, dtype in enumerate(X.dtypes):
            if dtype == np.object or pd.api.types.is_categorical_dtype(dtype):
                obj_cols.append(X.columns.values[idx])
        return obj_cols

    def fit(self, X, y=None):
        top_n = self.parameters['top_n']
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X_t = X.replace(np.nan, "nan")
        cols_to_encode = self._get_cat_cols(X_t)
        self.col_unique_values = {}
        categories = []

        # Find the top_n most common categories in each column
        for col in X_t.columns:
            if col in cols_to_encode:
                value_counts = X_t[col].value_counts(dropna=False).to_frame()
                if top_n is None or len(value_counts) <= top_n:
                    unique_values = value_counts.index.tolist()
                else:
                    value_counts = value_counts.sample(frac=1, random_state=self.random_state)
                    value_counts = value_counts.sort_values([col], ascending=False, kind='mergesort')
                    unique_values = value_counts.head(top_n).index.tolist()
                unique_values = np.sort(unique_values)
                self.col_unique_values[col] = unique_values
                categories.append(unique_values)

        if len(cols_to_encode) == 0:
            categories = 'auto'

        # Create an encoder to pass off the rest of the computation to
        encoder = SKOneHotEncoder(categories=categories,
                                  drop=self.drop,
                                  handle_unknown=self.handle_unknown)
        # self._component_obj = encoder
        self.encoder = encoder.fit(X_t[cols_to_encode])
        return encoder

    def transform(self, X, y=None):
        """One-hot encode the input DataFrame.

        Arguments:
            X (pd.DataFrame): Dataframe of features.
            y (pd.Series): Ignored.
        Returns:
            Transformed dataframe, where each categorical feature has been encoded into numerical columns using one-hot encoding.
        """
        try:
            col_values = self.col_unique_values
        except AttributeError:
            raise RuntimeError("You must fit one hot encoder before calling transform!")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X = X.replace(np.nan, "nan")

        X_t = pd.DataFrame()
        # Add the non-categorical columns, untouched
        for col in X.columns:
            if col not in col_values:
                X_t = pd.concat([X_t, X[col]], axis=1)

        # Call sklearn's transform on the categorical columns
        if len(col_values) != 0:
            cat_cols = self._get_cat_cols(X)
            X_cat = pd.DataFrame(self.encoder.transform(X[cat_cols]).toarray())
            X_cat.columns = self.encoder.get_feature_names(input_features=cat_cols)
            X_t = pd.concat([X_t.reindex(X_cat.index), X_cat], axis=1)

        return X_t
