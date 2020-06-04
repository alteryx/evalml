
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder as SKOneHotEncoder

from .encoder import CategoricalEncoder


class OneHotEncoder(CategoricalEncoder):

    """One-hot encoder to encode non-numeric data"""
    name = 'One Hot Encoder'
    hyperparameter_ranges = {}

    def __init__(self,
                 top_n=10,
                 categories="auto",
                 drop=None,
                 handle_unknown="ignore",
                 handle_missing="ignore",
                 random_state=0):
        """Initalizes an transformer that encodes categorical features in a one-hot numeric array."

        Arguments:
            top_n (int): Number of categories per column to maintain. If None, all categories will be encoded.
                Otherwise, the `n` most frequent will be encoded and all others will be dropped. Defaults to 10.
            categories (list): A list of categories that will be considered for each column, or `"auto"` if `top_n`
                is not None. Defaults to `auto`.
            drop (string): Method ("first" or "if_binary") to use to drop one category per feature. Can also be
                a list specifying which method to use for each feature. Defaults to None.
            handle_unknown (string): Option to "ignore" or "error" for unknown categories for a feature encountered
                during `fit` or `transform`. If `top_n` is used to limit the number of categories, this must be
                "ignore". Defaults to "ignore".
            handle_missing (string): Option to "ignore" or "error" for missing (NaN) values encountered during
                `fit` or `transform`. If this is set to "ignore" and NaN values are within the `n` most frequent,
                "nan" values will be encoded as their own column. Defaults to "ignore".
        """
        parameters = {"top_n": top_n,
                      "categories": categories,
                      "drop": drop,
                      "handle_unknown": handle_unknown,
                      "handle_missing": handle_missing}

        # Check correct inputs
        correct_options = ["ignore", "error"]
        if handle_unknown not in correct_options:
            raise ValueError("{} not a valid option for handle_unknown, options are {}".format(handle_unknown, correct_options))
        if handle_missing not in correct_options:
            raise ValueError("{} not a valid option for handle_missing, options are {}".format(handle_missing, correct_options))

        self.encoder = None
        self.drop = drop
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
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
        X_t = X
        cols_to_encode = self._get_cat_cols(X_t)
        self.col_unique_values = {}

        # Use the categories parameter
        if isinstance(self.parameters['categories'], list):
            if top_n is not None:
                raise ValueError("Cannot use categories and top_n arguments simultaneously")
            categories = self.parameters['categories']
            for col in X_t[cols_to_encode]:
                value_counts = X_t[col].value_counts(dropna=False).to_frame()
                unique_values = value_counts.index.tolist()
                self.col_unique_values[col] = np.sort(unique_values)

        # Use the top_n parameter
        else:
            categories = []
            if self.handle_missing == "ignore":
                X_t[cols_to_encode] = X_t[cols_to_encode].replace(np.nan, "nan")

            # Find the top_n most common categories in each column
            for col in X_t[cols_to_encode]:
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
        self.encoder = encoder.fit(X_t[cols_to_encode])
        return self

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
        cat_cols = self._get_cat_cols(X)

        if self.handle_missing == "ignore":
            X[cat_cols] = X[cat_cols].replace(np.nan, "nan")

        X_t = pd.DataFrame()
        # Add the non-categorical columns, untouched
        for col in X.columns:
            if col not in col_values:
                X_t = pd.concat([X_t, X[col]], axis=1)

        # Call sklearn's transform on the categorical columns
        if len(col_values) != 0:
            X_cat = pd.DataFrame(self.encoder.transform(X[cat_cols]).toarray())
            X_cat.columns = self.encoder.get_feature_names(input_features=cat_cols)
            X_t = pd.concat([X_t.reindex(X_cat.index), X_cat], axis=1)

        return X_t
