
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
                 categories=None,
                 drop=None,
                 handle_unknown="ignore",
                 handle_missing="error",
                 random_state=0):
        """Initalizes an transformer that encodes categorical features in a one-hot numeric array."

        Arguments:
            top_n (int): Number of categories per column to encode. If None, all categories will be encoded.
                Otherwise, the `n` most frequent will be encoded and all others will be dropped. Defaults to 10.
            categories (list): A two dimensional list of categories, where `categories[i]` is a list of the categories
                for the column at index `i`. This can also be `None`, or `"auto"` if `top_n` is not None. Defaults to None.
            drop (string): Method ("first" or "if_binary") to use to drop one category per feature. Can also be
                a list specifying which method to use for each feature. Defaults to None.
            handle_unknown (string): Whether to ignore or error for unknown categories for a feature encountered
                during `fit` or `transform`. If either `top_n` or `categories` is used to limit the number of categories
                per column, this must be "ignore". Defaults to "ignore".
            handle_missing (string): Options for how to handle missing (NaN) values encountered during
                `fit` or `transform`. If this is set to "as_category" and NaN values are within the `n` most frequent,
                "nan" values will be encoded as their own column. If this is set to "error", any missing
                values encountered will raise an error. Defaults to "error".
        """
        parameters = {"top_n": top_n,
                      "categories": categories,
                      "drop": drop,
                      "handle_unknown": handle_unknown,
                      "handle_missing": handle_missing}

        # Check correct inputs
        unknown_input_options = ["ignore", "error"]
        missing_input_options = ["as_category", "error"]
        if handle_unknown not in unknown_input_options:
            raise ValueError("Invalid input {} for handle_unknown".format(handle_unknown))
        if handle_missing not in missing_input_options:
            raise ValueError("Invalid input {} for handle_missing".format(handle_missing))
        if top_n is not None and categories is not None:
            raise ValueError("Cannot use categories and top_n arguments simultaneously")

        self._encoder = None
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

        if self.parameters['handle_missing'] == "as_category":
            X_t[cols_to_encode] = X_t[cols_to_encode].replace(np.nan, "nan")
        elif self.parameters['handle_missing'] == "error" and X.isnull().any().any():
            raise ValueError("Input contains NaN")

        if len(cols_to_encode) == 0:
            categories = 'auto'

        elif self.parameters['categories'] is not None:
            categories = self.parameters['categories']
            if len(categories) != len(cols_to_encode) or not isinstance(categories[0], list):
                raise ValueError('Categories argument must contain a list of categories for each categorical feature')

        else:
            categories = []
            for col in X_t[cols_to_encode]:
                value_counts = X_t[col].value_counts(dropna=False).to_frame()
                if top_n is None or len(value_counts) <= top_n:
                    unique_values = value_counts.index.tolist()
                else:
                    value_counts = value_counts.sample(frac=1, random_state=self.random_state)
                    value_counts = value_counts.sort_values([col], ascending=False, kind='mergesort')
                    unique_values = value_counts.head(top_n).index.tolist()
                unique_values = np.sort(unique_values)
                categories.append(unique_values)

        # Create an encoder to pass off the rest of the computation to
        self._encoder = SKOneHotEncoder(categories=categories,
                                        drop=self.parameters['drop'],
                                        handle_unknown=self.parameters['handle_unknown'])
        self._encoder.fit(X_t[cols_to_encode])
        return self

    def transform(self, X, y=None):
        """One-hot encode the input DataFrame.

        Arguments:
            X (pd.DataFrame): Dataframe of features.
            y (pd.Series): Ignored.
        Returns:
            Transformed dataframe, where each categorical feature has been encoded into numerical columns using one-hot encoding.
        """
        if self._encoder is None:
            raise RuntimeError("You must fit one hot encoder before calling transform!")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        cat_cols = self._get_cat_cols(X)

        if self.parameters['handle_missing'] == "as_category":
            X[cat_cols] = X[cat_cols].replace(np.nan, "nan")
        if self.parameters['handle_missing'] == "error" and X.isnull().any().any():
            raise ValueError("Input contains NaN")

        X_t = pd.DataFrame()
        # Add the non-categorical columns, untouched
        for col in X.columns:
            if col not in cat_cols:
                X_t = pd.concat([X_t, X[col]], axis=1)

        # Call sklearn's transform on the categorical columns
        if len(cat_cols) > 0:
            X_cat = pd.DataFrame(self._encoder.transform(X[cat_cols]).toarray())
            X_cat.columns = self._encoder.get_feature_names(input_features=cat_cols)
            X_t = pd.concat([X_t.reindex(X_cat.index), X_cat], axis=1)

        return X_t
