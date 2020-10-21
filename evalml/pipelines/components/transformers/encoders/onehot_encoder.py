
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder as SKOneHotEncoder

from ..transformer import Transformer

from evalml.pipelines.components import ComponentBaseMeta


class OneHotEncoderMeta(ComponentBaseMeta):
    """A version of the ComponentBaseMeta class which includes validation on an additional one-hot-encoder-specific method `categories`"""
    METHODS_TO_CHECK = ComponentBaseMeta.METHODS_TO_CHECK + ['categories', 'get_feature_names']


class OneHotEncoder(Transformer, metaclass=OneHotEncoderMeta):
    """One-hot encoder to encode non-numeric data."""
    name = 'One Hot Encoder'
    hyperparameter_ranges = {}

    def __init__(self,
                 top_n=10,
                 features_to_encode=None,
                 categories=None,
                 drop=None,
                 handle_unknown="ignore",
                 handle_missing="error",
                 random_state=0,
                 **kwargs):
        """Initalizes an transformer that encodes categorical features in a one-hot numeric array."

        Arguments:
            top_n (int): Number of categories per column to encode. If None, all categories will be encoded.
                Otherwise, the `n` most frequent will be encoded and all others will be dropped. Defaults to 10.
            features_to_encode (list(str)): List of columns to encode. All other columns will remain untouched.
                If None, all appropriate columns will be encoded. Defaults to None.
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
                      "features_to_encode": features_to_encode,
                      "categories": categories,
                      "drop": drop,
                      "handle_unknown": handle_unknown,
                      "handle_missing": handle_missing}
        parameters.update(kwargs)

        # Check correct inputs
        unknown_input_options = ["ignore", "error"]
        missing_input_options = ["as_category", "error"]
        if handle_unknown not in unknown_input_options:
            raise ValueError("Invalid input {} for handle_unknown".format(handle_unknown))
        if handle_missing not in missing_input_options:
            raise ValueError("Invalid input {} for handle_missing".format(handle_missing))
        if top_n is not None and categories is not None:
            raise ValueError("Cannot use categories and top_n arguments simultaneously")

        self.features_to_encode = features_to_encode
        self._encoder = None
        super().__init__(parameters=parameters,
                         component_obj=None,
                         random_state=random_state)
        self._initial_state = self.random_state.get_state()

    @staticmethod
    def _get_cat_cols(X):
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

        if self.features_to_encode is None:
            self.features_to_encode = self._get_cat_cols(X_t)
        invalid_features = [col for col in self.features_to_encode if col not in list(X.columns)]
        if len(invalid_features) > 0:
            raise ValueError("Could not find and encode {} in input data.".format(', '.join(invalid_features)))

        if self.parameters['handle_missing'] == "as_category":
            X_t[self.features_to_encode] = X_t[self.features_to_encode].replace(np.nan, "nan")
        elif self.parameters['handle_missing'] == "error" and X.isnull().any().any():
            raise ValueError("Input contains NaN")

        if len(self.features_to_encode) == 0:
            categories = 'auto'

        elif self.parameters['categories'] is not None:
            categories = self.parameters['categories']
            if len(categories) != len(self.features_to_encode) or not isinstance(categories[0], list):
                raise ValueError('Categories argument must contain a list of categories for each categorical feature')

        else:
            categories = []
            for col in X_t[self.features_to_encode]:
                value_counts = X_t[col].value_counts(dropna=False).to_frame()
                if top_n is None or len(value_counts) <= top_n:
                    unique_values = value_counts.index.tolist()
                else:
                    new_random_state = np.random.RandomState()
                    new_random_state.set_state(self._initial_state)
                    value_counts = value_counts.sample(frac=1, random_state=new_random_state)
                    value_counts = value_counts.sort_values([col], ascending=False, kind='mergesort')
                    unique_values = value_counts.head(top_n).index.tolist()
                unique_values = np.sort(unique_values)
                categories.append(unique_values)

        # Create an encoder to pass off the rest of the computation to
        self._encoder = SKOneHotEncoder(categories=categories,
                                        drop=self.parameters['drop'],
                                        handle_unknown=self.parameters['handle_unknown'])
        self._encoder.fit(X_t[self.features_to_encode])
        return self

    def transform(self, X, y=None):
        """One-hot encode the input DataFrame.

        Arguments:
            X (pd.DataFrame): Dataframe of features.
            y (pd.Series): Ignored.

        Returns:
            Transformed dataframe, where each categorical feature has been encoded into numerical columns using one-hot encoding.
        """

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        cat_cols = self.features_to_encode

        if self.parameters['handle_missing'] == "as_category":
            X[cat_cols] = X[cat_cols].replace(np.nan, "nan")
        if self.parameters['handle_missing'] == "error" and X.isnull().any().any():
            raise ValueError("Input contains NaN")

        X_t = pd.DataFrame()
        # Add the non-categorical columns, untouched
        for col in X.columns:
            if col not in cat_cols:
                X_t = pd.concat([X_t, X[col]], axis=1)
        # The call to pd.concat above changes the type of the index so we will manually keep it the same.
        if not X_t.empty:
            X_t.index = X.index

        # Call sklearn's transform on the categorical columns
        if len(cat_cols) > 0:
            X_cat = pd.DataFrame(self._encoder.transform(X[cat_cols]).toarray(), index=X.index)
            cat_cols_str = [str(c) for c in cat_cols]
            X_cat.columns = self._encoder.get_feature_names(input_features=cat_cols_str)
            X_t = pd.concat([X_t, X_cat], axis=1)

        return X_t

    def categories(self, feature_name):
        """Returns a list of the unique categories to be encoded for the particular feature, in order.

        Arguments:
            feature_name (str): the name of any feature provided to one-hot encoder during fit
        Returns:
            np.array: the unique categories, in the same dtype as they were provided during fit
        """
        try:
            index = self.features_to_encode.index(feature_name)
        except Exception:
            raise ValueError(f'Feature "{feature_name}" was not provided to one-hot encoder as a training feature')
        return self._encoder.categories_[index]

    def get_feature_names(self):
        """Return feature names for the input features after fitting.

        Returns:
            np.array: The feature names after encoding, provided in the same order as input_features.
        """
        return self._encoder.get_feature_names(self.features_to_encode)
