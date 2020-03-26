import numpy as np
import pandas as pd

from .encoder import CategoricalEncoder


class OneHotEncoder(CategoricalEncoder):

    """One-hot encoder to encode non-numeric data"""
    name = 'One Hot Encoder'
    hyperparameter_ranges = {}
    top_n = 10

    def __init__(self):
        """Initalizes self."""
        parameters = {}
        super().__init__(parameters=parameters,
                         component_obj=None,
                         random_state=0)

    def _get_cat_cols(self, df):
        """Get names of 'object' or 'categorical' columns in the DataFrame."""
        obj_cols = []
        for idx, dt in enumerate(df.dtypes):
            if dt == np.object or pd.api.types.is_categorical_dtype(dt):
                obj_cols.append(df.columns.values[idx])
        return obj_cols

    def fit(self, X, y=None):
        # ensure fit is called first
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_t = X
        self.cols_to_encode = self._get_cat_cols(X_t)
        self.col_values = {}
        for col in X_t.columns:
            if col in self.cols_to_encode:
                v = X_t[col].value_counts(dropna=False).to_frame()
                v.reset_index(inplace=True)
                v = v.sort_values([col, 'index'], ascending=[False, True])
                v.set_index('index', inplace=True)
                unique = v.head(self.top_n).index.tolist()
                self.col_values[col] = list(unique)
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
        X_t = X
        self.cols_to_encode = self._get_cat_cols(X_t)
        encoded_X = pd.DataFrame()
        for col in X_t.columns:
            if col in self.cols_to_encode:
                unique = self.col_values[col]
                for label in unique:
                    new_name = str(col) + "_" + str(label)
                    add = (X[col] == label).astype(int)
                    add = add.rename(new_name)
                    encoded_X = pd.concat([encoded_X, add], axis=1)
            else:
                encoded_X = pd.concat([encoded_X, X_t[col]], axis=1)
        X_t = encoded_X
        return X_t

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)
