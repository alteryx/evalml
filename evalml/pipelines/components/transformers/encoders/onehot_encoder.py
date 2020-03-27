
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

    def _get_cat_cols(self, X):
        """Get names of 'object' or 'categorical' columns in the DataFrame."""
        obj_cols = []
        for idx, dtype in enumerate(X.dtypes):
            if dtype == np.object or pd.api.types.is_categorical_dtype(dtype):
                obj_cols.append(X.columns.values[idx])
        return obj_cols

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X_t = X
        self.cols_to_encode = self._get_cat_cols(X_t)
        self.col_unique_values = {}
        for col in X_t.columns:
            if col in self.cols_to_encode:
                v = X_t[col].value_counts(dropna=False).to_frame()
                unique = []
                if len(v) <= self.top_n:
                    unique = v.index.tolist()
                else:
                    # v.reset_index(inplace=True)
                    v = v.sort_values([col], ascending=False)
                    # v.set_index('index', inplace=True)

                    last_row_val = v[col].iloc[-1]
                    # grab all that are less than that value
                    v_temp = v.loc[v[col] < last_row_val]
                    candidates = v.loc[v[col] == last_row_val]
                    num_to_sample = self.top_n - len(v_temp)
                    random_subset = candidates.sample(n=num_to_sample, random_state=self.random_state)
                    v_temp = v_temp.append(random_subset)
                    unique = v_temp.index.tolist()
                self.col_unique_values[col] = unique
        return self

    def transform(self, X, y=None):
        """One-hot encode the input DataFrame.

        Arguments:
            X (pd.DataFrame): Dataframe of features.
            y (pd.Series): Ignored.
        Returns:
            Transformed dataframe, where each categorical feature has been encoded into numerical columns using one-hot encoding.
        """
        col_values = None
        try:
            col_values = self.col_unique_values
        except AttributeError:
            raise RuntimeError("You must fit one hot encoder before calling transform!")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.cols_to_encode = self._get_cat_cols(X)
        X_t = pd.DataFrame()
        for col in X.columns:
            if col in self.cols_to_encode:
                unique = col_values[col]
                for label in unique:
                    new_name = str(col) + "_" + str(label)
                    add = (X[col] == label).astype(int)
                    add = add.rename(new_name)
                    X_t = pd.concat([X_t, add], axis=1)
            else:
                X_t = pd.concat([X_t, X[col]], axis=1)
        return X_t

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)
