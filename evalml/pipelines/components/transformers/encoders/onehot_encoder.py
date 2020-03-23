import category_encoders as ce
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
        encoder = ce.OneHotEncoder(use_cat_names=True, return_df=True)
        super().__init__(parameters=parameters,
                         component_obj=encoder,
                         random_state=0)

    def _get_cat_cols(self, df):
        """Get names of 'object' or 'categorical' columns in the DataFrame."""
        obj_cols = []
        for idx, dt in enumerate(df.dtypes):
            if dt == np.object or pd.api.types.is_categorical_dtype(dt):
                obj_cols.append(df.columns.values[idx])
        return obj_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """One-hot encode the input DataFrame.

        Arguments:
            X (pd.DataFrame): Dataframe of features.
            y (pd.Series): Ignored.
        Returns:
            Transformed dataframe, where each categorical feature has been encoded into numerical columns using one-hot encoding.

        Ange's own notes:
            1. Check which columns need to be encoded
                - check if dtype is object or category
            2. For each column that needs to be encoded:
                - check how many unique values are in the column
                - if greater than 10, get the top 10 most frequent values
                - otherwise, get all of them :)

                - Encode values for that column and add to the DataFrame
                - Make sure the column before encoding does not stay!
        ## todo: will this affect the original input? should it?

        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if X.isnull().any().any():
            raise ValueError("Dataframe to be encoded can not contain null values.")

        cols_to_encode = self._get_cat_cols(X)
        encoded_X = pd.DataFrame()
        for col in X.columns:
            if col in cols_to_encode:
                v = X[col].value_counts().to_frame()
                v.reset_index(inplace=True)
                v = v.sort_values([col, 'index'], ascending=[False, True])
                v.set_index('index', inplace=True)
                unique = v.head(self.top_n).index.tolist()
                for label in unique:
                    new_name = col + "_" + str(label)
                    add = (X[col] == label).astype(int)
                    add = add.rename(new_name)
                    encoded_X = pd.concat([encoded_X, add], axis=1)
                X.drop(col, axis=1, inplace=True)
            else:
                encoded_X = pd.concat([encoded_X, X[col]], axis=1)
        return encoded_X

    def fit_transform(self, X, y=None):
        return self.transform(X, y)
