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
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_t = X
        self.cols_to_encode = self._get_cat_cols(X_t)
        self.cols_to_drop = []
        encoded_X = pd.DataFrame()
        for col in X_t.columns:
            if col in self.cols_to_encode:
                # if X_t[col].isnull().any():
                #     X_t[col].fillna(-1, inplace=True)

                v = X_t[col].value_counts().to_frame()
                v.reset_index(inplace=True)
                v = v.sort_values([col, 'index'], ascending=[False, True])
                v.set_index('index', inplace=True)
                unique = v.head(self.top_n).index.tolist()
                for label in unique:
                    new_name = str(col) + "_" + str(label)
                    add = (X[col] == label).astype(int)
                    add = add.rename(new_name)
                    encoded_X = pd.concat([encoded_X, add], axis=1)
                    # self.encoded_cols = pd.concat([self.encoded_cols, add])


                self.cols_to_drop.append(col)
            # else:
            #     encoded_X = pd.concat([encoded_X, X_t[col]], axis=1)
        print (encoded_X.columns)
        self.encoded_cols = encoded_X
        return self


    def transform(self, X, y=None):
        """One-hot encode the input DataFrame.

        Arguments:
            X (pd.DataFrame): Dataframe of features.
            y (pd.Series): Ignored.
        Returns:
            Transformed dataframe, where each categorical feature has been encoded into numerical columns using one-hot encoding.
        """
        # import pdb; pdb.set_trace()
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # if X.isnull().any().any():
        #     raise ValueError("Dataframe to be encoded can not contain null values.")
        X_t = X
        # import pdb; pdb.set_trace()
        X_t = pd.concat([X_t, self.encoded_cols], axis=1)
        X_t = X_t.drop(self.cols_to_drop, axis=1)
        # import pdb; pdb.set_trace()
        return X_t
    # def transform(self, X, y=None):
    #     """One-hot encode the input DataFrame.

    #     Arguments:
    #         X (pd.DataFrame): Dataframe of features.
    #         y (pd.Series): Ignored.
    #     Returns:
    #         Transformed dataframe, where each categorical feature has been encoded into numerical columns using one-hot encoding.

    #     Ange's own notes:
    #         1. Check which columns need to be encoded
    #             - check if dtype is object or category
    #         2. For each column that needs to be encoded:
    #             - check how many unique values are in the column
    #             - if greater than 10, get the top 10 most frequent values
    #             - otherwise, get all of them :)

    #             - Encode values for that column and add to the DataFrame
    #             - Make sure the column before encoding does not stay!
    #     ## todo: will this affect the original input? should it?

    #     """
    #     # import pdb; pdb.set_trace()
    #     if not isinstance(X, pd.DataFrame):
    #         X = pd.DataFrame(X)

    #     # if X.isnull().any().any():
    #     #     raise ValueError("Dataframe to be encoded can not contain null values.")
    #     X_t = X.copy()
    #     cols_to_encode = self._get_cat_cols(X_t)
    #     encoded_X = pd.DataFrame()
    #     for col in X_t.columns:
    #         if col in cols_to_encode:
    #             if X_t[col].isnull().any():
    #                 X_t[col].fillna(-1, inplace=True)

    #             v = X_t[col].value_counts().to_frame()
    #             v.reset_index(inplace=True)
    #             v = v.sort_values([col, 'index'], ascending=[False, True])
    #             v.set_index('index', inplace=True)
    #             unique = v.head(self.top_n).index.tolist()
    #             for label in unique:
    #                 new_name = str(col) + "_" + str(label)
    #                 add = (X[col] == label).astype(int)
    #                 add = add.rename(new_name)
    #                 encoded_X = pd.concat([encoded_X, add], axis=1)
    #             X_t.drop(col, axis=1, inplace=True)
    #         else:
    #             encoded_X = pd.concat([encoded_X, X_t[col]], axis=1)
    #     print (encoded_X.columns)
    #     return encoded_X

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)
