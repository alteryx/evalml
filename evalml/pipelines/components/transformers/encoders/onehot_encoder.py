import category_encoders as ce
import pandas as pd
from .encoder import CategoricalEncoder
import numpy as np

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
        """
        One-hot encode the input DataFrame.

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
        """
        ### todo: will this affect the original input? should it?
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
                v = v.sort_values([col,'index'], ascending=[False, True])
                v.set_index('index', inplace=True)
                v.head(self.top_n).index.tolist()
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=False)
                encoded_X = pd.concat([encoded_X, dummies], axis=1)
                X.drop(col, axis=1, inplace=True)
            else:
                encoded_X = pd.concat([encoded_X, X[col]], axis=1)
        return encoded_X


    def fit_transform(self, X, y=None):
        return self.transform(X, y)



# def encode_features(feature_matrix):
#     encoded = []
#     feature_names = []
#     for feature in features:
#         feature_names.append(fname)
#     iterator = features
#     for f in iterator:
#         val_counts = X[f.get_name()].value_counts().to_frame()
#         index_name = val_counts.index.name
#         if index_name is None:
#             if 'index' in val_counts.columns:
#                 index_name = 'level_0'
#             else:
#                 index_name = 'index'
#         val_counts.reset_index(inplace=True)
#         val_counts = val_counts.sort_values([f.get_name(), index_name],
#                                             ascending=False)
#         val_counts.set_index(index_name, inplace=True)
#         select_n = top_n
#         if isinstance(top_n, dict):
#             select_n = top_n.get(f.get_name(), DEFAULT_TOP_N)
#         if drop_first:
#             select_n = min(len(val_counts), top_n)
#             select_n = max(select_n - 1, 1)
#         unique = val_counts.head(select_n).index.tolist()
#         for label in unique:
#             add = f == label
#             encoded.append(add)
#             X[add.get_name()] = (X[f.get_name()] == label).astype(int)

#         if include_unknown:
#             unknown = f.isin(unique).NOT().rename(f.get_name() + " is unknown")
#             encoded.append(unknown)
#             X[unknown.get_name()] = (~X[f.get_name()].isin(unique)).astype(int)

#         X.drop(f.get_name(), axis=1, inplace=True)

#     new_columns = []
#     for e in encoded:
#         new_columns.extend(e.get_feature_names())

#     new_X = X[new_columns]
#     iterator = new_X.columns
#     if verbose:
#         iterator = make_tqdm_iterator(iterable=new_X.columns,
#                                       total=len(new_X.columns),
#                                       desc="Encoding pass 2",
#                                       unit="feature")
#     for c in iterator:
#         try:
#             new_X[c] = pd.to_numeric(new_X[c], errors='raise')
#         except (TypeError, ValueError):
#             pass

#     return new_X, encoded