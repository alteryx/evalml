import pandas as pd
from sklearn.impute import SimpleImputer as SkImputer

from evalml.pipelines.components.transformers import Transformer
from evalml.utils.gen_utils import boolean, categorical_dtypes, numeric_dtypes


class Imputer(Transformer):
    """Imputes missing data according to a specified imputation strategy."""
    name = "Imputer"
    hyperparameter_ranges = {
        "categorical_impute_strategy": ["most_frequent"],
        "numeric_impute_strategy": ["mean", "median", "most_frequent"]
    }
    _valid_categorical_impute_strategies = set(["most_frequent", "constant"])
    _valid_numeric_impute_strategies = set(["mean", "median", "most_frequent", "constant"])

    def __init__(self, categorical_impute_strategy="most_frequent",
                 numeric_impute_strategy="mean",
                 fill_value=None, random_state=0, **kwargs):
        """Initalizes an transformer that imputes missing data according to the specified imputation strategy."

        Arguments:
            categorical_impute_strategy (string): Impute strategy to use for string, object, boolean, categorical dtypes. Valid values include "most_frequent" and "constant".
            numeric_impute_strategy (string): Impute strategy to use for numeric dtypes. Valid values include "mean", "median", "most_frequent", and "constant".
            fill_value (string): When impute_strategy == "constant", fill_value is used to replace missing data.
               Defaults to 0 when imputing  data and "missing_value" for strings or object data types.
        """
        if categorical_impute_strategy not in self._valid_categorical_impute_strategies:
            raise ValueError(f"{categorical_impute_strategy} is an invalid parameter. Valid categorical impute strategies are {', '.join(self._valid_numeric_impute_strategies)}")
        elif numeric_impute_strategy not in self._valid_numeric_impute_strategies:
            raise ValueError(f"{numeric_impute_strategy} is an invalid parameter. Valid impute strategies are {', '.join(self._valid_numeric_impute_strategies)}")

        parameters = {"categorical_impute_strategy": categorical_impute_strategy,
                      "numeric_impute_strategy": numeric_impute_strategy,
                      "fill_value": fill_value}
        parameters.update(kwargs)
        self._categorical_imputer = SkImputer(strategy=categorical_impute_strategy,
                                              fill_value=fill_value,
                                              **kwargs)
        self._numeric_imputer = SkImputer(strategy=numeric_impute_strategy,
                                          fill_value=fill_value,
                                          **kwargs)
        self._all_null_cols = None
        self._numeric_cols = None
        self._categorical_cols = None
        super().__init__(parameters=parameters,
                         component_obj=None,
                         random_state=random_state)

    def fit(self, X, y=None):
        """Fits imputer to data

        Arguments:
            X (pd.DataFrame or np.array): the input training data of shape [n_samples, n_features]
            y (pd.Series, optional): the target training labels of length [n_samples]

        Returns:
            self
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self._all_null_cols = set(X.columns) - set(X.dropna(axis=1, how='all').columns)
        X_copy = X.copy()
        X_null_dropped = X_copy.drop(self._all_null_cols, axis=1, errors='ignore')

        X_numerics = X_null_dropped.select_dtypes(include=numeric_dtypes)
        if len(X_numerics.columns) > 0:
            self._numeric_imputer.fit(X_numerics, y)
            self._numeric_cols = X_numerics.columns

        X_categorical = X_null_dropped.select_dtypes(include=categorical_dtypes + boolean)
        if len(X_categorical.columns) > 0:
            self._categorical_imputer.fit(X_categorical, y)
            self._categorical_cols = X_categorical.columns
        return self

    def transform(self, X, y=None):
        """Transforms data X by imputing missing values

        Arguments:
            X (pd.DataFrame): Data to transform
            y (pd.Series, optional): Input Labels
        Returns:
            pd.DataFrame: Transformed X
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_null_dropped = X.copy()
        X_null_dropped.drop(self._all_null_cols, inplace=True, axis=1, errors='ignore')
        if X_null_dropped.empty:
            return X_null_dropped
        dtypes = X_null_dropped.dtypes.to_dict()

        if self._numeric_cols is not None and len(self._numeric_cols) > 0:
            X_numeric = X_null_dropped[self._numeric_cols]
            X_null_dropped[X_numeric.columns] = self._numeric_imputer.transform(X_numeric)
        if self._categorical_cols is not None and len(self._categorical_cols) > 0:
            X_categorical = X_null_dropped[self._categorical_cols]
            X_null_dropped[X_categorical.columns] = self._categorical_imputer.transform(X_categorical)

        transformed = X_null_dropped.astype(dtypes)
        transformed.reset_index(inplace=True, drop=True)
        return transformed
