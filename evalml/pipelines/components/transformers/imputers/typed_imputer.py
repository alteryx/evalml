import pandas as pd
from sklearn.impute import SimpleImputer as SkImputer

from evalml.pipelines.components.transformers import Transformer
from evalml.utils.gen_utils import numerics


class TypedImputer(Transformer):
    """Imputes missing data according to a specified imputation strategy."""
    name = "Typed Imputer"
    hyperparameter_ranges = {
        "categorical_impute_strategy": ["most_frequent"],
        "numeric_impute_strategy": ["mean", "median", "most_frequent"]
    }
    _valid_categorical_impute_strategies = set(["most_frequent"])
    _valid_numeric_impute_strategies = set(["mean", "median", "most_frequent"])

    def __init__(self, categorical_impute_strategy="most_frequent",
                 numeric_impute_strategy="mean",
                 fill_value=None, random_state=0, **kwargs):
        """Initalizes an transformer that imputes missing data according to the specified imputation strategy."

        Arguments:
            impute_strategy (string): Impute strategy to use. Valid values include "mean", "median", "most_frequent", "constant" for
               numerical data, and "most_frequent", "constant" for object data types.
            fill_value (string): When impute_strategy == "constant", fill_value is used to replace missing data.
               Defaults to 0 when imputing numerical data and "missing_value" for strings or object data types.
        """
        if categorical_impute_strategy not in self._valid_categorical_impute_strategies:
            raise ValueError(f"{categorical_impute_strategy} is an invalid parameter. Valid categorical impute strategies are {', '.join(self._valid_numeric_impute_strategies)}")
        elif numeric_impute_strategy not in self._valid_numeric_impute_strategies:
            raise ValueError(f"{numeric_impute_strategy} is an invalid parameter. Valid numerical impute strategies are {', '.join(self._valid_numeric_impute_strategies)}")

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

        X_numerics = X.select_dtypes(include=numerics)
        if len(X_numerics.columns) > 0:
            self._numeric_imputer.fit(X_numerics, y)

        X_categorical = X.select_dtypes(exclude=numerics)
        if len(X_categorical.columns) > 0:
            self._categorical_imputer.fit(X_categorical, y)

        self._all_null_cols = set(X.columns) - set(X.dropna(axis=1, how='all').columns)
        return self

    def _transform_helper(self, X, y=None, calculate_numerics=False):
        """Helper method to transform data X by imputing missing values

        Arguments:
            X (pd.DataFrame): Data to transform
            y (pd.Series, optional): Input Labels
            calculate_numerics (bool): If True, computes transformation for numeric columns,
                otherwise computes transformation for all other columns.

        Returns:
            pd.DataFrame: Transformed X for input type or empty DataFrame if
                input does not contain selected datatypes.

        """
        if calculate_numerics:
            X_to_transform = X.select_dtypes(include=numerics)
            if len(X_to_transform.columns) == 0:
                return pd.DataFrame()
            X_t = self._numeric_imputer.transform(X_to_transform)
        else:
            X_to_transform = X.select_dtypes(exclude=numerics)
            if len(X_to_transform.columns) == 0:
                return pd.DataFrame()
            X_t = self._categorical_imputer.transform(X_to_transform)

        if not isinstance(X_t, pd.DataFrame):
            X_null_dropped = X_to_transform.drop(self._all_null_cols, axis=1, errors='ignore')
            if X_null_dropped.empty:
                return pd.DataFrame(X_t, columns=X_null_dropped.columns)
            return pd.DataFrame(X_t, columns=X_null_dropped.columns).astype(X_null_dropped.dtypes.to_dict())

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
        return pd.concat([self._transform_helper(X, y, calculate_numerics=True),
                          self._transform_helper(X, y, calculate_numerics=False)], axis=1)

    def fit_transform(self, X, y=None):
        """Fits on X and transforms X

        Arguments:
            X (pd.DataFrame): Data to fit and transform
            y (pd. DataFrame): Labels to fit and transform
        Returns:
            pd.DataFrame: Transformed X
        """
        return self.fit(X, y).transform(X, y)
