import pandas as pd
from sklearn.impute import SimpleImputer as SkImputer

from evalml.pipelines.components.transformers import Transformer


class PerColumnImputer(Transformer):
    """Imputes missing data according to a specified imputation strategy per column"""
    name = 'Per Column Imputer'
    hyperparameter_ranges = {}

    def __init__(self, impute_strategies=None, fill_value=None, random_state=0):
        """Initializes an transformer that imputes missing data according to the specified imputation strategy per column."

        Arguments:
            impute_strategies (dict): Column and impute strategy pairings.
                Valid values include "mean", "median", "most_frequent", "constant" for numerical data,
                and "most_frequent", "constant" for object data types.
            fill_value (string): When impute_strategy == "constant", fill_value is used to replace missing data.
               Defaults to 0 when imputing numerical data and "missing_value" for strings or object data types.
        """
        parameters = {"impute_strategies": impute_strategies,
                      "fill_value": fill_value}

        imputers = {column: SkImputer(strategy=impute_strategies[column], fill_value=fill_value) for column in impute_strategies} if impute_strategies else None

        super().__init__(parameters=parameters,
                         component_obj=imputers,
                         random_state=random_state)

    def fit(self, X, y=None):
        """Fits imputers on data X

        Arguments:
            X (pd.DataFrame): Data to fit
            y (pd.Series, optional): Input Labels
        Returns:
            self
        """

        for column, imputer in self._component_obj.items():
            X[column] = imputer.fit(X[[column]])

        return self

    def transform(self, X, y=None):
        """Transforms data X by imputing missing values

        Arguments:
            X (pd.DataFrame): Data to transform
            y (pd.Series, optional): Input Labels
        Returns:
            pd.DataFrame: Transformed X
        """
        X_t = X
        for column, imputer in self._component_obj.items():
            X_t[column] = imputer.transform(X[[column]])
            if not isinstance(X_t, pd.DataFrame) and isinstance(X, pd.DataFrame):
                # skLearn's SimpleImputer loses track of column type, so we need to restore
                X_t = pd.DataFrame(X_t, columns=X.columns).astype(X.dtypes.to_dict())
        return X_t

    def fit_transform(self, X, y=None):
        """Fits imputer on data X then imputes missing values in X

        Arguments:
            X (pd.DataFrame): Data to fit and transform
            y (pd.Series): Labels to fit and transform
        Returns:
            pd.DataFrame: Transformed X
        """
        X_t = X
        for column, imputer in self._component_obj.items():
            X_t[column] = imputer.fit_transform(X[[column]], y)
            if not isinstance(X_t, pd.DataFrame) and isinstance(X, pd.DataFrame):
                # skLearn's SimpleImputer loses track of column type, so we need to restore
                X_t = pd.DataFrame(X_t, columns=X.columns).astype(X.dtypes.to_dict())
        return X_t
