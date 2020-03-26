import pandas as pd
from sklearn.impute import SimpleImputer as SkImputer

from evalml.pipelines.components.transformers import Transformer


class SimpleImputer(Transformer):
    """Imputes missing data according to a specified imputation strategy"""
    name = 'Simple Imputer'
    hyperparameter_ranges = {"impute_strategy": ["mean", "median", "most_frequent"]}

    def __init__(self, impute_strategy='most_frequent', fill_value=None, random_state=0):
        """Initalizes an transformer that imputes missing data according to the specified imputation strategy."

        Arguments:
            impute_strategy (string): Impute strategy to use. Valid values include "mean", "median", "most_frequent", "constant" for
               numerical data, and "most_frequent", "constant" for object data types.
            fill_value (string): When impute_strategy == "constant", fill_value is used to replace missing data.
               Defaults to 0 when imputing numerical data and "missing_value" for strings or object data types.
        """
        self.impute_strategy = impute_strategy
        self.fill_value = fill_value
        imputer = SkImputer(strategy=impute_strategy,
                            fill_value=fill_value)
        super().__init__(component_obj=imputer,
                         random_state=random_state)

    def transform(self, X, y=None):
        """Transforms data X by imputing missing values

        Arguments:
            X (pd.DataFrame): Data to transform
            y (pd.Series, optional): Input Labels
        Returns:
            pd.DataFrame: Transformed X
        """
        X_t = super().transform(X, y=y)
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
        X_t = super().fit_transform(X, y=y)
        if not isinstance(X_t, pd.DataFrame) and isinstance(X, pd.DataFrame):
            # skLearn's SimpleImputer loses track of column type, so we need to restore
            X_t = pd.DataFrame(X_t, columns=X.columns).astype(X.dtypes.to_dict())
        return X_t
