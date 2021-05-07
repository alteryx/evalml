import pandas as pd
from sklearn.impute import SimpleImputer as SkImputer
from woodwork.logical_types import NaturalLanguage

from evalml.pipelines.components.transformers import Transformer
from evalml.utils import (
    _retain_custom_types_and_initalize_woodwork,
    infer_feature_types
)


class SimpleImputer(Transformer):
    """Imputes missing data according to a specified imputation strategy."""
    name = 'Simple Imputer'
    hyperparameter_ranges = {"impute_strategy": ["mean", "median", "most_frequent"]}

    def __init__(self, impute_strategy="most_frequent", fill_value=None, random_seed=0, **kwargs):
        """Initalizes an transformer that imputes missing data according to the specified imputation strategy."

        Arguments:
            impute_strategy (string): Impute strategy to use. Valid values include "mean", "median", "most_frequent", "constant" for
               numerical data, and "most_frequent", "constant" for object data types.
            fill_value (string): When impute_strategy == "constant", fill_value is used to replace missing data.
               Defaults to 0 when imputing numerical data and "missing_value" for strings or object data types.
            random_seed (int): Seed for the random number generator. Defaults to 0.
        """
        parameters = {"impute_strategy": impute_strategy,
                      "fill_value": fill_value}
        parameters.update(kwargs)
        imputer = SkImputer(strategy=impute_strategy,
                            fill_value=fill_value,
                            **kwargs)
        self._all_null_cols = None
        super().__init__(parameters=parameters,
                         component_obj=imputer,
                         random_seed=random_seed)

    def fit(self, X, y=None):
        """Fits imputer to data. 'None' values are converted to np.nan before imputation and are
            treated as the same.

        Arguments:
            X (pd.DataFrame or np.ndarray): the input training data of shape [n_samples, n_features]
            y (pd.Series, optional): the target training data of length [n_samples]

        Returns:
            self
        """
        X = infer_feature_types(X)

        # Not using select because we just need column names, not a new dataframe
        natural_language_columns = [col for col, ltype in X.ww.logical_types.items() if ltype == NaturalLanguage]
        if natural_language_columns:
            X = X.ww.copy()
            X.ww.set_types({col: "Categorical" for col in natural_language_columns})

        # Convert all bool dtypes to category for fitting
        if (X.dtypes == bool).all():
            X = X.astype('category')

        self._component_obj.fit(X, y)
        self._all_null_cols = set(X.columns) - set(X.dropna(axis=1, how='all').columns)
        return self

    def transform(self, X, y=None):
        """Transforms input by imputing missing values. 'None' and np.nan values are treated as the same.

        Arguments:
            X (pd.DataFrame): Data to transform
            y (pd.Series, optional): Ignored.

        Returns:
            pd.DataFrame: Transformed X
        """
        X_ww = infer_feature_types(X)
        original_logical_types = X_ww.ww.schema.logical_types

        # Return early since bool dtype doesn't support nans and sklearn errors if all cols are bool
        if (X_ww.dtypes == bool).all():
            return X_ww

        X_null_dropped = X_ww.ww.drop(self._all_null_cols)

        # Not using select because we just need column names, not a new dataframe
        X_ww.ww.set_types({col: "Categorical" for col, ltype in X_ww.ww.logical_types.items() if ltype == NaturalLanguage})

        X_t = self._component_obj.transform(X_ww)

        X_t = pd.DataFrame(X_t, columns=X_null_dropped.columns)
        if not X_null_dropped.empty:
            X_t.index = X_null_dropped.index

        return _retain_custom_types_and_initalize_woodwork(original_logical_types, X_t)

    def fit_transform(self, X, y=None):
        """Fits on X and transforms X

        Arguments:
            X (pd.DataFrame): Data to fit and transform
            y (pd.Series, optional): Target data.

        Returns:
            pd.DataFrame: Transformed X
        """
        return self.fit(X, y).transform(X, y)
