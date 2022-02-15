"""Component that imputes missing data according to a specified imputation strategy."""
import pandas as pd
from sklearn.impute import SimpleImputer as SkImputer
from woodwork.logical_types import Boolean, BooleanNullable, NaturalLanguage

from evalml.pipelines.components.transformers import Transformer
from evalml.utils import infer_feature_types


class SimpleImputer(Transformer):
    """Imputes missing data according to a specified imputation strategy.

    Args:
        impute_strategy (string): Impute strategy to use. Valid values include "mean", "median", "most_frequent", "constant" for
           numerical data, and "most_frequent", "constant" for object data types.
        fill_value (string): When impute_strategy == "constant", fill_value is used to replace missing data.
           Defaults to 0 when imputing numerical data and "missing_value" for strings or object data types.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Simple Imputer"
    hyperparameter_ranges = {"impute_strategy": ["mean", "median", "most_frequent"]}
    """{
        "impute_strategy": ["mean", "median", "most_frequent"]
    }"""

    def __init__(
        self, impute_strategy="most_frequent", fill_value=None, random_seed=0, **kwargs
    ):
        self.impute_strategy = impute_strategy
        parameters = {"impute_strategy": impute_strategy, "fill_value": fill_value}
        parameters.update(kwargs)
        imputer = SkImputer(strategy=impute_strategy, fill_value=fill_value, **kwargs)
        self._all_null_cols = None
        super().__init__(
            parameters=parameters, component_obj=imputer, random_seed=random_seed
        )

    def _get_columns_of_type(self, X, ww_dtype):
        return [
            col for col, ltype in X.ww.logical_types.items() if type(ltype) == ww_dtype
        ]

    def _drop_natural_language_columns(self, X):
        # Not using select because we just need column names, not a new dataframe
        natural_language_columns = self._get_columns_of_type(X, NaturalLanguage)
        if natural_language_columns:
            X = X.ww.copy()
            X = X.ww.drop(columns=natural_language_columns)
        return X, natural_language_columns

    def _set_boolean_columns_to_categorical(self, X):
        boolean_null_columns = self._get_columns_of_type(X, BooleanNullable)
        boolean_columns = self._get_columns_of_type(X, Boolean)
        boolean_columns += boolean_null_columns
        if boolean_columns:
            X = X.ww.copy()
            X.ww.set_types({col: "Categorical" for col in boolean_columns})
        return X

    def fit(self, X, y=None):
        """Fits imputer to data. 'None' values are converted to np.nan before imputation and are treated as the same.

        Args:
            X (pd.DataFrame or np.ndarray): the input training data of shape [n_samples, n_features]
            y (pd.Series, optional): the target training data of length [n_samples]

        Returns:
            self

        Raises:
            ValueError if non-numeric data is given to an imputer with "median" or "mean" strategies.
        """
        X = infer_feature_types(X)

        nan_ratio = X.ww.describe().loc["nan_count"] / X.shape[0]
        self._all_null_cols = nan_ratio[nan_ratio == 1].index.tolist()

        # Determine if imputer is being used with incompatible imputation strategies
        boolean_columns = self._get_columns_of_type(X, BooleanNullable)
        if self.impute_strategy in ["median", "mean"] and len(boolean_columns) > 0:
            raise ValueError(
                f"Cannot use {self.impute_strategy} strategy with non-numeric data: {boolean_columns} contain boolean values and cannot be imputed with the 'median' or 'mode' strategy."
            )

        X, _ = self._drop_natural_language_columns(X)
        X = self._set_boolean_columns_to_categorical(X)

        # If the Dataframe only had one natural language column, do nothing.
        if X.shape[1] == 0:
            return self

        self._component_obj.fit(X, y)
        return self

    def transform(self, X, y=None):
        """Transforms input by imputing missing values. 'None' and np.nan values are treated as the same.

        Args:
            X (pd.DataFrame): Data to transform.
            y (pd.Series, optional): Ignored.

        Returns:
            pd.DataFrame: Transformed X
        """
        X = infer_feature_types(X)
        original_schema = X.ww.schema

        # Return early since bool dtype doesn't support nans and sklearn errors if all cols are bool
        if (X.dtypes == bool).all():
            return X

        not_all_null_cols = [col for col in X.columns if col not in self._all_null_cols]
        original_index = X.index

        X_t, natural_language_columns = self._drop_natural_language_columns(X)
        if X_t.shape[-1] == 0:
            return X

        not_all_null_or_nat_lang_cols = [
            col for col in not_all_null_cols if col not in natural_language_columns
        ]

        X_t = self._component_obj.transform(X_t)
        X_t = pd.DataFrame(X_t, columns=not_all_null_or_nat_lang_cols)
        if natural_language_columns:
            X_t = pd.merge(
                X_t, X[natural_language_columns], left_index=True, right_index=True
            )
        if not_all_null_cols:
            X_t.index = original_index
        X_t.ww.init(schema=original_schema.get_subset_schema(X_t.columns))

        return X_t

    def fit_transform(self, X, y=None):
        """Fits on X and transforms X.

        Args:
            X (pd.DataFrame): Data to fit and transform
            y (pd.Series, optional): Target data.

        Returns:
            pd.DataFrame: Transformed X
        """
        return self.fit(X, y).transform(X, y)
