"""Component that imputes missing data according to a specified imputation strategy."""
import pandas as pd

from evalml.pipelines.components.transformers import Transformer
from evalml.pipelines.components.transformers.imputers import SimpleImputer
from evalml.utils import infer_feature_types


class Imputer(Transformer):
    """Imputes missing data according to a specified imputation strategy.

    Args:
        categorical_impute_strategy (string): Impute strategy to use for string, object, boolean, categorical dtypes. Valid values include "most_frequent" and "constant".
        numeric_impute_strategy (string): Impute strategy to use for numeric columns. Valid values include "mean", "median", "most_frequent", and "constant".
        categorical_fill_value (string): When categorical_impute_strategy == "constant", fill_value is used to replace missing data. The default value of None will fill with the string "missing_value".
        numeric_fill_value (int, float): When numeric_impute_strategy == "constant", fill_value is used to replace missing data. The default value of None will fill with 0.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Imputer"
    hyperparameter_ranges = {
        "categorical_impute_strategy": ["most_frequent"],
        "numeric_impute_strategy": ["mean", "median", "most_frequent"],
    }
    """{
        "categorical_impute_strategy": ["most_frequent"],
        "numeric_impute_strategy": ["mean", "median", "most_frequent"],
    }"""
    _valid_categorical_impute_strategies = set(["most_frequent", "constant"])
    _valid_numeric_impute_strategies = set(
        ["mean", "median", "most_frequent", "constant"]
    )

    def __init__(
        self,
        categorical_impute_strategy="most_frequent",
        categorical_fill_value=None,
        numeric_impute_strategy="mean",
        numeric_fill_value=None,
        random_seed=0,
        **kwargs,
    ):
        if categorical_impute_strategy not in self._valid_categorical_impute_strategies:
            raise ValueError(
                f"{categorical_impute_strategy} is an invalid parameter. Valid categorical impute strategies are {', '.join(self._valid_numeric_impute_strategies)}"
            )
        elif numeric_impute_strategy not in self._valid_numeric_impute_strategies:
            raise ValueError(
                f"{numeric_impute_strategy} is an invalid parameter. Valid impute strategies are {', '.join(self._valid_numeric_impute_strategies)}"
            )

        parameters = {
            "categorical_impute_strategy": categorical_impute_strategy,
            "numeric_impute_strategy": numeric_impute_strategy,
            "categorical_fill_value": categorical_fill_value,
            "numeric_fill_value": numeric_fill_value,
        }
        parameters.update(kwargs)
        self._categorical_imputer = SimpleImputer(
            impute_strategy=categorical_impute_strategy,
            fill_value=categorical_fill_value,
            **kwargs,
        )
        self._numeric_imputer = SimpleImputer(
            impute_strategy=numeric_impute_strategy,
            fill_value=numeric_fill_value,
            **kwargs,
        )
        self._all_null_cols = None
        self._numeric_cols = None
        self._categorical_cols = None
        super().__init__(
            parameters=parameters, component_obj=None, random_seed=random_seed
        )

    def fit(self, X, y=None):
        """Fits imputer to data. 'None' values are converted to np.nan before imputation and are treated as the same.

        Args:
            X (pd.DataFrame, np.ndarray): The input training data of shape [n_samples, n_features]
            y (pd.Series, optional): The target training data of length [n_samples]

        Returns:
            self
        """
        X = infer_feature_types(X)
        cat_cols = list(
            X.ww.select(["category", "boolean"], return_schema=True).columns
        )
        numeric_cols = list(X.ww.select(["numeric"], return_schema=True).columns)

        nan_ratio = X.ww.describe().loc["nan_count"] / X.shape[0]
        self._all_null_cols = nan_ratio[nan_ratio == 1].index.tolist()

        X_numerics = X[[col for col in numeric_cols if col not in self._all_null_cols]]
        if len(X_numerics.columns) > 0:
            self._numeric_imputer.fit(X_numerics, y)
            self._numeric_cols = X_numerics.columns

        X_categorical = X[[col for col in cat_cols if col not in self._all_null_cols]]
        if len(X_categorical.columns) > 0:
            self._categorical_imputer.fit(X_categorical, y)
            self._categorical_cols = X_categorical.columns
        return self

    def transform(self, X, y=None):
        """Transforms data X by imputing missing values. 'None' values are converted to np.nan before imputation and are treated as the same.

        Args:
            X (pd.DataFrame): Data to transform
            y (pd.Series, optional): Ignored.

        Returns:
            pd.DataFrame: Transformed X
        """
        X = infer_feature_types(X)
        if len(self._all_null_cols) == X.shape[1]:
            df = pd.DataFrame(index=X.index)
            df.ww.init()
            return df

        X_no_all_null = X.ww.drop(self._all_null_cols)

        if self._numeric_cols is not None and len(self._numeric_cols) > 0:
            X_numeric = X.ww[self._numeric_cols.tolist()]
            imputed = self._numeric_imputer.transform(X_numeric)
            X_no_all_null[X_numeric.columns] = imputed

        if self._categorical_cols is not None and len(self._categorical_cols) > 0:
            X_categorical = X.ww[self._categorical_cols.tolist()]
            imputed = self._categorical_imputer.transform(X_categorical)
            X_no_all_null[X_categorical.columns] = imputed

        return X_no_all_null
