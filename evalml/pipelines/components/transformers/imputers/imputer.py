"""Component that imputes missing data according to a specified imputation strategy."""
import pandas as pd
from woodwork import init_series

from evalml.pipelines.components.transformers import Transformer
from evalml.pipelines.components.transformers.imputers import KNNImputer, SimpleImputer
from evalml.utils import downcast_nullable_types, infer_feature_types
from evalml.utils.gen_utils import is_categorical_actually_boolean


class Imputer(Transformer):
    """Imputes missing data according to a specified imputation strategy.

    Args:
        categorical_impute_strategy (string): Impute strategy to use for string, object, boolean, categorical dtypes. Valid values include "most_frequent" and "constant".
        numeric_impute_strategy (string): Impute strategy to use for numeric columns. Valid values include "mean", "median", "most_frequent", and "constant".
        boolean_impute_strategy (string): Impute strategy to use for boolean columns. Valid values include "most_frequent" and "constant".
        categorical_fill_value (string): When categorical_impute_strategy == "constant", fill_value is used to replace missing data. The default value of None will fill with the string "missing_value".
        numeric_fill_value (int, float): When numeric_impute_strategy == "constant", fill_value is used to replace missing data. The default value of None will fill with 0.
        boolean_fill_value (bool): When boolean_impute_strategy == "constant", fill_value is used to replace missing data.  The default value of None will fill with True.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Imputer"
    hyperparameter_ranges = {
        "categorical_impute_strategy": ["most_frequent"],
        "numeric_impute_strategy": ["mean", "median", "most_frequent", "knn"],
        "boolean_impute_strategy": ["most_frequent", "knn"],
    }
    """{
        "categorical_impute_strategy": ["most_frequent"],
        "numeric_impute_strategy": ["mean", "median", "most_frequent", "knn"],
        "boolean_impute_strategy": ["most_frequent", "knn"]
    }"""
    _valid_categorical_impute_strategies = set(["most_frequent", "constant"])
    _valid_numeric_impute_strategies = set(
        ["mean", "median", "most_frequent", "constant", "knn"],
    )
    _valid_boolean_impute_strategies = set(["most_frequent", "constant", "knn"])

    def __init__(
        self,
        categorical_impute_strategy="most_frequent",
        categorical_fill_value=None,
        numeric_impute_strategy="mean",
        numeric_fill_value=None,
        boolean_impute_strategy="most_frequent",
        boolean_fill_value=None,
        random_seed=0,
        **kwargs,
    ):
        if categorical_impute_strategy not in self._valid_categorical_impute_strategies:
            raise ValueError(
                f"{categorical_impute_strategy} is an invalid parameter. Valid categorical imputation strategies are {', '.join(self._valid_numeric_impute_strategies)}",
            )
        if numeric_impute_strategy not in self._valid_numeric_impute_strategies:
            raise ValueError(
                f"{numeric_impute_strategy} is an invalid parameter. Valid numeric imputation strategies are {', '.join(self._valid_numeric_impute_strategies)}",
            )
        if boolean_impute_strategy not in self._valid_boolean_impute_strategies:
            raise ValueError(
                f"{boolean_impute_strategy} is an invalid parameter. Valid boolean imputation strategies are {', '.join(self._valid_boolean_impute_strategies)}",
            )

        parameters = {
            "categorical_impute_strategy": categorical_impute_strategy,
            "numeric_impute_strategy": numeric_impute_strategy,
            "boolean_impute_strategy": boolean_impute_strategy,
            "categorical_fill_value": categorical_fill_value,
            "numeric_fill_value": numeric_fill_value,
            "boolean_fill_value": boolean_fill_value,
        }
        parameters.update(kwargs)
        self._categorical_imputer = SimpleImputer(
            impute_strategy=categorical_impute_strategy,
            fill_value=categorical_fill_value,
            **kwargs,
        )
        if boolean_impute_strategy == "knn":
            self._boolean_imputer = KNNImputer(
                number_neighbors=1,
                **kwargs,
            )
        else:
            self._boolean_imputer = SimpleImputer(
                impute_strategy=boolean_impute_strategy,
                fill_value=boolean_fill_value,
                **kwargs,
            )
        if numeric_impute_strategy == "knn":
            self._numeric_imputer = KNNImputer(
                number_neighbors=3,
                **kwargs,
            )
        else:
            self._numeric_imputer = SimpleImputer(
                impute_strategy=numeric_impute_strategy,
                fill_value=numeric_fill_value,
                **kwargs,
            )
        self._all_null_cols = None
        self._numeric_cols = None
        self._categorical_cols = None
        self._boolean_cols = None
        super().__init__(
            parameters=parameters,
            component_obj=None,
            random_seed=random_seed,
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
        cat_cols = list(X.ww.select(["category"], return_schema=True).columns)
        bool_cols = list(
            X.ww.select(["BooleanNullable", "Boolean"], return_schema=True).columns,
        )
        numeric_cols = list(X.ww.select(["numeric"], return_schema=True).columns)

        # TODO: Remove this when columns with True/False/NaN are inferred properly as BooleanNullable.
        # If columns with boolean values and NaN are included with normal categorical columns, columns
        # with object dtypes are attempted to be cast to float64 with scikit-learn 1.1.  So we separate
        # boolean and categorical into separate imputers.
        for col in cat_cols:
            if is_categorical_actually_boolean(X, col):
                cat_cols.remove(col)
                bool_cols.append(col)

        nan_ratio = X.isna().sum() / X.shape[0]
        self._all_null_cols = nan_ratio[nan_ratio == 1].index.tolist()

        X_numerics = X[[col for col in numeric_cols if col not in self._all_null_cols]]
        if len(X_numerics.columns) > 0:
            self._numeric_imputer.fit(X_numerics, y)
            self._numeric_cols = X_numerics.columns

        X_categorical = X[[col for col in cat_cols if col not in self._all_null_cols]]
        if len(X_categorical.columns) > 0:
            self._categorical_imputer.fit(X_categorical, y)
            self._categorical_cols = X_categorical.columns

        X_boolean = X[[col for col in bool_cols if col not in self._all_null_cols]]
        if len(X_boolean.columns) > 0:
            self._boolean_imputer.fit(X_boolean, y)
            self._boolean_cols = X_boolean.columns
        return self

    def transform(self, X, y=None):
        """Transforms data X by imputing missing values.

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
            for numeric_col in X_numeric.columns:
                X_no_all_null.ww[numeric_col] = init_series(
                    imputed[numeric_col],
                    logical_type="Double",
                )

        if self._categorical_cols is not None and len(self._categorical_cols) > 0:
            X_categorical = X.ww[self._categorical_cols.tolist()]
            imputed = self._categorical_imputer.transform(X_categorical)
            X_no_all_null[X_categorical.columns] = imputed

        if self._boolean_cols is not None and len(self._boolean_cols) > 0:
            X_boolean = X.ww[self._boolean_cols.tolist()]
            imputed = self._boolean_imputer.transform(X_boolean)
            X_no_all_null[X_boolean.columns] = imputed

        X_no_all_null = downcast_nullable_types(X_no_all_null, ignore_null_cols=False)
        return X_no_all_null
