"""Component that imputes missing data according to a specified timeseries-specific imputation strategy."""
import pandas as pd

from evalml.pipelines.components.transformers import Transformer
from evalml.utils import infer_feature_types


class TimeSeriesImputer(Transformer):
    """Imputes missing data according to a specified timeseries-specific imputation strategy.

    Args:
        categorical_impute_strategy (string): Impute strategy to use for string, object, boolean, categorical dtypes.
                                              Valid values include "backwards_fill" and "forwards_fill". Defaults to "forwards_fill".
        numeric_impute_strategy (string): Impute strategy to use for numeric columns. Valid values include
                                          "backwards_fill", "forwards_fill", and "interpolate". Defaults to "interpolate".
        target_impute_strategy (string): Impute strategy to use for the target column. Valid values include "backwards_fill",
                                         "forwards_fill", and "interpolate". Defaults to "forwards_fill".
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    modifies_features = True
    modifies_target = True
    training_only = True

    name = "Time Series Imputer"
    hyperparameter_ranges = {
        "categorical_impute_strategy": ["backwards_fill", "forwards_fill"],
        "numeric_impute_strategy": ["backwards_fill", "forwards_fill", "interpolate"],
        "target_impute_strategy": ["backwards_fill", "forwards_fill", "interpolate"],
    }
    """{
        "categorical_impute_strategy": ["backwards_fill", "forwards_fill"],
        "numeric_impute_strategy": ["backwards_fill", "forwards_fill", "interpolate"],
        "target_impute_strategy": ["backwards_fill", "forwards_fill", "interpolate"],
    }"""
    _valid_categorical_impute_strategies = set(["backwards_fill", "forwards_fill"])
    _valid_numeric_impute_strategies = set(
        ["backwards_fill", "forwards_fill", "interpolate"]
    )
    _valid_target_impute_strategies = set(
        ["backwards_fill", "forwards_fill", "interpolate"]
    )

    def __init__(
        self,
        categorical_impute_strategy="forwards_fill",
        numeric_impute_strategy="interpolate",
        target_impute_strategy="forwards_fill",
        random_seed=0,
        **kwargs,
    ):
        if categorical_impute_strategy not in self._valid_categorical_impute_strategies:
            raise ValueError(
                f"{categorical_impute_strategy} is an invalid parameter. Valid categorical impute strategies are {', '.join(self._valid_numeric_impute_strategies)}"
            )
        elif numeric_impute_strategy not in self._valid_numeric_impute_strategies:
            raise ValueError(
                f"{numeric_impute_strategy} is an invalid parameter. Valid numeric impute strategies are {', '.join(self._valid_numeric_impute_strategies)}"
            )
        elif target_impute_strategy not in self._valid_target_impute_strategies:
            raise ValueError(
                f"{target_impute_strategy} is an invalid parameter. Valid target column impute strategies are {', '.join(self._valid_target_impute_strategies)}"
            )

        parameters = {
            "categorical_impute_strategy": categorical_impute_strategy,
            "numeric_impute_strategy": numeric_impute_strategy,
            "target_impute_strategy": target_impute_strategy,
        }
        parameters.update(kwargs)
        self._all_null_cols = None
        self._forwards_cols = None
        self._backwards_cols = None
        self._interpolate_cols = None
        self._impute_target = None
        super().__init__(
            parameters=parameters, component_obj=None, random_seed=random_seed
        )

    def fit(self, X, y=None):
        """Fits imputer to data.

        'None' values are converted to np.nan before imputation and are treated as the same.
        If a value is missing at the beginning or end of a column, that value will be imputed using
        backwards fill or forwards fill as necessary, respectively.

        Args:
            X (pd.DataFrame, np.ndarray): The input training data of shape [n_samples, n_features]
            y (pd.Series, optional): The target training data of length [n_samples]

        Returns:
            self
        """
        forwards_fill_cats = (
            ["category", "boolean"]
            if self.parameters["categorical_impute_strategy"] == "forwards_fill"
            else []
        )
        backwards_fill_cats = (
            ["category", "boolean"]
            if self.parameters["categorical_impute_strategy"] == "backwards_fill"
            else []
        )
        interpolation_cats = []
        if self.parameters["numeric_impute_strategy"] == "forwards_fill":
            forwards_fill_cats.append("numeric")
        elif self.parameters["numeric_impute_strategy"] == "backwards_fill":
            backwards_fill_cats.append("numeric")
        else:
            interpolation_cats.append("numeric")

        X = infer_feature_types(X)
        forwards_cols = list(
            X.ww.select(forwards_fill_cats, return_schema=True).columns
        )
        backwards_cols = list(
            X.ww.select(backwards_fill_cats, return_schema=True).columns
        )
        interpolation_cols = list(
            X.ww.select(interpolation_cats, return_schema=True).columns
        )

        nan_ratio = X.ww.describe().loc["nan_count"] / X.shape[0]
        self._all_null_cols = nan_ratio[nan_ratio == 1].index.tolist()

        X_forwards = X[[col for col in forwards_cols if col not in self._all_null_cols]]
        if len(X_forwards.columns) > 0:
            self._forwards_cols = X_forwards.columns

        X_backwards = X[
            [col for col in backwards_cols if col not in self._all_null_cols]
        ]
        if len(X_backwards.columns) > 0:
            self._backwards_cols = X_backwards.columns

        X_interpolate = X[
            [col for col in interpolation_cols if col not in self._all_null_cols]
        ]
        if len(X_interpolate.columns) > 0:
            self._interpolate_cols = X_interpolate.columns

        if y is not None:
            y = infer_feature_types(y)
            if y.isnull().any():
                self._impute_target = self.parameters["target_impute_strategy"]

        return self

    def transform(self, X, y=None):
        """Transforms data X by imputing missing values using specified timeseries-specific strategies. 'None' values are converted to np.nan before imputation and are treated as the same.

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
            return df, y

        X_no_all_null = X.ww.drop(self._all_null_cols)

        if self._forwards_cols is not None:
            X_forward = X.ww[self._forwards_cols.tolist()]
            imputed = X_forward.pad()
            imputed = imputed.bfill()  # Fill in the first value, if missing
            X_no_all_null[X_forward.columns] = imputed

        if self._backwards_cols is not None:
            X_backward = X.ww[self._backwards_cols.tolist()]
            imputed = X_backward.bfill()
            imputed = imputed.pad()  # Fill in the last value, if missing
            X_no_all_null[X_backward.columns] = imputed

        if self._interpolate_cols is not None:
            X_interpolate = X.ww[self._interpolate_cols.tolist()]
            imputed = X_interpolate.interpolate()
            imputed = imputed.bfill()  # Fill in the first value, if missing
            X_no_all_null[X_interpolate.columns] = imputed

        y_imputed = pd.Series(y)
        if self._impute_target == "forwards_fill":
            y_imputed = y.pad()
            y_imputed = y_imputed.bfill()
        elif self._impute_target == "backwards_fill":
            y_imputed = y.bfill()
            y_imputed = y_imputed.pad()
        elif self._impute_target == "interpolate":
            y_imputed = y.interpolate()
            y_imputed = y_imputed.bfill()

        return X_no_all_null, y_imputed
