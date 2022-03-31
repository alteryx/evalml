"""Component that imputes missing data according to a specified timeseries-specific imputation strategy."""
import pandas as pd

from evalml.pipelines.components.transformers import Transformer
from evalml.utils import infer_feature_types


class TimeSeriesImputer(Transformer):
    """Imputes missing data according to a specified timeseries-specific imputation strategy.

    This Transformer should be used after the `TimeSeriesRegularizer` in order to impute the missing values that were
    added to X and y (if passed).

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
        X = infer_feature_types(X)

        forwards_cols = []
        backwards_cols = []
        interpolation_cols = []
        if self.parameters["numeric_impute_strategy"] == "forwards_fill":
            if self.parameters["categorical_impute_strategy"] == "forwards_fill":
                forwards_cols = list(X.columns)
            else:
                forwards_cols = list(
                    X.ww.select(include="numeric", return_schema=True).columns
                )
                backwards_cols = list(
                    X.ww.select(exclude="numeric", return_schema=True).columns
                )
        elif self.parameters["numeric_impute_strategy"] == "backwards_fill":
            if self.parameters["categorical_impute_strategy"] == "backwards_fill":
                backwards_cols = list(X.columns)
            else:
                forwards_cols = list(
                    X.ww.select(exclude="numeric", return_schema=True).columns
                )
                backwards_cols = list(
                    X.ww.select(include="numeric", return_schema=True).columns
                )
        else:
            interpolation_cols = list(
                X.ww.select(include="numeric", return_schema=True).columns
            )
            if self.parameters["categorical_impute_strategy"] == "forwards_fill":
                forwards_cols = list(
                    X.ww.select(exclude="numeric", return_schema=True).columns
                )
            else:
                backwards_cols = list(
                    X.ww.select(exclude="numeric", return_schema=True).columns
                )

        nan_ratio = X.ww.describe().loc["nan_count"] / X.shape[0]
        self._all_null_cols = nan_ratio[nan_ratio == 1].index.tolist()

        X_forwards = [col for col in forwards_cols if col not in self._all_null_cols]
        if len(X_forwards) > 0:
            self._forwards_cols = X_forwards

        X_backwards = [col for col in backwards_cols if col not in self._all_null_cols]
        if len(X_backwards) > 0:
            self._backwards_cols = X_backwards

        X_interpolate = [
            col for col in interpolation_cols if col not in self._all_null_cols
        ]
        if len(X_interpolate) > 0:
            self._interpolate_cols = X_interpolate

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
            X_forward = X.ww[self._forwards_cols]
            imputed = X_forward.pad()
            imputed.bfill(inplace=True)  # Fill in the first value, if missing
            X_no_all_null[X_forward.columns] = imputed

        if self._backwards_cols is not None:
            X_backward = X.ww[self._backwards_cols]
            imputed = X_backward.bfill()
            imputed.pad(inplace=True)  # Fill in the last value, if missing
            X_no_all_null[X_backward.columns] = imputed

        if self._interpolate_cols is not None:
            X_interpolate = X.ww[self._interpolate_cols]
            imputed = X_interpolate.interpolate()
            imputed.bfill(inplace=True)  # Fill in the first value, if missing
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
