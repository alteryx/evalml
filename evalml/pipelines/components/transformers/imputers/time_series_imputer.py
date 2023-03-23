"""Component that imputes missing data according to a specified timeseries-specific imputation strategy."""
import pandas as pd
import woodwork as ww
from woodwork.logical_types import (
    BooleanNullable,
    Double,
)

from evalml.pipelines.components.transformers import Transformer
from evalml.utils import infer_feature_types
from evalml.utils.nullable_type_utils import (
    _determine_fractional_type,
    _determine_non_nullable_equivalent,
)


class TimeSeriesImputer(Transformer):
    """Imputes missing data according to a specified timeseries-specific imputation strategy.

    This Transformer should be used after the `TimeSeriesRegularizer` in order to impute the missing values that were
    added to X and y (if passed).

    Args:
        categorical_impute_strategy (string): Impute strategy to use for string, object, boolean, categorical dtypes.
            Valid values include "backwards_fill" and "forwards_fill". Defaults to "forwards_fill".
        numeric_impute_strategy (string): Impute strategy to use for numeric columns. Valid values include
            "backwards_fill", "forwards_fill", and "interpolate". Defaults to "interpolate".
        target_impute_strategy (string): Impute strategy to use for the target column. Valid values include
            "backwards_fill", "forwards_fill", and "interpolate". Defaults to "forwards_fill".
        random_seed (int): Seed for the random number generator. Defaults to 0.

    Raises:
        ValueError: If categorical_impute_strategy, numeric_impute_strategy, or target_impute_strategy is not one of the valid values.
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
        ["backwards_fill", "forwards_fill", "interpolate"],
    )
    _valid_target_impute_strategies = set(
        ["backwards_fill", "forwards_fill", "interpolate"],
    )

    # Incompatibility: https://github.com/alteryx/evalml/issues/4001
    # TODO: Remove when support is added https://github.com/alteryx/evalml/issues/4014
    _integer_nullable_incompatibilities = ["X", "y"]
    _boolean_nullable_incompatibilities = ["y"]

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
                f"{categorical_impute_strategy} is an invalid parameter. Valid categorical impute strategies are {', '.join(self._valid_numeric_impute_strategies)}",
            )
        elif numeric_impute_strategy not in self._valid_numeric_impute_strategies:
            raise ValueError(
                f"{numeric_impute_strategy} is an invalid parameter. Valid numeric impute strategies are {', '.join(self._valid_numeric_impute_strategies)}",
            )
        elif target_impute_strategy not in self._valid_target_impute_strategies:
            raise ValueError(
                f"{target_impute_strategy} is an invalid parameter. Valid target column impute strategies are {', '.join(self._valid_target_impute_strategies)}",
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
            parameters=parameters,
            component_obj=None,
            random_seed=random_seed,
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

        nan_ratio = X.isna().sum() / X.shape[0]
        self._all_null_cols = nan_ratio[nan_ratio == 1].index.tolist()

        def _filter_cols(impute_strat, X):
            """Function to return which columns of the dataset to impute given the impute strategy."""
            cols = []
            if self.parameters["categorical_impute_strategy"] == impute_strat:
                if self.parameters["numeric_impute_strategy"] == impute_strat:
                    cols = list(X.columns)
                else:
                    cols = list(X.ww.select(exclude=["numeric"]).columns)
            elif self.parameters["numeric_impute_strategy"] == impute_strat:
                cols = list(X.ww.select(include=["numeric"]).columns)

            X_cols = [col for col in cols if col not in self._all_null_cols]
            if len(X_cols) > 0:
                return X_cols

        self._forwards_cols = _filter_cols("forwards_fill", X)
        self._backwards_cols = _filter_cols("backwards_fill", X)
        self._interpolate_cols = _filter_cols("interpolate", X)

        if y is not None:
            y = infer_feature_types(y)
            if y.isnull().any():
                self._impute_target = self.parameters["target_impute_strategy"]

        return self

    def transform(self, X, y=None):
        """Transforms data X by imputing missing values using specified timeseries-specific strategies. 'None' values are converted to np.nan before imputation and are treated as the same.

        Args:
            X (pd.DataFrame): Data to transform.
            y (pd.Series, optional): Optionally, target data to transform.

        Returns:
            pd.DataFrame: Transformed X and y
        """
        if len(self._all_null_cols) == X.shape[1]:
            df = pd.DataFrame(index=X.index)
            df.ww.init()
            return df, y
        X = infer_feature_types(X)
        if y is not None:
            y = infer_feature_types(y)

        # This will change the logical type of BooleanNullable/IntegerNullable/AgeNullable columns with nans
        # so we save the original schema to recreate it where possible after imputation
        original_schema = X.ww.schema
        X, y = self._handle_nullable_types(X, y)

        X_not_all_null = X.ww.drop(self._all_null_cols)

        # Because the TimeSeriesImputer is always used with the TimeSeriesRegularizer,
        # many of the columns containing nans may have originally been non nullable logical types.
        # We will use the non nullable equivalents where possible
        original_schema = original_schema.get_subset_schema(
            list(X_not_all_null.columns),
        )
        new_ltypes = {
            col: _determine_non_nullable_equivalent(ltype)
            for col, ltype in original_schema.logical_types.items()
        }

        if self._forwards_cols is not None:
            X_forward = X[self._forwards_cols]
            imputed = X_forward.pad()
            imputed.bfill(inplace=True)  # Fill in the first value, if missing
            X_not_all_null[X_forward.columns] = imputed

        if self._backwards_cols is not None:
            X_backward = X[self._backwards_cols]
            imputed = X_backward.bfill()
            imputed.pad(inplace=True)  # Fill in the last value, if missing
            X_not_all_null[X_backward.columns] = imputed

        if self._interpolate_cols is not None:
            X_interpolate = X_not_all_null[self._interpolate_cols]
            imputed = X_interpolate.interpolate()
            imputed.bfill(inplace=True)  # Fill in the first value, if missing
            X_not_all_null[X_interpolate.columns] = imputed

            # Interpolate may add floating point values to integer data, so we
            # have to update those logical types from the ones passed in to a fractional type
            # Note we ignore all other types of columns to maintain the types specified above
            int_cols_to_update = original_schema._filter_cols(
                include=["IntegerNullable", "AgeNullable"],
            )
            new_int_ltypes = {
                col: _determine_fractional_type(ltype)
                for col, ltype in original_schema.logical_types.items()
                if col in int_cols_to_update
            }
            new_ltypes.update(new_int_ltypes)
        X_not_all_null.ww.init(schema=original_schema, logical_types=new_ltypes)

        y_imputed = pd.Series(y)
        if y is not None and len(y) > 0:
            if self._impute_target == "forwards_fill":
                y_imputed = y.pad()
                y_imputed.bfill(inplace=True)
            elif self._impute_target == "backwards_fill":
                y_imputed = y.bfill()
                y_imputed.pad(inplace=True)
            elif self._impute_target == "interpolate":
                y_imputed = y.interpolate()
                y_imputed.bfill(inplace=True)
            # Re-initialize woodwork with the downcast logical type
            y_imputed = ww.init_series(y_imputed, logical_type=y.ww.logical_type)

        return X_not_all_null, y_imputed

    def _handle_nullable_types(self, X=None, y=None):
        """Transforms X and y to remove any incompatible nullable types for the time series imputer when the interpolate method is used.

        Args:
            X (pd.DataFrame, optional): Input data to a component of shape [n_samples, n_features].
                May contain nullable types.
            y (pd.Series, optional): The target of length [n_samples]. May contain nullable types.

        Returns:
            X, y with any incompatible nullable types downcasted to compatible equivalents when interpolate is used. Is NoOp otherwise.
        """
        if self._impute_target == "interpolate":
            # For BooleanNullable, we have to avoid Categorical columns
            # since the category dtype also has incompatibilities with linear interpolate, which is expected
            if isinstance(y.ww.logical_type, BooleanNullable):
                y = ww.init_series(y, Double)
            else:
                _, y = super()._handle_nullable_types(None, y)
        if self._interpolate_cols is not None:
            X, _ = super()._handle_nullable_types(X, None)

        return X, y
