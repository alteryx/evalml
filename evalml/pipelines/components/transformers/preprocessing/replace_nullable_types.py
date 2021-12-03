"""Transformer to replace features with the new nullable dtypes with a dtype that is compatible in EvalML."""
from woodwork import init_series
from woodwork.logical_types import (
    BooleanNullable,
    Categorical,
    Double,
    IntegerNullable,
)

from evalml.pipelines.components.transformers import Transformer
from evalml.utils import infer_feature_types


class ReplaceNullableTypes(Transformer):
    """Transformer to replace features with the new nullable dtypes with a dtype that is compatible in EvalML."""

    name = "Replace Nullable Types Transformer"
    hyperparameter_ranges = {}
    modifies_target = True
    """{}"""

    def __init__(self, random_seed=0, **kwargs):
        parameters = {}
        parameters.update(kwargs)

        self._nullable_int_cols = []
        self._nullable_bool_cols = []
        self._nullable_target = None
        super().__init__(
            parameters=parameters, component_obj=None, random_seed=random_seed
        )

    def fit(self, X, y=None):
        """Fits component to data.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series, optional): The target training data of length [n_samples].

        Returns:
            self
        """
        X_t = infer_feature_types(X, ignore_nullable_types=True)
        self._nullable_int_cols = list(
            X_t.ww.select(["IntegerNullable"], return_schema=True).columns
        )
        self._nullable_bool_cols = list(
            X_t.ww.select(["BooleanNullable"], return_schema=True).columns
        )

        if y is None:
            self._nullable_target = None
        else:
            y = infer_feature_types(y, ignore_nullable_types=True)
            if isinstance(y.ww.logical_type, IntegerNullable):
                self._nullable_target = "nullable_int"
            elif isinstance(y.ww.logical_type, BooleanNullable):
                self._nullable_target = "nullable_bool"
        return self

    def transform(self, X, y=None):
        """Transforms data X by replacing columns that contain nullable types with the appropriate replacement type.

        "float64" for nullable integers and "category" for nullable booleans.

        Args:
            X (pd.DataFrame): Data to transform
            y (pd.Series, optional): Ignored.

        Returns:
            pd.DataFrame: Transformed X
        """
        X_t = infer_feature_types(X, ignore_nullable_types=True)
        for col in self._nullable_int_cols:
            new_col = X_t[col].astype("float64")
            X_t.ww.pop(col)
            X_t.ww[col] = new_col
        for col in self._nullable_bool_cols:
            new_col = X_t[col].astype("category")
            X_t.ww.pop(col)
            X_t.ww[col] = new_col

        if y is not None:
            y_t = infer_feature_types(y, ignore_nullable_types=True)
            y_t = init_series(y_t)
            if self._nullable_target is not None:
                if self._nullable_target == "nullable_int":
                    y_t = init_series(y_t.astype("float64"), logical_type=Double)
                elif self._nullable_target == "nullable_bool":
                    y_t = init_series(y_t.astype("category"), logical_type=Categorical)
        elif y is None:
            y_t = None

        return X_t, y_t
