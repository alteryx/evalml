"""Transformer to replace features with the new nullable dtypes with a dtype that is compatible in EvalML."""
from pandas.core import arrays as pca
from woodwork.logical_types import BooleanNullable, IntegerNullable

from evalml.pipelines.components.transformers import Transformer
from evalml.utils import infer_feature_types


def is_nullable_int(df, col):
    is_pandas_nullable = isinstance(df.dtypes[col], pca.integer.Int64Dtype)
    is_ww_nullable = isinstance(df.ww.logical_types[col], IntegerNullable)
    return is_ww_nullable or is_pandas_nullable


def replace_nullable_int(series):
    return series.astype("float64")


def is_nullable_bool(df, col):
    is_pandas_nullable = isinstance(df.dtypes[col], pca.boolean.BooleanDtype)
    is_ww_nullable = isinstance(df.ww.logical_types[col], BooleanNullable)
    return is_ww_nullable or is_pandas_nullable


def replace_nullable_bool(series):
    return series.astype("category")


class ReplaceNullableTypes(Transformer):
    """
    Transformer to replace features with the new nullable dtypes with a dtype that is compatible in EvalML.
    """

    name = "Replace Nullable Types Transformer"
    hyperparameter_ranges = {}
    """{}"""

    def __init__(self, random_seed=0, **kwargs):
        parameters = {}
        parameters.update(kwargs)

        self._nullable_int_cols = []
        self._nullable_bool_cols = []
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
        for col in X_t.columns:
            if is_nullable_int(X_t, col):
                self._nullable_int_cols.append(col)
            elif is_nullable_bool(X_t, col):
                self._nullable_bool_cols.append(col)
        return self

    def transform(self, X, y=None):
        """Transforms data X by dropping columns that contain either the nullable integer or nullable boolean types.

        Args:
            X (pd.DataFrame): Data to transform
            y (pd.Series, optional): Ignored.

        Returns:
            pd.DataFrame: Transformed X
        """
        X_t = infer_feature_types(X, ignore_nullable_types=True)
        for col in self._nullable_int_cols:
            X_t[col] = replace_nullable_int(X_t[col])
        for col in self._nullable_bool_cols:
            X_t[col] = replace_nullable_bool(X_t[col])
        return X_t
