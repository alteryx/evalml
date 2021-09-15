"""Woodwork utility methods."""
import numpy as np
import pandas as pd
import woodwork as ww
from woodwork.logical_types import Unknown

from evalml.utils.gen_utils import is_all_numeric

numeric_and_boolean_ww = [
    ww.logical_types.Integer.type_string,
    ww.logical_types.Double.type_string,
    ww.logical_types.Boolean.type_string,
]


def _numpy_to_pandas(array):
    if len(array.shape) == 1:
        data = pd.Series(array)
    else:
        data = pd.DataFrame(array)
    return data


def _list_to_pandas(list):
    return _numpy_to_pandas(np.array(list))


_nullable_types = {"Int64", "Float64", "boolean"}


def _raise_value_error_if_nullable_types_detected(data):
    types = {data.name: data.dtype} if isinstance(data, pd.Series) else data.dtypes
    cols_with_nullable_types = {
        col: str(ptype)
        for col, ptype in dict(types).items()
        if str(ptype) in _nullable_types
    }
    if cols_with_nullable_types:
        raise ValueError(
            "Evalml does not support the new pandas nullable types because "
            "our dependencies (sklearn, xgboost, lightgbm) do not support them yet."
            "If your data does not have missing values, please use the non-nullable types (bool, int64, float64). "
            "If your data does have missing values, use float64 for int and float columns and category for boolean columns. "
            f"These are the columns with nullable types: {list(cols_with_nullable_types.items())}"
        )


def infer_feature_types(data, feature_types=None):
    """Create a Woodwork structure from the given list, pandas, or numpy input, with specified types for columns. If a column's type is not specified, it will be inferred by Woodwork.

    Args:
        data (pd.DataFrame, pd.Series): Input data to convert to a Woodwork data structure.
        feature_types (string, ww.logical_type obj, dict, optional): If data is a 2D structure, feature_types must be a dictionary
            mapping column names to the type of data represented in the column. If data is a 1D structure, then feature_types must be
            a Woodwork logical type or a string representing a Woodwork logical type ("Double", "Integer", "Boolean", "Categorical", "Datetime", "NaturalLanguage")

    Returns:
        A Woodwork data structure where the data type of each column was either specified or inferred.

    Raises:
        ValueError: If there is a mismatch between the dataframe and the woodwork schema.
    """
    if isinstance(data, list):
        data = _list_to_pandas(data)
    elif isinstance(data, np.ndarray):
        data = _numpy_to_pandas(data)

    _raise_value_error_if_nullable_types_detected(data)

    def convert_all_nan_unknown_to_double(data):
        def is_column_pd_na(data, col):
            return data[col].isna().all()

        def is_column_unknown(data, col):
            return isinstance(data.ww.logical_types[col], Unknown)

        if isinstance(data, pd.DataFrame):
            all_null_unk_cols = [
                col
                for col in data.columns
                if (is_column_unknown(data, col) and is_column_pd_na(data, col))
            ]
            if len(all_null_unk_cols):
                for col in all_null_unk_cols:
                    data.ww.set_types({col: "Double"})
        return data

    if data.ww.schema is not None:
        if isinstance(data, pd.DataFrame) and not ww.is_schema_valid(
            data, data.ww.schema
        ):
            ww_error = ww.get_invalid_schema_message(data, data.ww.schema)
            if "dtype mismatch" in ww_error:
                ww_error = (
                    "Dataframe types are not consistent with logical types. This usually happens "
                    "when a data transformation does not go through the ww accessor. Call df.ww.init() to "
                    f"get rid of this message. This is a more detailed message about the mismatch: {ww_error}"
                )
            else:
                ww_error = f"{ww_error}. Please initialize ww with df.ww.init() to get rid of this message."
            raise ValueError(ww_error)
        data.ww.init(schema=data.ww.schema)
        return convert_all_nan_unknown_to_double(data)

    if isinstance(data, pd.Series):
        if all(data.isna()):
            data = data.replace(pd.NA, np.nan)
            feature_types = "Double"
        return ww.init_series(data, logical_type=feature_types)
    else:
        ww_data = data.copy()
        ww_data.ww.init(logical_types=feature_types)
        return convert_all_nan_unknown_to_double(ww_data)


def _convert_numeric_dataset_pandas(X, y):
    """Convert numeric and non-null data to pandas datatype. Raises ValueError if there is null or non-numeric data. Used with data sampler strategies.

    Args:
        X (pd.DataFrame, np.ndarray): Data to transform.
        y (pd.Series, np.ndarray): Target data.

    Returns:
        Tuple(pd.DataFrame, pd.Series): Transformed X and y.
    """
    X_ww = infer_feature_types(X)
    if not is_all_numeric(X_ww):
        raise ValueError(
            "Values not all numeric or there are null values provided in the dataset"
        )
    y_ww = infer_feature_types(y)
    return X_ww, y_ww
