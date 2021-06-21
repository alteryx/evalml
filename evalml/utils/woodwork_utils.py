import numpy as np
import pandas as pd
import woodwork as ww

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
    """Create a Woodwork structure from the given list, pandas, or numpy input, with specified types for columns.
        If a column's type is not specified, it will be inferred by Woodwork.

    Arguments:
        data (pd.DataFrame, pd.Series): Input data to convert to a Woodwork data structure.
        feature_types (string, ww.logical_type obj, dict, optional): If data is a 2D structure, feature_types must be a dictionary
            mapping column names to the type of data represented in the column. If data is a 1D structure, then feature_types must be
            a Woodwork logical type or a string representing a Woodwork logical type ("Double", "Integer", "Boolean", "Categorical", "Datetime", "NaturalLanguage")

    Returns:
        A Woodwork data structure where the data type of each column was either specified or inferred.
    """
    if isinstance(data, list):
        data = _list_to_pandas(data)
    elif isinstance(data, np.ndarray):
        data = _numpy_to_pandas(data)

    _raise_value_error_if_nullable_types_detected(data)

    if data.ww.schema is not None:
        if isinstance(data, pd.DataFrame) and not ww.is_schema_valid(
            data, data.ww.schema
        ):
            raise ValueError(ww.get_invalid_schema_message(data, data.ww.schema))
        data.ww.init(schema=data.ww.schema)
        return data

    if isinstance(data, pd.Series):
        return ww.init_series(data, logical_type=feature_types)
    else:
        ww_data = data.copy()
        ww_data.ww.init(logical_types=feature_types)
        return ww_data


def _retain_custom_types_and_initalize_woodwork(
    old_logical_types, new_dataframe, ltypes_to_ignore=None
):
    """
    Helper method which will take an old Woodwork data structure and a new pandas data structure and return a
    new data structure that will try to retain as many logical types from the old data structure that exist in the new
    pandas data structure as possible.

    Arguments:
        old_logical_types (Dict): Logical types to try to retain.
        new_dataframe (pd.DataFrame): Pandas data structure
        ltypes_to_ignore (list): List of Woodwork logical types to ignore. Columns from the old DataFrame that have a logical type
        specified in this list will not have their logical types carried over to the new DataFrame returned

    Returns:
        A new DataFrame where any of the columns that exist in the old input DataFrame and the new DataFrame try to retain
        the original logical type, if possible and not specified to be ignored.
    """
    if isinstance(new_dataframe, pd.Series):
        return ww.init_series(new_dataframe, old_logical_types)
    if ltypes_to_ignore is None:
        ltypes_to_ignore = []
    old_logical_types = {k: type(v) for k, v in old_logical_types.items()}
    col_intersection = set(old_logical_types.keys()).intersection(
        set(new_dataframe.columns)
    )
    retained_logical_types = {
        col: ltype
        for col, ltype in old_logical_types.items()
        if col in col_intersection and ltype not in ltypes_to_ignore
    }
    new_dataframe.ww.init(logical_types=retained_logical_types)
    return new_dataframe


def _convert_numeric_dataset_pandas(X, y):
    """Convert numeric and non-null data to pandas datatype. Raises ValueError if there is null or non-numeric data.
    Used with data sampler strategies.

    Arguments:
        X (pd.DataFrame, np.ndarray): Data to transform
        y (pd.Series, np.ndarray): Target data

    Returns:
        Tuple(pd.DataFrame, pd.Series): Transformed X and y"""
    X_ww = infer_feature_types(X)
    if not is_all_numeric(X_ww):
        raise ValueError(
            "Values not all numeric or there are null values provided in the dataset"
        )
    y_ww = infer_feature_types(y)
    return X_ww, y_ww
