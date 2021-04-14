
import numpy as np
import pandas as pd
import woodwork as ww

from evalml.utils.gen_utils import is_all_numeric

numeric_and_boolean_ww = [ww.logical_types.Integer, ww.logical_types.Double, ww.logical_types.Boolean]


def infer_feature_types(data, feature_types=None):
    """Create a Woodwork structure from the given list, pandas, or numpy input, with specified types for columns.
        If a column's type is not specified, it will be inferred by Woodwork.

    Arguments:
        data (pd.DataFrame): Input data to convert to a Woodwork data structure.
        feature_types (string, ww.logical_type obj, dict, optional): If data is a 2D structure, feature_types must be a dictionary
            mapping column names to the type of data represented in the column. If data is a 1D structure, then feature_types must be
            a Woodwork logical type or a string representing a Woodwork logical type ("Double", "Integer", "Boolean", "Categorical", "Datetime", "NaturalLanguage")

    Returns:
        A Woodwork data structure where the data type of each column was either specified or inferred.
    """
    ww_data = data
    if isinstance(data, ww.DataTable) or isinstance(data, ww.DataColumn):
        return ww_data
    if isinstance(data, list):
        ww_data = np.array(data)

    ww_data = ww_data.copy()
    if len(ww_data.shape) == 1:
        name = ww_data.name if isinstance(ww_data, pd.Series) else None
        return ww.DataColumn(ww_data, name=name, logical_type=feature_types)
    return ww.DataTable(ww_data, logical_types=feature_types)


def _convert_woodwork_types_wrapper(pd_data):
    """
    Converts a pandas data structure that may have extension or nullable dtypes to dtypes that numpy can understand and handle.

    Arguments:
        pd_data (pd.Series, pd.DataFrame, pd.ExtensionArray): Pandas data structure

    Returns:
        Modified pandas data structure (pd.DataFrame or pd.Series) with original data and dtypes that can be handled by numpy
    """
    nullable_to_numpy_mapping = {pd.Int64Dtype: 'int64',
                                 pd.BooleanDtype: 'bool',
                                 pd.StringDtype: 'object'}
    nullable_to_numpy_mapping_nan = {pd.Int64Dtype: 'float64',
                                     pd.BooleanDtype: 'object',
                                     pd.StringDtype: 'object'}

    if isinstance(pd_data, pd.api.extensions.ExtensionArray):
        if pd.isna(pd_data).any():
            return pd.Series(pd_data.to_numpy(na_value=np.nan), dtype=nullable_to_numpy_mapping_nan[type(pd_data.dtype)])
        return pd.Series(pd_data.to_numpy(na_value=np.nan), dtype=nullable_to_numpy_mapping[type(pd_data.dtype)])
    if (isinstance(pd_data, pd.Series) and type(pd_data.dtype) in nullable_to_numpy_mapping):
        if pd.isna(pd_data).any():
            return pd.Series(pd_data.to_numpy(na_value=np.nan), dtype=nullable_to_numpy_mapping_nan[type(pd_data.dtype)], index=pd_data.index, name=pd_data.name)
        return pd.Series(pd_data.to_numpy(na_value=np.nan), dtype=nullable_to_numpy_mapping[type(pd_data.dtype)], index=pd_data.index, name=pd_data.name)
    if isinstance(pd_data, pd.DataFrame):
        for col_name, col in pd_data.iteritems():
            if type(col.dtype) in nullable_to_numpy_mapping:
                if pd.isna(pd_data[col_name]).any():
                    pd_data[col_name] = pd.Series(pd_data[col_name].to_numpy(na_value=np.nan), dtype=nullable_to_numpy_mapping_nan[type(pd_data[col_name].dtype)])
                else:
                    pd_data[col_name] = pd_data[col_name].astype(nullable_to_numpy_mapping[type(col.dtype)])
    return pd_data


def _retain_custom_types_and_initalize_woodwork(old_woodwork_data, new_pandas_data, ltypes_to_ignore=None):
    """
    Helper method which will take an old Woodwork data structure and a new pandas data structure and return a
    new data structure that will try to retain as many logical types from the old data structure that exist in the new
    pandas data structure as possible.

    Arguments:
        old_woodwork_data (ww.DataTable): Woodwork data structure to use
        new_pandas_data (pd.DataFrame): Pandas data structure
        ltypes_to_ignore (list): List of Woodwork logical types to ignore. Columns from the old DataTable that have a logical type
        specified in this list will not have their logical types carried over to the new DataTable returned

    Returns:
        A new DataTable where any of the columns that exist in the old input DataTable and the new DataFrame try to retain
        the original logical type, if possible and not specified to be ignored.
    """
    if isinstance(old_woodwork_data, ww.DataColumn):
        if str(new_pandas_data.dtype) != old_woodwork_data.logical_type.pandas_dtype:
            try:
                return ww.DataColumn(new_pandas_data, logical_type=old_woodwork_data.logical_type)
            except (ValueError, TypeError):
                pass
        return ww.DataColumn(new_pandas_data)

    retained_logical_types = {}
    if ltypes_to_ignore is None:
        ltypes_to_ignore = []
    col_intersection = set(old_woodwork_data.columns).intersection(set(new_pandas_data.columns))
    logical_types = old_woodwork_data.logical_types
    for col in col_intersection:
        if logical_types[col] in ltypes_to_ignore:
            continue
        if str(new_pandas_data[col].dtype) != logical_types[col].pandas_dtype:
            try:
                new_pandas_data[col].astype(logical_types[col].pandas_dtype)
                retained_logical_types[col] = old_woodwork_data[col].logical_type
            except (ValueError, TypeError):
                pass
    return ww.DataTable(new_pandas_data, logical_types=retained_logical_types)


def _convert_numeric_dataset_pandas(X, y):
    """Convert numeric and non-null data to pandas datatype. Raises ValueError if there is null or non-numeric data.
    Used with data sampler strategies.

    Arguments:
        X (pd.DataFrame, np.ndarray, ww.DataTable): Data to transform
        y (pd.Series, np.ndarray, ww.DataColumn): Target data

    Returns:
        Tuple(pd.DataFrame, pd.Series): Transformed X and y"""
    X_ww = infer_feature_types(X)
    if not is_all_numeric(X_ww):
        raise ValueError('Values not all numeric or there are null values provided in the dataset')
    y_ww = infer_feature_types(y)
    X_ww = _convert_woodwork_types_wrapper(X_ww.to_dataframe())
    y_ww = _convert_woodwork_types_wrapper(y_ww.to_series())
    return X_ww, y_ww
