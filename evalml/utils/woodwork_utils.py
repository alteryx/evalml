
import numpy as np
import pandas as pd
import woodwork as ww

numeric_and_boolean_ww = [ww.logical_types.Integer, ww.logical_types.Double, ww.logical_types.Boolean]


def infer_feature_types(data, feature_types=None):
    """Create a Woodwork structure from the given pandas or numpy input, with specified types for columns.
        If a column's type is not specified, it will be inferred by Woodwork.

    Arguments:
        data (pd.DataFrame): Input data to convert to a Woodwork data structure.
        feature_types (string, ww.logical_type obj, dict, optional): If data is a 2D structure, feature_types must be a dictionary
            mapping column names to the type of data represented in the column. If data is a 1D structure, then feature_types must be
            a Woodwork logical type or a string representing a Woodwork logical type ("Double", "Integer", "Boolean", "Categorical", "Datetime", "NaturalLanguage")

    Returns:
        A Woodwork data structure where the data type of each column was either specified or inferred.
    """
    ww_data = _convert_to_woodwork_structure(data)
    if feature_types is not None:
        if len(ww_data.shape) == 1:
            ww_data = ww_data.set_logical_type(feature_types)
        else:
            ww_data = ww_data.set_types(logical_types=feature_types)
    return ww_data


def _convert_to_woodwork_structure(data):
    """
    Takes input data structure, and if it is not a Woodwork data structure already, will convert it to a Woodwork DataTable or DataColumn structure.
    """
    ww_data = data
    if isinstance(data, ww.DataTable) or isinstance(data, ww.DataColumn):
        return ww_data
    if isinstance(data, list):
        ww_data = np.array(data)

    ww_data = ww_data.copy()
    if len(ww_data.shape) == 1:
        name = ww_data.name if isinstance(ww_data, pd.Series) else None
        return ww.DataColumn(ww_data, name=name)
    return ww.DataTable(ww_data)


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


def reconvert(old_logical_types, new_dataframe):
    """
    Helper method in EvalML which will take old datatable, new dataframe and return new datatable,
    handling updates in logical types.
    """
