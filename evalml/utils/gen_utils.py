import importlib

import numpy as np


def import_or_raise(library, error_msg=None):
    '''
    Attempts to import the requested library by name.
    If the import fails, raises an ImportError.

    Arguments:
        library (str): the name of the library
        error_msg (str): error message to return if the import fails
    '''
    try:
        return importlib.import_module(library)
    except ImportError:
        if error_msg:
            raise ImportError(error_msg)
        raise ImportError("Failed to import {}".format(library))


def convert_to_seconds(input_str):
    hours = {'h', 'hr', 'hour', 'hours'}
    minutes = {'m', 'min', 'minute', 'minutes'}
    seconds = {'s', 'sec', 'second', 'seconds'}
    value, unit = input_str.split()
    if unit[-1] == 's' and len(unit) != 1:
        unit = unit[:-1]
    if unit in seconds:
        return float(value)
    elif unit in minutes:
        return float(value) * 60
    elif unit in hours:
        return float(value) * 3600
    else:
        msg = "Invalid unit. Units must be hours, mins, or seconds. Received '{}'".format(unit)
        raise AssertionError(msg)


def normalize_confusion_matrix(conf_mat, option='true'):
    """Normalizes a confusion matrix.

    Arguments:
        conf_mat (pd.DataFrame or np.array): confusion matrix to normalize
        option ({'true', 'pred', 'all'}): Option to normalize over the true (rows), predicted (columns) or all values. Defaults to 'true'.

    Returns:
        A normalized version of the input confusion matrix.

    """
    if option == 'true':
        conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1, keepdims=True)
    elif option == 'pred':
        conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=0, keepdims=True)
    elif option == 'all':
        conf_mat = conf_mat.astype('float') / conf_mat.sum()

    conf_mat = np.nan_to_num(conf_mat)
    return conf_mat
