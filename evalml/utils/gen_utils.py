import importlib

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state


def import_or_raise(library, error_msg=None):
    """Attempts to import the requested library by name.
    If the import fails, raises an ImportError.

    Arguments:
        library (str): the name of the library
        error_msg (str): error message to return if the import fails
    """
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


def get_random_state(seed):
    """Generates a numpy.random.RandomState instance using seed

    Arguments:
        seed (None, int, np.random.RandomState object): seed to generate numpy.random.RandomState with
    """
    return check_random_state(seed)


def get_random_seed(random_state, min_bound=None, max_bound=None):
    """Given a numpy.random.RandomState object, generate an int representing a seed value for another random number generator. Or, if given an int, simply pass that int.

    Arguments:
        random_state (int, numpy.random.RandomState): random state
        min_bound (None, int): if not default of None, will be min bound when generating seed (inclusive)
        max_bound (None, int): if not default of None, will be max bound when generating seed (exclusive)

    Returns:
        int: seed for random number generator
    """
    if isinstance(random_state, (int, np.integer)):
        return random_state
    iinfo = np.iinfo(np.integer)
    if min_bound is None:
        min_bound = iinfo.min
    if max_bound is None:
        max_bound = iinfo.max
    return random_state.randint(min_bound, max_bound)


def normalize_confusion_matrix(conf_mat, option='true'):
    """Normalizes a confusion matrix.

    Arguments:
        conf_mat (pd.DataFrame or np.array): confusion matrix to normalize
        option ({'true', 'pred', 'all', None}): Option to normalize over the rows ('true'), columns ('pred') or all ('all') values. If option is None, returns original confusion matrix. Defaults to 'true'.

    Returns:
        A normalized version of the input confusion matrix.

    """
    if option is None:
        return conf_mat
    elif option == 'true':
        conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
    elif option == 'pred':
        conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=0)
    elif option == 'all':
        conf_mat = conf_mat.astype('float') / conf_mat.sum().sum()

    if isinstance(conf_mat, pd.DataFrame):
        conf_mat = conf_mat.fillna(0)
    else:
        conf_mat = np.nan_to_num(conf_mat)

    return conf_mat


class classproperty:
    """Allows function to be accessed as a class level property.
        Example:
        class LogisticRegressionPipeline:
            component_graph = ['Simple Imputer', 'Logistic Regression Classifier']

            @classproperty
            def summary(cls):
            summary = ""
            for component in cls.component_graph:
                component = handle_component(component)
                summary += component.name + " + "
            return summary

            assert LogisticRegressionPipeline.summary == "Simple Imputer + Logistic Regression Classifier + "
            assert LogisticRegressionPipeline().summary == "Simple Imputer + Logistic Regression Classifier + "
    """

    def __init__(self, func):
        self.func = func

    def __get__(self, _, klass):
        return self.func(klass)
