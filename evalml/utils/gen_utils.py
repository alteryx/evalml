import importlib
import warnings
from collections import namedtuple

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state

from evalml.exceptions import (
    EnsembleMissingPipelinesError,
    MissingComponentError
)
from evalml.utils import get_logger

logger = get_logger(__file__)

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
boolean = ['bool']
numeric_and_boolean_dtypes = numeric_dtypes + boolean
categorical_dtypes = ['object', 'category']
datetime_dtypes = [np.datetime64]


def import_or_raise(library, error_msg=None, warning=False):
    """Attempts to import the requested library by name.
    If the import fails, raises an ImportError or warning.

    Arguments:
        library (str): the name of the library
        error_msg (str): error message to return if the import fails
        warning (bool): if True, import_or_raise gives a warning instead of ImportError. Defaults to False.
    """
    try:
        return importlib.import_module(library)
    except ImportError:
        if error_msg is None:
            error_msg = ""
        msg = (f"Missing optional dependency '{library}'. Please use pip to install {library}. {error_msg}")
        if warning:
            warnings.warn(msg)
        else:
            raise ImportError(msg)
    except Exception as ex:
        msg = (f"An exception occurred while trying to import `{library}`: {str(ex)}")
        if warning:
            warnings.warn(msg)
        else:
            raise Exception(msg)


def convert_to_seconds(input_str):
    """Converts a string describing a length of time to its length in seconds."""
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


# specifies the min and max values a seed to np.random.RandomState is allowed to take.
# these limits were chosen to fit in the numpy.int32 datatype to avoid issues with 32-bit systems
# see https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.RandomState.html
SEED_BOUNDS = namedtuple('SEED_BOUNDS', ('min_bound', 'max_bound'))(0, 2**31 - 1)


def get_random_state(seed):
    """Generates a numpy.random.RandomState instance using seed.

    Arguments:
        seed (None, int, np.random.RandomState object): seed to use to generate numpy.random.RandomState. Must be between SEED_BOUNDS.min_bound and SEED_BOUNDS.max_bound, inclusive. Otherwise, an exception will be thrown.
    """
    if isinstance(seed, (int, np.integer)) and (seed < SEED_BOUNDS.min_bound or SEED_BOUNDS.max_bound < seed):
        raise ValueError('Seed "{}" is not in the range [{}, {}], inclusive'.format(seed, SEED_BOUNDS.min_bound, SEED_BOUNDS.max_bound))
    return check_random_state(seed)


def get_random_seed(random_state, min_bound=SEED_BOUNDS.min_bound, max_bound=SEED_BOUNDS.max_bound):
    """Given a numpy.random.RandomState object, generate an int representing a seed value for another random number generator. Or, if given an int, return that int.

    To protect against invalid input to a particular library's random number generator, if an int value is provided, and it is outside the bounds "[min_bound, max_bound)", the value will be projected into the range between the min_bound (inclusive) and max_bound (exclusive) using modular arithmetic.

    Arguments:
        random_state (int, numpy.random.RandomState): random state
        min_bound (None, int): if not default of None, will be min bound when generating seed (inclusive). Must be less than max_bound.
        max_bound (None, int): if not default of None, will be max bound when generating seed (exclusive). Must be greater than min_bound.

    Returns:
        int: seed for random number generator
    """
    if not min_bound < max_bound:
        raise ValueError("Provided min_bound {} is not less than max_bound {}".format(min_bound, max_bound))
    if isinstance(random_state, np.random.RandomState):
        return random_state.randint(min_bound, max_bound)
    if random_state < min_bound or random_state >= max_bound:
        return ((random_state - min_bound) % (max_bound - min_bound)) + min_bound
    return random_state


def check_random_state_equality(random_state, other_random_state):
    """Method to check for equality of two numpy.random.RandomState objects"""
    for self_rs_attr, other_rs_attr in zip(random_state.get_state(), other_random_state.get_state()):
        if isinstance(self_rs_attr, np.ndarray) and isinstance(other_rs_attr, np.ndarray):
            if not (self_rs_attr == other_rs_attr).all():
                return False
        else:
            if not (self_rs_attr == other_rs_attr):
                return False
    return True


class classproperty:
    """Allows function to be accessed as a class level property.
        Example:
        class LogisticRegressionBinaryPipeline(PipelineBase):
            component_graph = ['Simple Imputer', 'Logistic Regression Classifier']

            @classproperty
            def summary(cls):
            summary = ""
            for component in cls.component_graph:
                component = handle_component_class(component)
                summary += component.name + " + "
            return summary

            assert LogisticRegressionBinaryPipeline.summary == "Simple Imputer + Logistic Regression Classifier + "
            assert LogisticRegressionBinaryPipeline().summary == "Simple Imputer + Logistic Regression Classifier + "
    """

    def __init__(self, func):
        self.func = func

    def __get__(self, _, klass):
        return self.func(klass)


def _get_subclasses(base_class):
    """Gets all of the leaf nodes in the hiearchy tree for a given base class.

    Arguments:
        base_class (abc.ABCMeta): Class to find all of the children for.

    Returns:
        subclasses (list): List of all children that are not base classes.
    """

    classes_to_check = base_class.__subclasses__()
    subclasses = []

    while classes_to_check:
        subclass = classes_to_check.pop()
        children = subclass.__subclasses__()

        if children:
            classes_to_check.extend(children)
        else:
            subclasses.append(subclass)

    return subclasses


_not_used_in_automl = {'BaselineClassifier', 'BaselineRegressor',
                       'StackedEnsembleClassifier', 'StackedEnsembleRegressor',
                       'ModeBaselineBinaryPipeline', 'BaselineBinaryPipeline', 'MeanBaselineRegressionPipeline',
                       'BaselineRegressionPipeline', 'ModeBaselineMulticlassPipeline', 'BaselineMulticlassPipeline'}


def get_importable_subclasses(base_class, used_in_automl=True):
    """Get importable subclasses of a base class. Used to list all of our
    estimators, transformers, components and pipelines dynamically.

    Arguments:
        base_class (abc.ABCMeta): Base class to find all of the subclasses for.
        args (list): Args used to instantiate the subclass. [{}] for a pipeline, and [] for
            all other classes.
        used_in_automl: Not all components/pipelines/estimators are used in automl search. If True,
            only include those subclasses that are used in the search. This would mean excluding classes related to
            ExtraTrees, ElasticNet, and Baseline estimators.

    Returns:
        List of subclasses.
    """
    all_classes = _get_subclasses(base_class)

    classes = []
    for cls in all_classes:
        if 'evalml.pipelines' not in cls.__module__:
            continue
        try:
            cls()
            classes.append(cls)
        except (ImportError, MissingComponentError, TypeError):
            logger.debug(f'Could not import class {cls.__name__} in get_importable_subclasses')
        except EnsembleMissingPipelinesError:
            classes.append(cls)
    if used_in_automl:
        classes = [cls for cls in classes if cls.__name__ not in _not_used_in_automl]

    return classes


def _rename_column_names_to_numeric(X):
    """Used in XGBoost classifier and regressor classes to rename column names
        when the input is a pd.DataFrame in case it has column names that contain symbols ([, ], <) that XGBoost cannot natively handle.

    Arguments:
        X (pd.DataFrame): the input training data of shape [n_samples, n_features]

    Returns:
        Transformed X where column names are renamed to numerical values
    """
    name_to_col_num = dict((col, col_num) for col_num, col in enumerate(X.columns.values))
    return X.rename(columns=name_to_col_num, inplace=False)


def jupyter_check():
    """Get whether or not the code is being run in a Ipython environment (such as Jupyter Notebook or Jupyter Lab)

    Arguments:
        None

    Returns:
        Boolean: True if Ipython, False otherwise
    """
    try:
        ipy = import_or_raise("IPython")
        return ipy.core.getipython.get_ipython()
    except Exception:
        return False


def safe_repr(value):
    """Convert the given value into a string that can safely be used for repr

    Arguments:
        value: the item to convert

    Returns:
        String representation of the value
    """
    if isinstance(value, float):
        if pd.isna(value):
            return 'np.nan'
        if np.isinf(value):
            return f"float('{repr(value)}')"
    return repr(value)


def is_all_numeric(df):
    """Checks if the given DataFrame contains only numeric values

    Arguments:
        df (DataFrame): The DataFrame to check datatypes of

    Returns:
        True if all the DataFrame columns are numeric and are not missing any values, False otherwise
    """
    if df.isnull().any().any():
        return False
    for dtype in df.dtypes:
        if dtype not in numeric_and_boolean_dtypes:
            return False
    return True


def _convert_woodwork_types_wrapper(pd_data):
    """
    Converts a pandas data structure that may have extension or nullable dtypes to dtypes that numpy can understand and handle.

    Arguments:
        pd_data (pd.Series, pd.DataFrame, pd.ExtensionArray): Pandas data structure

    Returns:
        New pandas data structure (pd.DataFrame or pd.Series) with original data and dtypes that can be handled by numpy
    """
    nullable_to_numpy_mapping = {pd.Int64Dtype: 'int64',
                                 pd.BooleanDtype: 'bool',
                                 pd.StringDtype: 'object'}
    if isinstance(pd_data, pd.api.extensions.ExtensionArray):
        pd_data = pd.Series(pd_data)
    if isinstance(pd_data, pd.Series) and type(pd_data.dtype) in nullable_to_numpy_mapping:
        return pd_data.astype(nullable_to_numpy_mapping[type(pd_data.dtype)])
    if isinstance(pd_data, pd.DataFrame):
        for col_name, col in pd_data.iteritems():
            if type(col.dtype) in nullable_to_numpy_mapping:
                pd_data[col_name] = pd_data[col_name].astype(nullable_to_numpy_mapping[type(col.dtype)])
    return pd_data
