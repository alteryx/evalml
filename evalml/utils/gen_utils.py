import pandas as pd
from pandas.api.types import is_numeric_dtype

from evalml.utils import Logger


def summarize_table(X):
    """
    Prints a table-level summary for a DataFrame, including information about number of rows and columns by datatype

    Arguments:
        X (pd.DataFrame)
    
    TODO: doctest
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    logger = Logger()
    logger.log_title("Summary for table:")

    logger.log("Number of rows: {}".format(X.shape[0]))
    logger.log("Number of columns: {}".format(X.shape[1]))
    logger.log("Total size of DataFrame: {} bytes".format(X.memory_usage(index=True).sum()))

    logger.log_subtitle("Number of columns by data type:")
    counts = X.dtypes.value_counts()
    for dtype in counts.index:
        logger.log('{}: {}'.format(dtype, counts[dtype]))


def summarize_col(X, col):
    """
    Prints a summary about a column in a DataFrame.
    If the column is numeric, prints min, max, mean, and std.
    If the column is categorical, prints number of unique values and most frequent value.

    Arguments:
        X (pd.DataFrame)
        col (int): index of column in DataFrame

    TODO: doctest
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    X_col = X.iloc[:,col]

    logger = Logger()
    logger.log_title("Summary for column {}".format(col))

    logger.log("Datatype of col: {}".format(X_col.dtype))
    logger.log("Number of non-NaN elements in col {}: {}".format(col, X.count(axis=0)[col]))
    logger.log("Total size of col: {} bytes".format(X.memory_usage(index=True)[col]))

    if is_numeric_dtype(X_col):
        logger.log_subtitle("Statistics for numerical column:")
        logger.log("min: {}".format(X_col.min()))
        logger.log("max: {}".format(X_col.max()))
        logger.log("mean: {}".format(X_col.mean()))
        logger.log("std: {}".format(X_col.std()))

    if pd.api.types.is_categorical_dtype(X_col):
        logger.log_subtitle("Statistics for categorical column:")
        logger.log("number of unique values: {}".format(X_col.nunique()))
        logger.log("most common value: {}".format(X_col.value_counts().idxmax()))
