"""General utility methods for loading demo-related data sets."""
import pandas as pd

from evalml.utils import infer_feature_types


def load_data(path, index, target, n_rows=None, drop=None, verbose=True, **kwargs):
    """Load features and target from file.

    Args:
        path (str): Path to file or a http/ftp/s3 URL.
        index (str): Column for index.
        target (str): Column for target.
        n_rows (int): Number of rows to return. Defaults to None.
        drop (list): List of columns to drop. Defaults to None.
        verbose (bool): If True, prints information about features and target. Defaults to True.
        **kwargs: Other keyword arguments that should be passed to panda's `read_csv` method.

    Returns:
        pd.DataFrame, pd.Series: Features matrix and target.
    """
    feature_matrix = pd.read_csv(path, index_col=index, nrows=n_rows, **kwargs)

    targets = [target] + (drop or [])
    y = feature_matrix[target]
    X = feature_matrix.drop(columns=targets)

    if verbose:
        # number of features
        print(number_of_features(X.dtypes), end="\n\n")

        # number of total training examples
        info = "Number of training examples: {}"
        print(info.format(len(X)), end="\n")

        # target distribution
        print(target_distribution(y))

    return infer_feature_types(X), infer_feature_types(y)


def number_of_features(dtypes):
    """Get the number of features of each specific dtype in a DataFrame.

    Args:
        dtypes (pd.Series): DataFrame.dtypes to get the number of features for.

    Returns:
        pd.Series: dtypes and the number of features for each input type.

    Example:
        >>> X = pd.DataFrame()
        >>> X["integers"] = [i for i in range(10)]
        >>> X["floats"] = [float(i) for i in range(10)]
        >>> X["strings"] = [str(i) for i in range(10)]
        >>> X["booleans"] = [bool(i%2) for i in range(10)]

        Lists the number of columns corresponding to each dtype.

        >>> number_of_features(X.dtypes)
                     Number of Features
        Boolean                       1
        Categorical                   1
        Numeric                       2
    """
    dtype_to_vtype = {
        "bool": "Boolean",
        "int32": "Numeric",
        "int64": "Numeric",
        "float64": "Numeric",
        "object": "Categorical",
        "datetime64[ns]": "Datetime",
    }

    vtypes = dtypes.astype(str).map(dtype_to_vtype).value_counts()
    return vtypes.sort_index().to_frame("Number of Features")


def target_distribution(targets):
    """Get the target distributions.

    Args:
        targets (pd.Series): Target data.

    Returns:
        pd.Series: Target data and their frequency distribution as percentages.

    Examples:
        >>> y = pd.Series([1, 2, 4, 1, 3, 3, 1, 2])
        >>> target_distribution(y)
        Targets
        1    37.50%
        2    25.00%
        3    25.00%
        4    12.50%
        dtype: object
        >>> y = pd.Series([True, False, False, False, True])
        >>> target_distribution(y)
        Targets
        False    60.00%
        True     40.00%
        dtype: object
    """
    distribution = targets.value_counts() / len(targets)
    return distribution.mul(100).apply("{:.2f}%".format).rename_axis("Targets")
