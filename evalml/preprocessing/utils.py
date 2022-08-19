"""Helpful preprocessing utilities."""
import pandas as pd
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

from evalml.preprocessing.data_splitters import TrainingValidationSplit
from evalml.problem_types import is_classification, is_regression, is_time_series
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


def split_data(
    X,
    y,
    problem_type,
    problem_configuration=None,
    test_size=None,
    random_seed=0,
):
    """Split data into train and test sets.

    Args:
        X (pd.DataFrame or np.ndarray): data of shape [n_samples, n_features]
        y (pd.Series, or np.ndarray): target data of length [n_samples]
        problem_type (str or ProblemTypes): type of supervised learning problem. see evalml.problem_types.problemtype.all_problem_types for a full list.
        problem_configuration (dict): Additional parameters needed to configure the search. For example,
            in time series problems, values should be passed in for the time_index, gap, and max_delay variables.
        test_size (float): What percentage of data points should be included in the test set. Defaults to 0.2 (20%) for non-timeseries problems and 0.1
            (10%) for timeseries problems.
        random_seed (int): Seed for the random number generator. Defaults to 0.

    Returns:
        pd.DataFrame, pd.DataFrame, pd.Series, pd.Series: Feature and target data each split into train and test sets.

    Examples:
        >>> X = pd.DataFrame([1, 2, 3, 4, 5, 6], columns=["First"])
        >>> y = pd.Series([8, 9, 10, 11, 12, 13])
        ...
        >>> X_train, X_validation, y_train, y_validation = split_data(X, y, "regression", random_seed=42)
        >>> X_train
           First
        5      6
        2      3
        4      5
        3      4
        >>> X_validation
           First
        0      1
        1      2
        >>> y_train
        5    13
        2    10
        4    12
        3    11
        dtype: int64
        >>> y_validation
        0    8
        1    9
        dtype: int64
    """
    X = infer_feature_types(X)
    y = infer_feature_types(y)

    data_splitter = None
    if is_time_series(problem_type):
        if test_size is None:
            test_size = 0.1
            if (
                problem_configuration is not None
                and "forecast_horizon" in problem_configuration
            ):
                fh_pct = problem_configuration["forecast_horizon"] / len(X)
                test_size = max(test_size, fh_pct)
        data_splitter = TrainingValidationSplit(
            test_size=test_size,
            shuffle=False,
            stratify=None,
            random_seed=random_seed,
        )
    else:
        if test_size is None:
            test_size = 0.2
        if is_regression(problem_type):
            data_splitter = ShuffleSplit(
                n_splits=1,
                test_size=test_size,
                random_state=random_seed,
            )
        elif is_classification(problem_type):
            data_splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=test_size,
                random_state=random_seed,
            )

    train, test = next(data_splitter.split(X, y))

    X_train = X.ww.iloc[train]
    X_test = X.ww.iloc[test]
    y_train = y.ww.iloc[train]
    y_test = y.ww.iloc[test]

    return X_train, X_test, y_train, y_test


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
