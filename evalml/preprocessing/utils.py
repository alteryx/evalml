"""Helpful preprocessing utilities."""
import pandas as pd
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

from evalml.pipelines.utils import stack_data, stack_X, unstack_multiseries
from evalml.preprocessing.data_splitters import TrainingValidationSplit
from evalml.problem_types import (
    is_classification,
    is_multiseries,
    is_regression,
    is_time_series,
)
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


def split_multiseries_data(X, y, series_id, time_index, **kwargs):
    """Split stacked multiseries data into train and test sets. Unstacked data can use `split_data`.

    Args:
        X (pd.DataFrame): The input training data of shape [n_samples*n_series, n_features].
        y (pd.Series): The target training targets of length [n_samples*n_series].
        series_id (str): Name of column containing series id.
        time_index (str): Name of column containing time index.
        **kwargs: Additional keyword arguments to pass to the split_data function.

    Returns:
        pd.DataFrame, pd.DataFrame, pd.Series, pd.Series: Feature and target data each split into train and test sets.
    """
    X_unstacked, y_unstacked = unstack_multiseries(
        X,
        y,
        series_id,
        time_index,
        y.name,
    )
    (
        X_train_unstacked,
        X_holdout_unstacked,
        y_train_unstacked,
        y_holdout_unstacked,
    ) = split_data(
        X_unstacked, y_unstacked, problem_type="time series regression", **kwargs
    )

    # Get unique series value from X if there is only the time_index column
    # Otherwise, this information is generated in `stack_X` from the column values
    series_id_values = set(X[series_id]) if len(X_unstacked.columns) == 1 else None

    X_train = stack_X(
        X_train_unstacked,
        series_id,
        time_index,
        series_id_values=series_id_values,
    )
    X_holdout = stack_X(
        X_holdout_unstacked,
        series_id,
        time_index,
        starting_index=X_train.index[-1] + 1,
        series_id_values=series_id_values,
    )
    y_train = stack_data(y_train_unstacked)
    y_holdout = stack_data(y_holdout_unstacked, starting_index=y_train.index[-1] + 1)

    return X_train, X_holdout, y_train, y_holdout


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

    Raises:
        ValueError: If the problem_configuration is missing or does not contain both a time_index and series_id for multiseries problems.

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
    if is_multiseries(problem_type) and isinstance(y, pd.Series):
        if problem_configuration is None:
            raise ValueError(
                "split_data requires problem_configuration for multiseries problems",
            )
        series_id = problem_configuration.get("series_id")
        time_index = problem_configuration.get("time_index")
        if series_id is None or time_index is None:
            raise ValueError(
                "split_data needs both series_id and time_index values in the problem_configuration to split multiseries data",
            )
        return split_multiseries_data(
            X,
            y,
            series_id,
            time_index,
            problem_configuration=problem_configuration,
            test_size=test_size,
            random_seed=random_seed,
        )

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
        >>> print(target_distribution(y).to_string())
        Targets
        1    37.50%
        2    25.00%
        3    25.00%
        4    12.50%
        >>> y = pd.Series([True, False, False, False, True])
        >>> print(target_distribution(y).to_string())
        Targets
        False    60.00%
        True     40.00%
    """
    distribution = targets.value_counts() / len(targets)
    return distribution.mul(100).apply("{:.2f}%".format).rename_axis("Targets")
