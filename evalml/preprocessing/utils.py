"""Helpful preprocessing utilities."""
import pandas as pd
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

from evalml.preprocessing.data_splitters import TrainingValidationSplit
from evalml.problem_types import (
    is_classification,
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


def split_data(
    X, y, problem_type, problem_configuration=None, test_size=0.2, random_seed=0
):
    """Split data into train and test sets.

    Args:
        X (pd.DataFrame or np.ndarray): data of shape [n_samples, n_features]
        y (pd.Series, or np.ndarray): target data of length [n_samples]
        problem_type (str or ProblemTypes): type of supervised learning problem. see evalml.problem_types.problemtype.all_problem_types for a full list.
        problem_configuration (dict): Additional parameters needed to configure the search. For example,
            in time series problems, values should be passed in for the date_index, gap, and max_delay variables.
        test_size (float): What percentage of data points should be included in the test set. Defaults to 0.2 (20%).
        random_seed (int): Seed for the random number generator. Defaults to 0.

    Returns:
        pd.DataFrame, pd.DataFrame, pd.Series, pd.Series: Feature and target data each split into train and test sets.
    """
    X = infer_feature_types(X)
    y = infer_feature_types(y)

    data_splitter = None
    if is_time_series(problem_type):
        data_splitter = TrainingValidationSplit(
            test_size=test_size, shuffle=False, stratify=None, random_seed=random_seed
        )
    elif is_regression(problem_type):
        data_splitter = ShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_seed
        )
    elif is_classification(problem_type):
        data_splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_seed
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
    """
    distribution = targets.value_counts() / len(targets)
    return distribution.mul(100).apply("{:.2f}%".format).rename_axis("Targets")
