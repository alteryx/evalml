import pandas as pd
from dask import dataframe as dd
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit


def load_data(path, index, label, drop=None, verbose=True, **kwargs):
    """Load features and labels from file(s).

    Args:
        path (str) : path to file(s)
        index (str) : column for index
        label (str) : column for labels
        drop (list) : columns to drop
        verbose (bool) : whether to print information about features and labels

    Returns:
        DataFrame, Series : features and labels
    """
    if '*' in path:
        feature_matrix = dd.read_csv(path, **kwargs).set_index(index, sorted=True)

        labels = [label] + (drop or [])
        y = feature_matrix[label].compute()
        X = feature_matrix.drop(labels=labels, axis=1).compute()
    else:
        feature_matrix = pd.read_csv(path, index_col=index, **kwargs)

        labels = [label] + (drop or [])
        y = feature_matrix[label]
        X = feature_matrix.drop(columns=labels)

    if verbose:
        # number of features
        print(number_of_features(X.dtypes), end='\n\n')

        # number of training examples
        info = 'Number of training examples: {}'
        print(info.format(len(X)), end='\n\n')

        # label distribution
        print(label_distribution(y))

    return X, y


def split_data(X, y, regression=False, test_size=.2, random_state=None):
    """Splits data into train and test sets.

    Args:
        X (DataFrame) : features
        y (Series) : labels
        regression (bool): if true, do not use stratified split
        test_size (float) : percent of train set to holdout for testing
        random_state (int) : seed for the random number generator

    Returns:
        DataFrame, DataFrame, Series, Series : features and labels each split into train and test sets
    """
    if regression:
        CV_method = ShuffleSplit(n_splits=1,
                                 test_size=test_size,
                                 random_state=0)
    else:
        CV_method = StratifiedShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state)
    train, test = next(CV_method.split(X, y))
    X_train = X.iloc[train]
    X_test = X.iloc[test]
    y_train = y.iloc[train]
    y_test = y.iloc[test]
    return X_train, X_test, y_train, y_test


def number_of_features(dtypes):
    dtype_to_vtype = {
        'bool': 'Boolean',
        'int32': 'Numeric',
        'int64': 'Numeric',
        'float64': 'Numeric',
        'object': 'Categorical',
        'datetime64[ns]': 'Datetime',
    }

    vtypes = dtypes.astype(str).map(dtype_to_vtype).value_counts()
    return vtypes.sort_index().to_frame('Number of Features')


def label_distribution(labels):
    distribution = labels.value_counts() / len(labels)
    return distribution.mul(100).apply('{:.2f}%'.format).rename_axis('Labels')


def detect_label_leakage(X, y, threshold=.95):
    """Check if any of the features are highly correlated with the target.

    Currently only supports binary and numeric targets and features

    Args:
        X (pd.DataFrame): The input features to check
        y (pd.Series): the labels
        threshold (float): the correlation threshold to be considered leakage. Defaults to .95

    Returns:
        leakage, dictionary of features with leakage and corresponding threshold
    """

    # only select numeric
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'bool']
    X = X.select_dtypes(include=numerics)

    if len(X.columns) == 0:
        return {}

    corrs = X.corrwith(y).abs()
    out = corrs[corrs >= threshold]
    return out.to_dict()


def detect_highly_null(X, percent_threshold=.95):
    """ Checks if there are any highly-null columns in a dataframe.

    Args:
        X (DataFrame) : features
        percent_threshold(float): Require that percentage of non-null values to not be considered "highly-null", defaults to .95

    Returns:
        a set of features that are highly-null
    """
    threshold = len(X) * percent_threshold
    num_nonnan = X.count()
    filtered = num_nonnan[num_nonnan < threshold]
    return (set(filtered.index))
