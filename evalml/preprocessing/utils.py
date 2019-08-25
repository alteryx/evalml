import pandas as pd
from dask import dataframe as dd
from sklearn.model_selection import StratifiedShuffleSplit


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


def split_data(x, y, test_size=.2, random_state=None):
    """Splits data into train and test sets.

    Args:
        x (DataFrame) : features
        y (Series) : labels
        holdout (float) : percent to holdout from train set for testing
        random_state (int) : seed for the random number generator

    Returns:
        DataFrame, DataFrame, Series, Series : features and labels each split into train and test sets
    """
    stratified = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train, test = next(stratified.split(x, y))
    x_train = x.loc[x.index[train]]
    x_test = x.loc[x.index[test]]
    y_train = y.loc[y.index[train]]
    y_test = y.loc[y.index[test]]
    return x_train, x_test, y_train, y_test


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
