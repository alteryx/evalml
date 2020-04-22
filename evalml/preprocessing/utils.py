import pandas as pd
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit


def load_data(path, index, label, n_rows=None, drop=None, verbose=True, **kwargs):
    """Load features and labels from file(s).

    Arguments:
        path (str): Path to file or a http/ftp/s3 URL
        index (str): Column for index
        label (str): Column for labels
        n_rows (int): Number of rows to return
        drop (list): List of columns to drop
        verbose (bool): If True, prints information about features and labels

    Returns:
        pd.DataFrame, pd.Series: features and labels
    """

    feature_matrix = pd.read_csv(path, index_col=index, nrows=n_rows, **kwargs)

    labels = [label] + (drop or [])
    y = feature_matrix[label]
    X = feature_matrix.drop(columns=labels)

    if verbose:
        # number of features
        print(number_of_features(X.dtypes), end='\n\n')

        # number of total training examples
        info = 'Number of training examples: {}'
        print(info.format(len(X)), end='\n')

        # label distribution
        print(label_distribution(y))

    return X, y


def split_data(X, y, regression=False, test_size=.2, random_state=None):
    """Splits data into train and test sets.

    Arguments:
        X (pd.DataFrame or np.array): data of shape [n_samples, n_features]
        y (pd.Series): labels of length [n_samples]
        regression (bool): if true, do not use stratified split
        test_size (float): percent of train set to holdout for testing
        random_state (int, np.random.RandomState): seed for the random number generator

    Returns:
        pd.DataFrame, pd.DataFrame, pd.Series, pd.Series: features and labels each split into train and test sets
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

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
    """Get the number of features for specific dtypes

    Arguments:

        dtypes (pd.Series): dtypes to get the number of features for

    Returns:
        pd.Series: dtypes and the number of features for each input type
    """
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
    """Get the label distributions
    Arguments:
        labels (pd.Series): Label values

    Returns:
        pd.Series: Label values and their frequency distribution as percentages.
    """
    distribution = labels.value_counts() / len(labels)
    return distribution.mul(100).apply('{:.2f}%'.format).rename_axis('Labels')


def drop_nan_target_rows(X, y):
    """Drops rows in X and y when row in the target y has a value of NaN.

    Arguments:
        X (pd.DataFrame): Data to transform
        y (pd.Series): Target values

    Returns:
        pd.DataFrame: Transformed X (and y, if passed in) with rows that had a NaN value removed.
    """
    X_t = X
    y_t = y

    if not isinstance(X_t, pd.DataFrame):
        X_t = pd.DataFrame(X_t)

    if not isinstance(y_t, pd.Series):
        y_t = pd.Series(y_t)

    # drop rows where corresponding y is NaN
    y_null_indices = y_t.index[y_t.isna()]
    X_t = X_t.drop(index=y_null_indices)
    y_t.dropna(inplace=True)
    return X_t, y_t
