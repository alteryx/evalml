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


def detect_id_columns(X, threshold=1.0):
    """Check if any of the features are ID columns.

    Currently only performs these simple checks:
        - column name is "id"
        - column name ends in "_id"
        - column contains all unique values

    Args:
        X (pd.DataFrame): The input features to check
        threshold (float): the probability threshold to be considered an ID column. Defaults to 1.0

    Returns:
        A dictionary of features with column name or index and their corresponding probability
    """
    id_cols = {}
    cols_named_id = (X.columns[X.columns.str.match('id', case=False)])  # columns whose name is "id"
    id_cols.update([(col, 0.95) for col in cols_named_id])

    check_all_unique = (X.nunique() == len(X))
    cols_with_all_unique = check_all_unique[check_all_unique].index.tolist()  # columns whose values are all unique
    id_cols.update([(col, 1.0) if col in id_cols else (col, 0.95) for col in cols_with_all_unique])

    col_ends_with_id = (X.columns[X.columns.str.lower().str.endswith('_id')])  # columns whose name ends with "_id"
    id_cols.update([(col, 1.0) if col in id_cols else (col, 0.95) for col in col_ends_with_id])

    id_cols_above_threshold = {key: value for key, value in id_cols.items() if value >= threshold}

    return id_cols_above_threshold
