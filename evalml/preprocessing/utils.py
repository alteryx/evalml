import pandas as pd
from dask import dataframe as dd
from sklearn.model_selection import StratifiedShuffleSplit


def load_data(path, index, label, drop=None, verbose=True, **kwargs):
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
