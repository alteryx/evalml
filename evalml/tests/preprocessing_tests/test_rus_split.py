import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold, train_test_split

from evalml.preprocessing.data_splitters import (
    RandomUnderSamplerCVSplit,
    RandomUnderSamplerTVSplit
)

im = pytest.importorskip('imblearn.under_sampling', reason='Skipping plotting test because imblearn not installed')


def test_rus_nsplits():
    assert RandomUnderSamplerTVSplit().get_n_splits() == 1
    assert RandomUnderSamplerCVSplit().get_n_splits() == 3
    assert RandomUnderSamplerCVSplit(n_splits=5).get_n_splits() == 5


@pytest.mark.parametrize('data_type', ['np', 'pd', 'ww'])
@pytest.mark.parametrize('dataset', [0, 1])
def test_rus_tv_default(data_type, make_data_type, dataset, X_y_binary, X_y_multi):
    if dataset == 0:
        X, y = X_y_binary
    else:
        X, y = X_y_multi

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    rus = im.RandomUnderSampler(random_state=0)
    X_resample, y_resample = rus.fit_resample(X_train, y_train)
    initial_results = [(X_resample, y_resample), (X_test, y_test)]
    X_transform, y_transform = rus.fit_resample(X, y)

    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)
    russplit = RandomUnderSamplerTVSplit(random_state=0, test_size=0.2)
    for i, j in enumerate(russplit.split(X, y)):
        for idx, tup in enumerate(j):
            for jdx, val in enumerate(tup):
                if jdx == 1:
                    np.testing.assert_equal(val.to_series().values, initial_results[idx][jdx])
                else:
                    np.testing.assert_equal(val.to_dataframe().values, initial_results[idx][jdx])

    X_russplit, y_russplit = russplit.transform(X, y)
    np.testing.assert_equal(X_transform, X_russplit.to_dataframe().values)
    np.testing.assert_equal(y_transform, y_russplit.to_series().values)


@pytest.mark.parametrize('data_type', ['np', 'pd', 'ww'])
@pytest.mark.parametrize('dataset', [0, 1])
def test_rus_cv_default(data_type, make_data_type, dataset, X_y_binary, X_y_multi):
    if dataset == 0:
        X, y = X_y_binary
    else:
        X, y = X_y_multi
    skf = StratifiedKFold(shuffle=True, n_splits=3, random_state=0)
    rus = im.RandomUnderSampler(random_state=0)
    initial_results = []
    for i, (train_indices, test_indices) in enumerate(skf.split(X, y)):
        X_resample, y_resample = rus.fit_resample(X[train_indices], y[train_indices])
        initial_results.append([(X_resample, y_resample), (X[test_indices], y[test_indices])])
    X_transform, y_transform = rus.fit_resample(X, y)

    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)
    russplit = RandomUnderSamplerCVSplit(random_state=0)
    for i, j in enumerate(russplit.split(X, y)):
        for idx, tup in enumerate(j):
            for jdx, val in enumerate(tup):
                if jdx == 1:
                    np.testing.assert_equal(val.to_series().values, initial_results[i][idx][jdx])
                else:
                    np.testing.assert_equal(val.to_dataframe().values, initial_results[i][idx][jdx])

    X_russplit, y_russplit = russplit.transform(X, y)
    np.testing.assert_equal(X_transform, X_russplit.to_dataframe().values)
    np.testing.assert_equal(y_transform, y_russplit.to_series().values)
