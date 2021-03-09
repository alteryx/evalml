import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold, train_test_split

from evalml.preprocessing.data_splitters import (
    KMeansSMOTECVSplit,
    KMeansSMOTETVSplit,
    RandomUnderSamplerCVSplit,
    RandomUnderSamplerTVSplit,
    SMOTENCCVSplit,
    SMOTENCTVSplit,
    SMOTETomekCVSplit,
    SMOTETomekTVSplit
)
from evalml.preprocessing.data_splitters.base_splitters import (
    BaseSamplingSplitter
)

im_os = pytest.importorskip('imblearn.over_sampling', reason='Skipping data splitter test because imblearn not installed')
im_com = pytest.importorskip('imblearn.combine', reason='Skipping data splitter test because imblearn not installed')
im_us = pytest.importorskip('imblearn.under_sampling', reason='Skipping data splitter test because imblearn not installed')


def test_empty_sampler_tv(X_y_binary):
    # ensure base splitters work with no sampler
    X, y = X_y_binary
    bss = BaseSamplingSplitter(sampler=None, split_type="TV", random_seed=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    for i, (train, test) in enumerate(bss.split(X, y)):
        np.testing.assert_equal(X_train, train[0])
        np.testing.assert_equal(y_train, train[1])
        np.testing.assert_equal(X_test, test[0])
        np.testing.assert_equal(y_test, test[1])

    X2, y2 = bss.transform_sample(X, y)
    np.testing.assert_equal(X2, X)
    np.testing.assert_equal(y2, y)


def test_empty_sampler_cv(X_y_binary):
    # ensure base splitters work with no sampler
    X, y = X_y_binary
    r = BaseSamplingSplitter(n_splits=3, shuffle=True, split_type="CV", random_seed=0)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

    for i in range(3):
        train, test = next(r.split(X, y))
        skf_train, skf_test = next(skf.split(X, y))
        X_train = X[skf_train]
        X_test = X[skf_test]
        y_train = y[skf_train]
        y_test = y[skf_test]
        np.testing.assert_equal(X_train, train[0])
        np.testing.assert_equal(y_train, train[1])
        np.testing.assert_equal(X_test, test[0])
        np.testing.assert_equal(y_test, test[1])

    X2, y2 = r.transform_sample(X, y)
    np.testing.assert_equal(X2, X)
    np.testing.assert_equal(y2, y)


@pytest.mark.parametrize("splitter",
                         [KMeansSMOTECVSplit, KMeansSMOTETVSplit,
                          SMOTETomekCVSplit, SMOTETomekTVSplit,
                          RandomUnderSamplerCVSplit, RandomUnderSamplerTVSplit,
                          SMOTENCCVSplit, SMOTENCTVSplit])
def test_data_splitter_smote_nsplits(splitter):
    args = {}
    if "SMOTENC" in splitter.__name__:
        args = {"categorical_features": [1]}
    if "TVSplit" in splitter.__name__:
        assert splitter(**args).get_n_splits() == 1
    else:
        assert splitter(**args).get_n_splits() == 3
        assert splitter(**args, n_splits=5).get_n_splits() == 5


def test_kmeans_kwargs():
    km = KMeansSMOTECVSplit(cluster_balance_threshold=0.01)
    assert km.sampler.cluster_balance_threshold == 0.01

    km = KMeansSMOTETVSplit(cluster_balance_threshold=0.01)
    assert km.sampler.cluster_balance_threshold == 0.01


@pytest.mark.parametrize("categorical_features,error", [(None, True),
                                                        ([], True),
                                                        ((0), True),
                                                        (1, True),
                                                        (True, True),
                                                        (False, True),
                                                        ([True, True], True),
                                                        ([True, False], False),
                                                        ([1, 2], False),
                                                        ([0], False),
                                                        ([1], False)])
@pytest.mark.parametrize("splitter", [SMOTENCCVSplit, SMOTENCTVSplit])
def test_smotenc_error(splitter, categorical_features, error):
    if error:
        with pytest.raises(ValueError, match="Categorical feature array must"):
            splitter(categorical_features=categorical_features)
    else:
        splitter(categorical_features=categorical_features)


@pytest.mark.parametrize("value", [np.nan, "hello"])
@pytest.mark.parametrize("splitter",
                         [KMeansSMOTECVSplit, KMeansSMOTETVSplit,
                          SMOTETomekCVSplit, SMOTETomekTVSplit,
                          RandomUnderSamplerCVSplit, RandomUnderSamplerTVSplit,
                          SMOTENCCVSplit, SMOTENCTVSplit])
def test_data_splitter_error(splitter, value, X_y_binary):
    X, y = X_y_binary
    X = pd.DataFrame(X)
    y = pd.Series(y)
    X.iloc[0, :] = value
    if "SMOTENC" not in splitter.__name__:
        data_split = splitter()
    else:
        data_split = splitter(categorical_features=[0])
    with pytest.raises(ValueError, match="Values not all numeric or there are null values"):
        # handles both TV and CV iterations
        next(data_split.split(X, y))
    with pytest.raises(ValueError, match="Values not all numeric or there are null values"):
        data_split.transform_sample(X, y)


@pytest.mark.parametrize('data_type', ['np', 'pd', 'ww'])
@pytest.mark.parametrize('dataset', [0, 1])
@pytest.mark.parametrize("splitter,sampler",
                         [(KMeansSMOTETVSplit, im_os.KMeansSMOTE),
                          (SMOTETomekTVSplit, im_com.SMOTETomek),
                          (RandomUnderSamplerTVSplit, im_us.RandomUnderSampler),
                          (SMOTENCTVSplit, im_os.SMOTENC)])
def test_data_splitter_tv_default(splitter, sampler, data_type, make_data_type, dataset, X_y_binary, X_y_multi):
    if dataset == 0:
        X, y = X_y_binary
    else:
        X, y = X_y_multi

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    if "SMOTENC" not in splitter.__name__:
        # sampler refers to the original data sampler strategy from the imblearn library,
        # while splitter refers to our data splitter object
        data_splitter = splitter(random_seed=0, test_size=0.2)
        sample_method = sampler(random_state=0)
    else:
        data_splitter = splitter(categorical_features=[0], random_seed=0, test_size=0.2)
        sample_method = sampler(categorical_features=[0], random_state=0)
        assert data_splitter.categorical_features == [0]
    X_resample, y_resample = sample_method.fit_resample(X_train, y_train)
    initial_results = [(X_resample, y_resample), (X_test, y_test)]
    X_transform, y_transform = sample_method.fit_resample(X, y)

    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)
    for i, j in enumerate(data_splitter.split(X, y)):
        for idx, tup in enumerate(j):  # for each (X, y) in split
            for jdx, val in enumerate(tup):  # for each array in (X, y) pair
                np.testing.assert_equal(val.values, initial_results[idx][jdx])

    X_data_split, y_data_split = data_splitter.transform_sample(X, y)
    np.testing.assert_equal(X_transform, X_data_split.values)
    np.testing.assert_equal(y_transform, y_data_split.values)


@pytest.mark.parametrize('data_type', ['np', 'pd', 'ww'])
@pytest.mark.parametrize('dataset', [0, 1])
@pytest.mark.parametrize("splitter,sampler",
                         [(KMeansSMOTECVSplit, im_os.KMeansSMOTE),
                          (SMOTETomekCVSplit, im_com.SMOTETomek),
                          (RandomUnderSamplerCVSplit, im_us.RandomUnderSampler),
                          (SMOTENCCVSplit, im_os.SMOTENC)])
def test_data_splitter_cv_default(splitter, sampler, data_type, make_data_type, dataset, X_y_binary, X_y_multi):
    if dataset == 0:
        X, y = X_y_binary
    else:
        X, y = X_y_multi
    skf = StratifiedKFold(shuffle=True, n_splits=3, random_state=0)
    if "SMOTENC" not in splitter.__name__:
        data_splitter = splitter(random_seed=0)
        sample_method = sampler(random_state=0)
    else:
        data_splitter = splitter(categorical_features=[0], random_seed=0)
        sample_method = sampler(categorical_features=[0], random_state=0)
        assert data_splitter.categorical_features == [0]
    initial_results = []
    for i, (train_indices, test_indices) in enumerate(skf.split(X, y)):
        X_resample, y_resample = sample_method.fit_resample(X[train_indices], y[train_indices])
        initial_results.append([(X_resample, y_resample), (X[test_indices], y[test_indices])])
    X_transform, y_transform = sample_method.fit_resample(X, y)

    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)
    for i, j in enumerate(data_splitter.split(X, y)):  # for each split
        for idx, tup in enumerate(j):  # for each (X, y) pair
            for jdx, val in enumerate(tup):  # for each array in (X, y)
                np.testing.assert_equal(val.values, initial_results[i][idx][jdx])

    X_data_split, y_data_split = data_splitter.transform_sample(X, y)
    np.testing.assert_equal(X_transform, X_data_split.values)
    np.testing.assert_equal(y_transform, y_data_split.values)
