import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from evalml.preprocessing.data_splitters import (
    BalancedClassificationDataCVSplit,
    BalancedClassificationDataTVSplit,
    BalancedClassificationSampler,
    TrainingValidationSplit
)


@pytest.mark.parametrize("splitter",
                         [BalancedClassificationDataCVSplit,
                          BalancedClassificationDataTVSplit])
def test_data_splitter_params(splitter):
    bcs = splitter()
    assert bcs.sampling_ratio == 0.25
    assert bcs.min_samples == 100

    bcs = splitter(sampling_ratio=0.3, min_samples=1, min_percentage=0.5)
    assert bcs.sampling_ratio == 0.3
    assert bcs.min_samples == 1
    assert bcs.min_percentage == 0.5


@pytest.mark.parametrize("splitter",
                         [BalancedClassificationDataCVSplit,
                          BalancedClassificationDataTVSplit])
def test_data_splitter_nsplits(splitter):
    if "TV" in splitter.__name__:
        assert splitter().get_n_splits() == 1
    else:
        assert splitter().get_n_splits() == 3
        assert splitter(n_splits=5).get_n_splits() == 5


@pytest.mark.parametrize("value", [np.nan, "hello"])
@pytest.mark.parametrize("splitter",
                         [BalancedClassificationDataCVSplit,
                          BalancedClassificationDataTVSplit])
def test_data_splitter_no_error(splitter, value, X_y_binary):
    X, y = X_y_binary
    X = pd.DataFrame(X)
    y = pd.Series(y)
    X.iloc[0, :] = value
    data_split = splitter()
    # handles both TV and CV iterations
    next(data_split.split(X, y))
    data_split.transform_sample(X, y)


@pytest.mark.parametrize('balanced_splitter,data_splitter',
                         [
                             (BalancedClassificationDataTVSplit(sampling_ratio=1, min_samples=50, test_size=0.2, shuffle=True, random_seed=0),
                              TrainingValidationSplit(test_size=0.2, shuffle=True, random_seed=0)),
                             (BalancedClassificationDataCVSplit(sampling_ratio=1, min_samples=50, shuffle=True, n_splits=3, random_seed=0),
                              StratifiedKFold(shuffle=True, n_splits=3, random_state=0))
                         ])
@pytest.mark.parametrize('data_type', ['np', 'pd', 'ww'])
def test_data_splitters_data_type(data_type, balanced_splitter, data_splitter, make_data_type, X_y_binary):
    X, y = X_y_binary
    # make imbalanced
    X_extended = np.append(X, X, 0)
    y_extended = np.append(y, np.array([0] * len(y)), 0)
    sample_method = BalancedClassificationSampler(sampling_ratio=1, min_samples=50, random_seed=0)

    initial_results = []
    for i, (train_indices, test_indices) in enumerate(data_splitter.split(X_extended, y_extended)):
        new_train_indices = sample_method.fit_resample(X_extended[train_indices], y_extended[train_indices])
        initial_results.append([new_train_indices, test_indices])
    indices = sample_method.fit_resample(X_extended, y_extended)

    X_extended = make_data_type(data_type, X_extended)
    y_extended = make_data_type(data_type, y_extended)
    for i, (train, test) in enumerate(balanced_splitter.split(X_extended, y_extended)):  # for each split
        assert len(train) == len(initial_results[i][0])
        assert len(test) == len(initial_results[i][1])
        assert len(train) + len(test) < len(X_extended)

    final_indices = balanced_splitter.transform_sample(X_extended, y_extended)
    assert len(final_indices) == len(indices)
    assert len(final_indices) <= 50 * 2
    assert isinstance(final_indices, list)


@pytest.mark.parametrize('balanced_splitter,data_splitter',
                         [
                             (BalancedClassificationDataTVSplit(sampling_ratio=1, min_samples=50, test_size=0.2, shuffle=True, random_seed=0),
                              TrainingValidationSplit(test_size=0.2, shuffle=True, random_seed=0)),
                             (BalancedClassificationDataCVSplit(sampling_ratio=1, min_samples=50, shuffle=True, n_splits=3, random_seed=0),
                              StratifiedKFold(shuffle=True, n_splits=3, random_state=0))
                         ])
@pytest.mark.parametrize('dataset', ['binary', 'multiclass'])
def test_data_splitters_dataset(dataset, balanced_splitter, data_splitter, make_data_type, X_y_binary, X_y_multi):
    if dataset == 'binary':
        X, y = X_y_binary
    else:
        X, y = X_y_multi
    dataset = 0 if dataset == 'binary' else 1
    # make imbalanced
    X_extended = np.append(X, X, 0)
    y_extended = np.append(y, np.array([0] * len(y)), 0)
    sample_method = BalancedClassificationSampler(sampling_ratio=1, min_samples=50, random_seed=0)

    initial_results = []
    for i, (train_indices, test_indices) in enumerate(data_splitter.split(X_extended, y_extended)):
        new_train_indices = sample_method.fit_resample(X_extended[train_indices], y_extended[train_indices])
        initial_results.append([new_train_indices, test_indices])
    indices = sample_method.fit_resample(X_extended, y_extended)

    # change to woodwork
    X_extended = make_data_type("ww", X_extended)
    y_extended = make_data_type("ww", y_extended)
    for i, (train, test) in enumerate(balanced_splitter.split(X_extended, y_extended)):  # for each split
        assert len(train) == len(initial_results[i][0])
        assert len(test) == len(initial_results[i][1])
        assert len(train) + len(test) < len(X_extended)

    final_indices = balanced_splitter.transform_sample(X_extended, y_extended)
    assert len(final_indices) == len(indices)
    assert len(final_indices) <= 50 * (dataset + 2)
    assert isinstance(final_indices, list)


@pytest.mark.parametrize("splitters", [BalancedClassificationDataTVSplit, BalancedClassificationDataCVSplit])
def test_data_splitters_balanced(splitters, X_y_binary, X_y_multi):
    X, y = X_y_binary
    splitter = splitters()

    for i, (train_indices, test_indices) in enumerate(splitter.split(X, y)):
        assert len(train_indices) + len(test_indices) == len(X)

    X, y = X_y_multi
    for i, (train_indices, test_indices) in enumerate(splitter.split(X, y)):
        assert len(train_indices) + len(test_indices) == len(X)


@pytest.mark.parametrize("splitters", [BalancedClassificationDataTVSplit, BalancedClassificationDataCVSplit])
def test_data_splitters_severe_imbalanced(splitters, X_y_binary, X_y_multi):
    X, y = X_y_binary
    y[0] = 0
    y[1:] = 1
    splitter = splitters()

    for i, (train_indices, test_indices) in enumerate(splitter.split(X, y)):
        assert len(train_indices) + len(test_indices) == len(X)

    X, y = X_y_multi
    y[0] = 0
    y[1:50] = 1
    y[50:] = 2
    for i, (train_indices, test_indices) in enumerate(splitter.split(X, y)):
        assert len(train_indices) + len(test_indices) == len(X)


def test_data_splitters_imbalanced_binary_tv():
    X = pd.DataFrame({"a": [i for i in range(1000)],
                      "b": [i % 5 for i in range(1000)]})
    # make y a 9:1 class ratio
    y = pd.Series([0] * 100 + [1] * 900)
    splitter = BalancedClassificationDataTVSplit()

    for i, (train_indices, test_indices) in enumerate(splitter.split(X, y)):
        assert len(test_indices) == 250   # test_size defaults to 0.25
        # remaining data will still preserve 9:1 ratio, which we want to get to 4:1
        # we don't know the exact number since we don't stratify split
        assert len(train_indices) < 500
        # we can only test the balance of the train since the split isn't stratified
        y_balanced_train = y.iloc[train_indices]
        y_train_counts = y_balanced_train.value_counts(normalize=True)
        assert max(y_train_counts.values) == 0.8


def test_data_splitters_imbalanced_multiclass_tv():
    X = pd.DataFrame({"a": [i for i in range(1500)],
                      "b": [i % 5 for i in range(1500)]})
    # make y a 8:1:1 class ratio
    y = pd.Series([0] * 150 + [1] * 1200 + [2] * 150)
    splitter = BalancedClassificationDataTVSplit()

    for i, (train_indices, test_indices) in enumerate(splitter.split(X, y)):
        assert len(test_indices) == 375   # test_size defaults to 0.25
        # we don't know the exact number since we don't stratify split
        assert len(train_indices) < 1000
        # we can only test the balance of the train since the split isn't stratified
        y_balanced_train = y.iloc[train_indices]
        y_train_counts = y_balanced_train.value_counts(normalize=True)
        # assert the values are around 2/3 for the majority class
        assert max(y_train_counts.values) < 7 / 10
        assert max(y_train_counts.values) > 6 / 10


def test_data_splitters_imbalanced_binary_cv():
    X = pd.DataFrame({"a": [i for i in range(1200)],
                      "b": [i % 5 for i in range(1200)]})
    # make y a 9:1 class ratio
    y = pd.Series([0] * 120 + [1] * 1080)
    splitter = BalancedClassificationDataCVSplit()

    for i, (train_indices, test_indices) in enumerate(splitter.split(X, y)):
        assert len(test_indices) == 400
        # remaining data will still preserve 9:1 ratio, which we want to get to 4:1
        assert len(train_indices) == 400
        y_balanced_train = y.iloc[train_indices]
        y_train_counts = y_balanced_train.value_counts(normalize=True)
        assert max(y_train_counts.values) == 0.8
        assert y_train_counts[1] == 0.8
        y_test = y.iloc[test_indices]
        y_test_counts = y_test.value_counts(normalize=True)
        assert max(y_test_counts.values) == 0.9


def test_data_splitters_imbalanced_multiclass_cv():
    X = pd.DataFrame({"a": [i for i in range(1200)],
                      "b": [i % 5 for i in range(1200)]})
    # make y a 8:1:1 class ratio
    y = pd.Series([0] * 120 + [1] * 960 + [2] * 120)
    splitter = BalancedClassificationDataCVSplit()

    for i, (train_indices, test_indices) in enumerate(splitter.split(X, y)):
        assert len(test_indices) == 400
        assert len(train_indices) == 480
        y_balanced_train = y.iloc[train_indices]
        y_train_counts = y_balanced_train.value_counts(normalize=True)
        assert max(y_train_counts.values) == 2 / 3
        y_test = y.iloc[test_indices]
        y_test_counts = y_test.value_counts(normalize=True)
        assert max(y_test_counts.values) == 0.8


def test_data_splitters_severe_imbalanced_binary_cv():
    X = pd.DataFrame({"a": [i for i in range(1200)],
                      "b": [i % 5 for i in range(1200)]})
    y = pd.Series([0] * 60 + [1] * 1140)
    splitter = BalancedClassificationDataCVSplit()

    for i, (train_indices, test_indices) in enumerate(splitter.split(X, y)):
        assert len(test_indices) == 400
        assert len(train_indices) == 800
        y_balanced_train = y.iloc[train_indices]
        y_train_counts = y_balanced_train.value_counts(normalize=True)
        assert max(y_train_counts.values) == 1140 / 1200
        y_test = y.iloc[test_indices]
        y_test_counts = y_test.value_counts(normalize=True)
        assert max(y_test_counts.values) == 1140 / 1200

    # no longer severe imbalance, we will resample
    splitter = BalancedClassificationDataCVSplit(min_percentage=0.001)

    for i, (train_indices, test_indices) in enumerate(splitter.split(X, y)):
        assert len(test_indices) == 400
        assert len(train_indices) == 200
        y_balanced_train = y.iloc[train_indices]
        y_train_counts = y_balanced_train.value_counts(normalize=True)
        assert max(y_train_counts.values) == 0.8
        y_test = y.iloc[test_indices]
        y_test_counts = y_test.value_counts(normalize=True)
        assert max(y_test_counts.values) == 1140 / 1200
