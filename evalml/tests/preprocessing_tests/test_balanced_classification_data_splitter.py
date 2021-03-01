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
                             (BalancedClassificationDataTVSplit(balanced_ratio=1, min_samples=50, test_size=0.2, shuffle=True, random_seed=0),
                              TrainingValidationSplit(test_size=0.2, shuffle=True, random_state=0)),
                             (BalancedClassificationDataCVSplit(balanced_ratio=1, min_samples=50, shuffle=True, n_splits=3, random_seed=0),
                              StratifiedKFold(shuffle=True, n_splits=3, random_state=0))
                         ])
@pytest.mark.parametrize('data_type', ['np', 'pd', 'ww'])
@pytest.mark.parametrize('dataset', ['binary', 'multiclass'])
def test_data_splitters_default(data_type, dataset, balanced_splitter, data_splitter, make_data_type, X_y_binary, X_y_multi):
    if dataset == 'binary':
        X, y = X_y_binary
    else:
        X, y = X_y_multi
    dataset = 0 if dataset == 'binary' else 1
    # make imbalanced
    X_extended = np.append(X, X, 0)
    y_extended = np.append(y, np.array([0] * len(y)), 0)
    sample_method = BalancedClassificationSampler(balanced_ratio=1, min_samples=50, random_seed=0)

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
    assert len(final_indices) <= 50 * (dataset + 2)
    assert isinstance(final_indices, list)
