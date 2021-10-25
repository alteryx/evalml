import numpy as np
import pandas as pd
import pytest

from evalml.preprocessing.data_splitters import TrainingValidationSplit


def test_tvsplit_nsplits():
    assert TrainingValidationSplit().get_n_splits() == 1


def test_tvsplit_default():
    X = pd.DataFrame({"col1": np.arange(0, 10)})
    y = pd.Series(np.arange(100, 110), name="target")
    splitter = TrainingValidationSplit()
    splits = list(splitter.split(X, y=y))
    assert len(splits) == 1 and len(splits[0]) == 2
    # sklearn train_test_split will do a 75/25 split by default
    np.testing.assert_equal(splits[0][0], [0, 1, 2, 3, 4, 5, 6])
    np.testing.assert_equal(splits[0][1], [7, 8, 9])


def test_tvsplit_size():
    X = pd.DataFrame({"col1": np.arange(0, 10)})
    y = pd.Series(np.arange(100, 110), name="target")
    splitter = TrainingValidationSplit(test_size=0.2, train_size=0.3)
    splits = list(splitter.split(X, y=y))
    assert len(splits) == 1 and len(splits[0]) == 2
    np.testing.assert_equal(splits[0][0], [0, 1, 2])
    np.testing.assert_equal(splits[0][1], [3, 4])

    splitter = TrainingValidationSplit(test_size=2, train_size=3)
    splits = list(splitter.split(X, y=y))
    assert len(splits) == 1 and len(splits[0]) == 2
    np.testing.assert_equal(splits[0][0], [0, 1, 2])
    np.testing.assert_equal(splits[0][1], [3, 4])


def test_tvsplit_shuffle():
    X = pd.DataFrame({"col1": np.arange(0, 10)})
    y = pd.Series(np.arange(100, 110), name="target")
    splitter = TrainingValidationSplit(shuffle=True, random_seed=0)
    splits = list(splitter.split(X, y=y))
    assert len(splits) == 1 and len(splits[0]) == 2
    np.testing.assert_equal(splits[0][0], [9, 1, 6, 7, 3, 0, 5])
    np.testing.assert_equal(splits[0][1], [2, 8, 4])


def test_tvsplit_stratify():
    X = pd.DataFrame({"col1": np.arange(0, 10)})
    y = pd.Series(np.arange(5).repeat(2), name="target")
    splitter = TrainingValidationSplit(
        train_size=5, test_size=5, shuffle=True, stratify=y, random_seed=0
    )
    splits = list(splitter.split(X, y=y))
    assert len(splits) == 1 and len(splits[0]) == 2
    np.testing.assert_equal(splits[0][0], [1, 4, 2, 8, 7])
    np.testing.assert_equal(splits[0][1], [3, 6, 9, 0, 5])


@pytest.mark.parametrize("random_seed", [0, 11, 57, 99, 1000])
def test_tvsplit_always_within_bounds_with_custom_index(random_seed):
    N = 11000
    X = pd.DataFrame({"col1": np.arange(0, N)}, index=np.arange(20000, 20000 + N))
    splitter = TrainingValidationSplit(
        train_size=0.75, shuffle=True, random_seed=random_seed
    )
    splits = list(splitter.split(X, y=None))
    assert np.all(np.logical_and(splits[0][0] < N, splits[0][0] >= 0))
    assert np.all(np.logical_and(splits[0][1] < N, splits[0][1] >= 0))
