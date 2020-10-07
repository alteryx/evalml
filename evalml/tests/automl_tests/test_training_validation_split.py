import numpy as np
import pandas as pd

from evalml.automl import TrainingValidationSplit


def test_tvsplit_nsplits():
    assert TrainingValidationSplit().get_n_splits() == 1


def test_tvsplit_default():
    X = pd.DataFrame({'col1': np.arange(0, 10)})
    y = pd.Series(np.arange(100, 110), name='target')
    splitter = TrainingValidationSplit()
    splits = splitter.split(X, y=y)
    assert len(splits) == 1 and len(splits[0]) == 2
    # sklearn train_test_split will do a 75/25 split by default
    pd.testing.assert_index_equal(splits[0][0], pd.Int64Index([0, 1, 2, 3, 4, 5, 6], dtype='int64'))
    pd.testing.assert_index_equal(splits[0][1], pd.Int64Index([7, 8, 9], dtype='int64'))


def test_tvsplit_size():
    X = pd.DataFrame({'col1': np.arange(0, 10)})
    y = pd.Series(np.arange(100, 110), name='target')
    splitter = TrainingValidationSplit(test_size=0.2, train_size=0.3)
    splits = splitter.split(X, y=y)
    assert len(splits) == 1 and len(splits[0]) == 2
    pd.testing.assert_index_equal(splits[0][0], pd.Int64Index([0, 1, 2], dtype='int64'))
    pd.testing.assert_index_equal(splits[0][1], pd.Int64Index([3, 4], dtype='int64'))

    splitter = TrainingValidationSplit(test_size=2, train_size=3)
    splits = splitter.split(X, y=y)
    assert len(splits) == 1 and len(splits[0]) == 2
    pd.testing.assert_index_equal(splits[0][0], pd.Int64Index([0, 1, 2], dtype='int64'))
    pd.testing.assert_index_equal(splits[0][1], pd.Int64Index([3, 4], dtype='int64'))


def test_tvsplit_shuffle():
    X = pd.DataFrame({'col1': np.arange(0, 10)})
    y = pd.Series(np.arange(100, 110), name='target')
    splitter = TrainingValidationSplit(shuffle=True, random_state=0)
    splits = splitter.split(X, y=y)
    assert len(splits) == 1 and len(splits[0]) == 2
    pd.testing.assert_index_equal(splits[0][0], pd.Int64Index([9, 1, 6, 7, 3, 0, 5], dtype='int64'))
    pd.testing.assert_index_equal(splits[0][1], pd.Int64Index([2, 8, 4], dtype='int64'))


def test_tvsplit_stratify():
    X = pd.DataFrame({'col1': np.arange(0, 10)})
    y = pd.Series(np.arange(5).repeat(2), name='target')
    splitter = TrainingValidationSplit(train_size=5, test_size=5, shuffle=True, stratify=y, random_state=0)
    splits = splitter.split(X, y=y)
    assert len(splits) == 1 and len(splits[0]) == 2
    pd.testing.assert_index_equal(splits[0][0], pd.Int64Index([1, 4, 2, 8, 7], dtype='int64'))
    pd.testing.assert_index_equal(splits[0][1], pd.Int64Index([3, 6, 9, 0, 5], dtype='int64'))
