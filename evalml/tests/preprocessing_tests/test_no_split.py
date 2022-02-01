import numpy as np
import pandas as pd

from evalml.preprocessing.data_splitters import NoSplit


def test_nosplit_nsplits():
    assert NoSplit().get_n_splits() == 0
    assert not NoSplit().is_cv


def test_nosplit_default():
    X = pd.DataFrame({"col1": np.arange(0, 10)})
    y = pd.Series(np.arange(100, 110), name="target")
    splitter = NoSplit()
    splits = list(splitter.split(X, y=y))
    assert len(splits) == 1 and len(splits[0]) == 2

    np.testing.assert_equal(splits[0][0], np.arange(0, 10))
    assert len(splits[0][1]) == 0
