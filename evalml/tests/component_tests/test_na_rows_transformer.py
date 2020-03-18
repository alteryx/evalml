import numpy as np
import pandas as pd

from evalml.pipelines.components import DropNaNRowsTransformer


def test_fit():
    y = pd.Series([1, 0, 1, np.nan])
    X = pd.DataFrame()
    X["a"] = ['a', 'b', 'c', 'd']
    X["b"] = [1, 2, 3, 0]
    X["c"] = [np.nan, 0, 0, 0]
    X["d"] = [0, 0, 0, 0]

    # test not passing in y
    transformer = DropNaNRowsTransformer()
    X_t = transformer.fit_transform(X)
    assert len(X_t) == 3

    X_t = transformer.transform(X)
    assert len(X_t) == 3

    transformer = DropNaNRowsTransformer()
    X_t = transformer.fit_transform(X, y)
    assert len(X_t) == 2

    X_t = transformer.transform(X, y)
    assert len(X_t) == 2
