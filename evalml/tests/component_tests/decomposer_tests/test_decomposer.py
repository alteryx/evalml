from datetime import datetime

import numpy as np
import pandas as pd

from evalml.pipelines.components.transformers.preprocessing.polynomial_decomposer import (
    PolynomialDecomposer,
)


def test_set_time_index():
    x = np.arange(0, 2 * np.pi, 0.01)
    dts = pd.date_range(datetime.today(), periods=len(x))
    X = pd.DataFrame({"x": x})
    X = X.set_index(dts)
    y = pd.Series(np.sin(x))

    assert isinstance(y.index, pd.RangeIndex)

    # Use the PolynomialDecomposer since we can't use a Decomposer class as it
    # has abstract methods.
    decomposer = PolynomialDecomposer()
    y_time_index = decomposer._set_time_index(X, y)
    assert isinstance(y_time_index.index, pd.DatetimeIndex)
