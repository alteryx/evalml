import numpy as np
import pandas as pd

from evalml.pipelines.components import SimpleImputer


def test_fit():
    X = pd.DataFrame([[np.nan, 0, 1, np.nan],
                      [1, 2, 3, 2],
                      [1, 2, 3, 0]])
    # test impute_strategy
    transformer = SimpleImputer(impute_strategy='mean')
    X_expected_arr = pd.DataFrame([[1.0, 0, 1, 1.0],
                                   [1.0, 2, 3, 2.0],
                                   [1.0, 2, 3, 0.0]])
    X_t = transformer.fit_transform(X)
    assert X_t.equals(X_expected_arr)

    X_t = transformer.transform(X)
    assert X_t.equals(X_expected_arr)

    # test impute strategy is constant and fill value is not specified
    X = pd.DataFrame([[np.nan, 0, 1, np.nan],
                      ["a", 2, np.nan, 3],
                      ["b", 2, 3, 0]])

    transformer = SimpleImputer(impute_strategy='constant', fill_value=3)
    X_expected_arr = pd.DataFrame([[3, 0, 1.0, 3.0],
                                   ["a", 2, 3.0, 3.0],
                                   ["b", 2, 3.0, 0.0]])
    X_t = transformer.fit_transform(X)
    assert X_t.equals(X_expected_arr)

    X_t = transformer.transform(X)
    assert X_t.equals(X_expected_arr)
