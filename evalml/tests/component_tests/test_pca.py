import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from evalml.pipelines.components import PCA


def test_pca_numeric():
    X = pd.DataFrame([[3, 0, 1, 6],
                      [1, 2, 1, 6],
                      [10, 2, 1, 6],
                      [10, 2, 2, 5],
                      [6, 2, 2, 5]])
    pca = PCA()
    expected_X_t = pd.DataFrame([[3.176246, 1.282616],
                                 [4.969987, -0.702976],
                                 [-3.954182, 0.429071],
                                 [-4.079174, -0.252790],
                                 [-0.112877, -0.755922]])
    X_t = pd.DataFrame(pca.fit_transform(X))
    assert_frame_equal(X_t, expected_X_t)


def test_pca_array():
    X = [[3, 0, 1, 6],
         [1, 2, 1, 6],
         [10, 2, 1, 6],
         [10, 2, 2, 5],
         [6, 2, 2, 5]]
    pca = PCA()
    expected_X_t = pd.DataFrame([[3.176246, 1.282616],
                                 [4.969987, -0.702976],
                                 [-3.954182, 0.429071],
                                 [-4.079174, -0.252790],
                                 [-0.112877, -0.755922]])
    X_t = pd.DataFrame(pca.fit_transform(X))
    assert_frame_equal(X_t, expected_X_t)


def test_pca_invalid():
    X = pd.DataFrame([[3, 0, 1, 6],
                      [1, None, 1, 6],
                      [10, 2, 1, 6],
                      [10, 2, 2, np.nan],
                      [None, 2, 2, 5]])
    pca = PCA()
    with pytest.raises(ValueError, match="must be numeric"):
        pca.fit(X)

    X = pd.DataFrame([[3, 0, 1, 6],
                      ['a', 'b', 'a', 'b'],
                      [10, 2, 1, 6],
                      [10, 2, 2, 23],
                      [0, 2, 2, 5]])
    pca = PCA()
    with pytest.raises(ValueError, match="must be numeric"):
        pca.fit(X)
