import numpy as np
import pandas as pd
import pytest
import woodwork as ww
from pandas.testing import assert_frame_equal
from woodwork.logical_types import Double, Integer

from evalml.pipelines.components import PCA


@pytest.mark.parametrize('data_type', ['pd', 'ww'])
def test_pca_numeric(data_type, make_data_type):
    X = pd.DataFrame([[3, 0, 1, 6],
                      [1, 2, 1, 6],
                      [10, 2, 1, 6],
                      [10, 2, 2, 5],
                      [6, 2, 2, 5]])
    X = make_data_type(data_type, X)
    pca = PCA()
    expected_X_t = pd.DataFrame([[3.176246, 1.282616],
                                 [4.969987, -0.702976],
                                 [-3.954182, 0.429071],
                                 [-4.079174, -0.252790],
                                 [-0.112877, -0.755922]],
                                columns=[f"component_{i}" for i in range(2)])
    X_t = pca.fit_transform(X)
    assert_frame_equal(expected_X_t, X_t.to_dataframe())


def test_pca_array():
    X = np.array([[3, 0, 1, 6],
                  [1, 2, 1, 6],
                  [10, 2, 1, 6],
                  [10, 2, 2, 5],
                  [6, 2, 2, 5]])
    pca = PCA()
    expected_X_t = pd.DataFrame([[3.176246, 1.282616],
                                 [4.969987, -0.702976],
                                 [-3.954182, 0.429071],
                                 [-4.079174, -0.252790],
                                 [-0.112877, -0.755922]],
                                columns=[f"component_{i}" for i in range(2)])
    pca.fit(X)
    X_t = pca.transform(X)
    assert_frame_equal(expected_X_t, X_t.to_dataframe())


def test_pca_invalid():
    X = pd.DataFrame([[3, 0, 1, 6],
                      [1, None, 1, 6],
                      [10, 2, 1, 6],
                      [10, 2, 2, np.nan],
                      [None, 2, 2, 5]])
    pca = PCA()
    with pytest.raises(ValueError, match="must be all numeric"):
        pca.fit(X)

    X = pd.DataFrame([[3, 0, 1, 6],
                      ['a', 'b', 'a', 'b'],
                      [10, 2, 1, 6],
                      [10, 2, 2, 23],
                      [0, 2, 2, 5]])
    pca = PCA()
    with pytest.raises(ValueError, match="must be all numeric"):
        pca.fit_transform(X)

    X_ok = pd.DataFrame([[3, 0, 1, 6],
                         [1, 2, 1, 6],
                         [10, 2, 1, 6],
                         [10, 2, 2, 5],
                         [6, 2, 2, 5]])
    pca = PCA()
    pca.fit(X_ok)
    with pytest.raises(ValueError, match="must be all numeric"):
        pca.transform(X)


def test_variance():
    X = pd.DataFrame([[3, 0, 1, 6, 5, 10],
                      [1, 2, 1, 3, 11, 4],
                      [10, 2, 1, 12, 5, 6],
                      [10, 6, 4, 4, 0, 1],
                      [6, 8, 9, 3, 1, 5]])
    pca = PCA(variance=0.97)
    expected_X_t = pd.DataFrame([[-5.581732, 0.469307, 3.985657, 1.760273],
                                 [-6.961064, -5.026062, -3.170519, -0.624576],
                                 [-1.352624, 7.778657, -0.778879, -1.554429],
                                 [7.067179, 0.645894, -2.633617, 2.159135],
                                 [6.828241, -3.867796, 2.597358, -1.740404]],
                                columns=[f"component_{i}" for i in range(4)])
    X_t_90 = pca.fit_transform(X)
    assert_frame_equal(expected_X_t, X_t_90.to_dataframe())

    pca = PCA(variance=0.75)
    X_t_75 = pca.fit_transform(X)
    assert X_t_75.shape[1] < X_t_90.shape[1]

    pca = PCA(variance=0.50)
    X_t_50 = pca.fit_transform(X)
    assert X_t_50.shape[1] < X_t_75.shape[1]


def test_n_components():
    X = pd.DataFrame([[3, 0, 1, 6, 5, 10],
                      [1, 2, 1, 3, 11, 4],
                      [10, 2, 1, 12, 5, 6],
                      [10, 6, 4, 4, 0, 1],
                      [6, 8, 9, 3, 1, 5]])
    pca = PCA(n_components=5)
    X_t = pca.fit_transform(X)
    assert X_t.shape[1] == 5

    pca = PCA(n_components=3)
    X_t = pca.fit_transform(X)
    assert X_t.shape[1] == 3

    pca = PCA(n_components=1)
    X_t = pca.fit_transform(X)
    assert X_t.shape[1] == 1


@pytest.mark.parametrize("X_df", [pd.DataFrame(pd.Series([1, 2, 3], dtype="Int64")),
                                  pd.DataFrame(pd.Series([1., 2., 3.], dtype="float")),
                                  pd.DataFrame(pd.Series([True, False, True], dtype="boolean"))])
def test_pca_woodwork_custom_overrides_returned_by_components(X_df):
    y = pd.Series([1, 2, 1])
    override_types = [Integer, Double]
    for logical_type in override_types:
        X = ww.DataTable(X_df, logical_types={0: logical_type})
        pca = PCA(n_components=1)
        pca.fit(X)
        transformed = pca.transform(X, y)
        assert isinstance(transformed, ww.DataTable)
        assert transformed.logical_types == {'component_0': ww.logical_types.Double}
