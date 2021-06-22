import numpy as np
import pandas as pd
import pytest
import woodwork as ww
from pandas.testing import assert_frame_equal
from woodwork.logical_types import Double, Integer

from evalml.pipelines.components import LinearDiscriminantAnalysis


def test_lda_invalid_init():
    with pytest.raises(
        ValueError,
        match="Invalid number of compponents for Linear Discriminant Analysis",
    ):
        LinearDiscriminantAnalysis(n_components=-1)


@pytest.mark.parametrize("data_type", ["pd", "ww"])
def test_lda_numeric(data_type, make_data_type):
    X = pd.DataFrame(
        [[3, 0, 1, 6], [1, 2, 1, 6], [10, 2, 1, 6], [10, 2, 2, 5], [6, 2, 2, 5]]
    )
    y = pd.Series([0, 1, 0, 1, 1])
    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)
    lda = LinearDiscriminantAnalysis()
    expected_X_t = pd.DataFrame(
        [
            [-3.7498560857993817],
            [1.984459921694517],
            [-3.234411950294312],
            [1.3401547523131798],
            [3.659653362085993],
        ],
        columns=["component_0"],
    )
    X_t = lda.fit_transform(X, y)
    assert_frame_equal(expected_X_t, X_t)


def test_lda_array():
    X = np.array(
        [[3, 0, 1, 6], [1, 2, 1, 6], [10, 2, 1, 6], [10, 2, 2, 5], [6, 2, 2, 5]]
    )
    y = np.array([2, 2, 0, 1, 0])
    lda = LinearDiscriminantAnalysis()
    expected_X_t = pd.DataFrame(
        [
            [-0.6412164311777084, 0.5197032695565076],
            [0.9499648898073094, -0.6919658287324498],
            [0.7364892645407753, 0.884637532109161],
            [-0.570057889422197, -0.005831184057363141],
            [-0.4751798337481819, -0.7065437888758568],
        ],
        columns=[f"component_{i}" for i in range(2)],
    )
    lda.fit(X, y)
    X_t = lda.transform(X)
    assert_frame_equal(expected_X_t, X_t)


def test_lda_invalid():
    X = pd.DataFrame(
        [
            [3, 0, 1, 6],
            [1, None, 1, 6],
            [10, 2, 1, 6],
            [10, 2, 2, np.nan],
            [None, 2, 2, 5],
        ]
    )
    y = [2, 0, 1, 1, 0]
    lda = LinearDiscriminantAnalysis()
    with pytest.raises(ValueError, match="must be all numeric"):
        lda.fit(X, y)

    X = pd.DataFrame(
        [
            [3, 0, 1, 6],
            ["a", "b", "a", "b"],
            [10, 2, 1, 6],
            [10, 2, 2, 23],
            [0, 2, 2, 5],
        ]
    )
    lda = LinearDiscriminantAnalysis()
    with pytest.raises(ValueError, match="must be all numeric"):
        lda.fit_transform(X, y)

    X_ok = pd.DataFrame(
        [[3, 0, 1, 6], [1, 2, 1, 6], [10, 2, 1, 6], [10, 2, 2, 5], [6, 2, 2, 5]]
    )
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_ok, y)
    with pytest.raises(ValueError, match="must be all numeric"):
        lda.transform(X)


def test_n_components():
    X = pd.DataFrame(
        [
            [3, 0, 1, 6, 5, 10],
            [1, 3, 1, 3, 11, 4],
            [10, 2, 3, 12, 5, 6],
            [10, 6, 4, 3, 0, 1],
            [6, 8, 9, 3, 3, 5],
            [3, 2, 1, 2, 1, 3],
            [12, 11, 1, 1, 3, 3],
        ]
    )
    y = [0, 3, 3, 1, 2, 0, 2]

    lda = LinearDiscriminantAnalysis(n_components=3)
    X_t = lda.fit_transform(X, y)
    assert X_t.shape[1] == 3

    lda = LinearDiscriminantAnalysis(n_components=1)
    X_t = lda.fit_transform(X, y)
    assert X_t.shape[1] == 1


def test_invalid_n_components():
    X = pd.DataFrame(
        [
            [3, 0, 1, 6, 5, 10],
            [1, 3, 1, 3, 11, 4],
            [10, 2, 3, 12, 5, 6],
            [10, 6, 4, 3, 0, 1],
            [6, 8, 9, 3, 3, 5],
            [3, 2, 1, 2, 1, 3],
            [12, 11, 1, 1, 3, 3],
        ]
    )
    y = [0, 1, 2, 1, 2, 0, 2]
    lda_invalid = LinearDiscriminantAnalysis(n_components=4)
    with pytest.raises(ValueError, match="is too large"):
        lda_invalid.fit(X, y)

    X = pd.DataFrame(
        [
            [3, 0, 1],
            [1, 3, 1],
            [10, 2, 3],
            [10, 6, 4],
            [6, 8, 9],
            [3, 2, 1],
            [12, 11, 1],
        ]
    )
    y = [0, 1, 2, 3, 4, 3, 4, 5]
    lda_invalid = LinearDiscriminantAnalysis(n_components=4)
    with pytest.raises(ValueError, match="is too large"):
        lda_invalid.fit(X, y)


def test_lda_woodwork_custom_overrides_returned_by_components():
    X_df = pd.DataFrame(
        [[3, 0, 1, 6], [1, 2, 1, 6], [10, 2, 1, 6], [10, 2, 2, 5], [6, 2, 2, 5]]
    )
    y = pd.Series([0, 1, 0, 1, 1])
    override_types = [Integer, Double]
    for logical_type in override_types:
        X_df.ww.init(
            logical_types={
                0: logical_type,
                1: logical_type,
                2: logical_type,
                3: logical_type,
            }
        )
        lda = LinearDiscriminantAnalysis(n_components=1)
        lda.fit(X_df, y)
        transformed = lda.transform(X_df, y)
        assert isinstance(transformed, pd.DataFrame)
        assert {k: type(v) for k, v in transformed.ww.logical_types.items()} == {
            "component_0": ww.logical_types.Double
        }
