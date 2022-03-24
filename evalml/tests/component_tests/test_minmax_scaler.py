import pandas as pd

from evalml.pipelines.components import MinMaxScaler


def test_minmax_scaler_numeric_only(X_y_binary):
    X, y = X_y_binary
    scaler = MinMaxScaler()

    scaler.fit(X, y)
    X_t = scaler.transform(X)
    for col in X_t.columns:
        assert min(X_t[col]) >= 0.0
        assert max(X_t[col]) <= 1.0

    scaler = MinMaxScaler(feature_range=(-1, 3))
    X_t = scaler.fit_transform(X, y)
    for col in X_t.columns:
        assert -1 <= min(X_t[col]) < 0
        assert 1 < max(X_t[col]) <= 3.1


def test_minmax_scaler_numeric_and_categorical():
    X = pd.DataFrame(
        {
            "col_1": [1, -2, 5, 8, 0],
            "col_2": ["a", "b", "a", "c", "b"],
            "col_3": [0.2, 3.0, 4.7, 2.2, 0.9],
        }
    )
    scaler = MinMaxScaler()
    scaler.fit(X)
    X_t = scaler.transform(X)

    assert min(X_t["col_1"]) >= 0
    assert max(X_t["col_1"]) <= 1
    pd.testing.assert_series_equal(X_t["col_2"], X["col_2"], check_dtype=False)
    assert min(X_t["col_3"]) >= 0
    assert max(X_t["col_3"]) <= 1


def test_minmax_scaler_categorical_only():
    X = pd.DataFrame(
        {
            "col_1": ["a", "a", "b", "b", "a"],
            "col_2": ["a", "b", "a", "c", "b"],
            "col_3": [False, True, True, False, True],
        }
    )
    scaler = MinMaxScaler()
    scaler.fit(X)
    X_t = scaler.transform(X)

    pd.testing.assert_frame_equal(X, X_t, check_dtype=False)
