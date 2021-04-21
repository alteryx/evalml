import numpy as np
import pandas as pd
import pytest
import woodwork as ww

from evalml.utils import _convert_numeric_dataset_pandas, infer_feature_types


def test_infer_feature_types():
    X_dt = pd.DataFrame([[1, 2], [3, 4]])
    X_dt.ww.init()
    pd.testing.assert_frame_equal(X_dt, infer_feature_types(X_dt))

    X_dc = ww.init_series(pd.Series([1, 2, 3, 4]))
    pd.testing.assert_series_equal(X_dc, infer_feature_types(X_dc))

    X_pd = pd.DataFrame({0: pd.Series([1, 2], dtype="int64"),
                         1: pd.Series([3, 4], dtype="int64")})
    pd.testing.assert_frame_equal(X_pd, infer_feature_types(X_pd))

    X_pd = pd.Series([1, 2, 3, 4], dtype="int64")
    pd.testing.assert_series_equal(X_pd, infer_feature_types(X_pd))

    X_list = [1, 2, 3, 4]
    X_expected = ww.init_series(pd.Series(X_list))
    pd.testing.assert_series_equal(X_expected, infer_feature_types(X_list))
    assert X_list == [1, 2, 3, 4]

    X_np = np.array([1, 2, 3, 4])
    X_expected = ww.init_series(pd.Series(X_np))
    pd.testing.assert_series_equal(X_expected, infer_feature_types(X_np))
    assert np.array_equal(X_np, np.array([1, 2, 3, 4]))

    X_np = np.array([[1, 2], [3, 4]])
    X_expected = pd.DataFrame(X_np)
    X_expected.ww.init()
    pd.testing.assert_frame_equal(X_expected, infer_feature_types(X_np))
    assert np.array_equal(X_np, np.array([[1, 2], [3, 4]]))


def test_infer_feature_types_series_name():
    name = "column with name"
    X_pd = pd.Series([1, 2, 3, 4], dtype="int64", name=name)
    X_dc = infer_feature_types(X_pd)
    assert X_dc.name == name
    pd.testing.assert_series_equal(X_pd, X_dc)


def test_infer_feature_types_dataframe():
    X_pd = pd.DataFrame({0: pd.Series([1, 2]),
                         1: pd.Series([3, 4])})
    pd.testing.assert_frame_equal(X_pd, infer_feature_types(X_pd), check_dtype=False)

    X_pd = pd.DataFrame({0: pd.Series([1, 2], dtype="int64"),
                         1: pd.Series([3, 4], dtype="int64")})
    pd.testing.assert_frame_equal(X_pd, infer_feature_types(X_pd))

    X_expected = X_pd.copy()
    X_expected[0] = X_expected[0].astype("category")
    pd.testing.assert_frame_equal(X_expected, infer_feature_types(X_pd, {0: "categorical"}))
    pd.testing.assert_frame_equal(X_expected, infer_feature_types(X_pd, {0: ww.logical_types.Categorical}))


def test_infer_feature_types_series():
    X_pd = pd.Series([1, 2, 3, 4])
    X_expected = X_pd.astype("int64")
    pd.testing.assert_series_equal(X_expected, infer_feature_types(X_pd))

    X_pd = pd.Series([1, 2, 3, 4], dtype="int64")
    pd.testing.assert_series_equal(X_pd, infer_feature_types(X_pd))

    X_pd = pd.Series([1, 2, 3, 4], dtype="int64")
    X_expected = X_pd.astype("category")
    pd.testing.assert_series_equal(X_expected, infer_feature_types(X_pd, "categorical"))

    X_pd = pd.Series([1, 2, 3, 4], dtype="int64")
    X_expected = X_pd.astype("category")
    pd.testing.assert_series_equal(X_expected, infer_feature_types(X_pd, ww.logical_types.Categorical))


@pytest.mark.parametrize("value,error",
                         [
                             (1, False), (-1, False),
                             (2.3, False), (None, True),
                             (np.nan, True), ("hello", True)
                         ])
@pytest.mark.parametrize("datatype", ["np", "pd", "ww"])
def test_convert_numeric_dataset_pandas(datatype, value, error, make_data_type):
    if datatype == "np" and value == "hello":
        pytest.skip("Unsupported configuration")

    X = pd.DataFrame([[1, 2, 3, 4], [2, value, 4, value]])
    y = pd.Series([0, 1])
    X = make_data_type(datatype, X)
    y = make_data_type(datatype, y)

    if error:
        with pytest.raises(ValueError, match="Values not all numeric or there are null"):
            _convert_numeric_dataset_pandas(X, y)
    else:
        X_transformed, y_transformed = _convert_numeric_dataset_pandas(X, y)
        X_ww = infer_feature_types(X)
        y_ww = infer_feature_types(y)

        pd.testing.assert_frame_equal(X_ww, X_transformed)
        pd.testing.assert_series_equal(y_ww, y_transformed)


def test_infer_feature_types_value_error():

    df = pd.DataFrame({"a": pd.Series([1, 2, 3]),
                       "b": pd.Series([4, 5, 6]),
                       "c": pd.Series([True, False, True])})
    df.ww.init(logical_types={"a": "IntegerNullable", "c": "BooleanNullable"})
    with pytest.raises(ValueError):
        infer_feature_types(df)
