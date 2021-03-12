import numpy as np
import pandas as pd
import pytest
import woodwork as ww

from evalml.utils import (
    _convert_numeric_dataset_pandas,
    _convert_woodwork_types_wrapper,
    infer_feature_types
)


def test_convert_woodwork_types_wrapper_with_nan():
    y = _convert_woodwork_types_wrapper(pd.Series([1, 2, None], dtype="Int64"))
    pd.testing.assert_series_equal(y, pd.Series([1, 2, np.nan], dtype="float64"))

    y = _convert_woodwork_types_wrapper(pd.array([1, 2, None], dtype="Int64"))
    pd.testing.assert_series_equal(y, pd.Series([1, 2, np.nan], dtype="float64"))

    y = _convert_woodwork_types_wrapper(pd.Series(["a", "b", None], dtype="string"))
    pd.testing.assert_series_equal(y, pd.Series(["a", "b", np.nan], dtype="object"))

    y = _convert_woodwork_types_wrapper(pd.array(["a", "b", None], dtype="string"))
    pd.testing.assert_series_equal(y, pd.Series(["a", "b", np.nan], dtype="object"))

    y = _convert_woodwork_types_wrapper(pd.Series([True, False, None], dtype="boolean"))
    pd.testing.assert_series_equal(y, pd.Series([True, False, np.nan]))

    y = _convert_woodwork_types_wrapper(pd.array([True, False, None], dtype="boolean"))
    pd.testing.assert_series_equal(y, pd.Series([True, False, np.nan]))


def test_convert_woodwork_types_wrapper():
    y = _convert_woodwork_types_wrapper(pd.Series([1, 2, 3], dtype="Int64"))
    pd.testing.assert_series_equal(y, pd.Series([1, 2, 3], dtype="int64"))

    y = _convert_woodwork_types_wrapper(pd.array([1, 2, 3], dtype="Int64"))
    pd.testing.assert_series_equal(y, pd.Series([1, 2, 3], dtype="int64"))

    y = _convert_woodwork_types_wrapper(pd.Series(["a", "b", "a"], dtype="string"))
    pd.testing.assert_series_equal(y, pd.Series(["a", "b", "a"], dtype="object"))

    y = _convert_woodwork_types_wrapper(pd.array(["a", "b", "a"], dtype="string"))
    pd.testing.assert_series_equal(y, pd.Series(["a", "b", "a"], dtype="object"))

    y = _convert_woodwork_types_wrapper(pd.Series([True, False, True], dtype="boolean"))
    pd.testing.assert_series_equal(y, pd.Series([True, False, True], dtype="bool"))

    y = _convert_woodwork_types_wrapper(pd.array([True, False, True], dtype="boolean"))
    pd.testing.assert_series_equal(y, pd.Series([True, False, True], dtype="bool"))


def test_convert_woodwork_types_wrapper_series_name():
    name = "my series name"
    series_with_name = pd.Series([1, 2, 3], name=name)
    y = _convert_woodwork_types_wrapper(series_with_name)
    assert y.name == name


def test_convert_woodwork_types_wrapper_dataframe():
    X = pd.DataFrame({"Int series": pd.Series([1, 2, 3], dtype="Int64"),
                      "Int array": pd.array([1, 2, 3], dtype="Int64"),
                      "Int series with nan": pd.Series([1, 2, None], dtype="Int64"),
                      "Int array with nan": pd.array([1, 2, None], dtype="Int64"),
                      "string series": pd.Series(["a", "b", "a"], dtype="string"),
                      "string array": pd.array(["a", "b", "a"], dtype="string"),
                      "string series with nan": pd.Series(["a", "b", None], dtype="string"),
                      "string array with nan": pd.array(["a", "b", None], dtype="string"),
                      "boolean series": pd.Series([True, False, True], dtype="boolean"),
                      "boolean array": pd.array([True, False, True], dtype="boolean"),
                      "boolean series with nan": pd.Series([True, False, None], dtype="boolean"),
                      "boolean array with nan": pd.array([True, False, None], dtype="boolean")
                      })
    X_expected = pd.DataFrame({"Int series": pd.Series([1, 2, 3], dtype="int64"),
                               "Int array": pd.array([1, 2, 3], dtype="int64"),
                               "Int series with nan": pd.Series([1, 2, np.nan], dtype="float64"),
                               "Int array with nan": pd.array([1, 2, np.nan], dtype="float64"),
                               "string series": pd.Series(["a", "b", "a"], dtype="object"),
                               "string array": pd.array(["a", "b", "a"], dtype="object"),
                               "string series with nan": pd.Series(["a", "b", np.nan], dtype="object"),
                               "string array with nan": pd.array(["a", "b", np.nan], dtype="object"),
                               "boolean series": pd.Series([True, False, True], dtype="bool"),
                               "boolean array": pd.array([True, False, True], dtype="bool"),
                               "boolean series with nan": pd.Series([True, False, np.nan], dtype="object"),
                               "boolean array with nan": pd.array([True, False, np.nan], dtype="object")
                               })
    pd.testing.assert_frame_equal(X_expected, _convert_woodwork_types_wrapper(X))


def testinfer_feature_types():
    X_dt = ww.DataTable(pd.DataFrame([[1, 2], [3, 4]]))
    pd.testing.assert_frame_equal(X_dt.to_dataframe(), infer_feature_types(X_dt).to_dataframe())

    X_dc = ww.DataColumn(pd.Series([1, 2, 3, 4]))
    pd.testing.assert_series_equal(X_dc.to_series(), infer_feature_types(X_dc).to_series())

    X_pd = pd.DataFrame({0: pd.Series([1, 2], dtype="Int64"),
                         1: pd.Series([3, 4], dtype="Int64")})
    pd.testing.assert_frame_equal(X_pd, infer_feature_types(X_pd).to_dataframe())

    X_pd = pd.Series([1, 2, 3, 4], dtype="Int64")
    pd.testing.assert_series_equal(X_pd, infer_feature_types(X_pd).to_series())

    X_list = [1, 2, 3, 4]
    X_expected = ww.DataColumn(pd.Series(X_list))
    pd.testing.assert_series_equal(X_expected.to_series(), infer_feature_types(X_list).to_series())
    assert X_list == [1, 2, 3, 4]

    X_np = np.array([1, 2, 3, 4])
    X_expected = ww.DataColumn(pd.Series(X_np))
    pd.testing.assert_series_equal(X_expected.to_series(), infer_feature_types(X_np).to_series())
    assert np.array_equal(X_np, np.array([1, 2, 3, 4]))

    X_np = np.array([[1, 2], [3, 4]])
    X_expected = ww.DataTable(pd.DataFrame(X_np))
    pd.testing.assert_frame_equal(X_expected.to_dataframe(), infer_feature_types(X_np).to_dataframe())
    assert np.array_equal(X_np, np.array([[1, 2], [3, 4]]))


def testinfer_feature_types_series_name():
    name = "column with name"
    X_pd = pd.Series([1, 2, 3, 4], dtype="Int64", name=name)
    X_dc = infer_feature_types(X_pd)
    assert X_dc.name == name
    pd.testing.assert_series_equal(X_pd, X_dc.to_series())


def test_infer_feature_types_dataframe():
    X_pd = pd.DataFrame({0: pd.Series([1, 2]),
                         1: pd.Series([3, 4])})
    pd.testing.assert_frame_equal(X_pd, infer_feature_types(X_pd).to_dataframe(), check_dtype=False)

    X_pd = pd.DataFrame({0: pd.Series([1, 2], dtype="Int64"),
                         1: pd.Series([3, 4], dtype="Int64")})
    pd.testing.assert_frame_equal(X_pd, infer_feature_types(X_pd).to_dataframe())

    X_expected = X_pd.copy()
    X_expected[0] = X_expected[0].astype("category")
    pd.testing.assert_frame_equal(X_expected, infer_feature_types(X_pd, {0: "categorical"}).to_dataframe())
    pd.testing.assert_frame_equal(X_expected, infer_feature_types(X_pd, {0: ww.logical_types.Categorical}).to_dataframe())


def test_infer_feature_types_series():
    X_pd = pd.Series([1, 2, 3, 4])
    X_expected = X_pd.astype("Int64")
    pd.testing.assert_series_equal(X_expected, infer_feature_types(X_pd).to_series())

    X_pd = pd.Series([1, 2, 3, 4], dtype="Int64")
    pd.testing.assert_series_equal(X_pd, infer_feature_types(X_pd).to_series())

    X_pd = pd.Series([1, 2, 3, 4], dtype="Int64")
    X_expected = X_pd.astype("category")
    pd.testing.assert_series_equal(X_expected, infer_feature_types(X_pd, "categorical").to_series())

    X_pd = pd.Series([1, 2, 3, 4], dtype="Int64")
    X_expected = X_pd.astype("category")
    pd.testing.assert_series_equal(X_expected, infer_feature_types(X_pd, ww.logical_types.Categorical).to_series())


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

        X_ww = _convert_woodwork_types_wrapper(X_ww.to_dataframe())
        y_ww = _convert_woodwork_types_wrapper(y_ww.to_series())
        pd.testing.assert_frame_equal(X_ww, X_transformed)
        pd.testing.assert_series_equal(y_ww, y_transformed)
