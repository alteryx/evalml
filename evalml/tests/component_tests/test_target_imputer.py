import numpy as np
import pandas as pd
import pytest
import woodwork as ww
from pandas.testing import assert_series_equal
from woodwork.logical_types import (
    Boolean,
    Categorical,
    Double,
    Integer,
    NaturalLanguage
)

from evalml.pipelines.components import TargetImputer


def test_target_imputer_median():
    y = pd.Series([np.nan, 1, 10, 10, 6])
    imputer = TargetImputer(impute_strategy='median')
    y_expected = pd.Series([8, 1, 10, 10, 6])
    y_t = imputer.fit_transform(None, y)
    assert_series_equal(y_expected, y_t.to_series(), check_dtype=False)


def test_target_imputer_mean():
    y = pd.Series([np.nan, 2, 0])
    imputer = TargetImputer(impute_strategy='mean')
    y_expected = pd.Series([1, 2, 0])
    y_t = imputer.fit_transform(None, y)
    assert_series_equal(y_expected, y_t.to_series(), check_dtype=False)


def test_target_imputer_constant():
    y = pd.Series([np.nan, 0, 5])
    imputer = TargetImputer(impute_strategy='constant', fill_value=3)
    y_expected = pd.Series([3, 0, 5])
    y_t = imputer.fit_transform(None, y)
    assert_series_equal(y_expected, y_t.to_series(), check_dtype=False)

    y = pd.Series([np.nan, "a", "b"])
    imputer = TargetImputer(impute_strategy='constant', fill_value=3)
    y_expected = pd.Series([3, "a", "b"]).astype("category")
    y_t = imputer.fit_transform(None, y)
    assert_series_equal(y_expected, y_t.to_series(), check_dtype=False)

    # TODO
    # test impute strategy is constant and fill value is not specified


def test_target_imputer_most_frequent():
    y = pd.Series([np.nan, "a", "b"])
    imputer = TargetImputer(impute_strategy='most_frequent')
    y_expected = pd.Series(["a", "a", "b"]).astype("category")
    y_t = imputer.fit_transform(None, y)
    assert_series_equal(y_expected, y_t.to_series(), check_dtype=False)

    y = pd.Series([np.nan, 1, 1, 2])
    imputer = TargetImputer(impute_strategy='most_frequent')
    y_expected = pd.Series([1, 1, 1, 2])
    y_t = imputer.fit_transform(None, y)
    assert_series_equal(y_expected, y_t.to_series(), check_dtype=False)


def test_target_imputer_col_with_non_numeric_with_numeric_strategy():
    y = pd.Series([np.nan, "a", "b"])
    imputer = TargetImputer(impute_strategy='mean')
    with pytest.raises(ValueError, match="Cannot use mean strategy with non-numeric data"):
        imputer.fit_transform(None, y)
    with pytest.raises(ValueError, match="Cannot use mean strategy with non-numeric data"):
        imputer.fit(None, y)
    imputer = TargetImputer(impute_strategy='median')
    with pytest.raises(ValueError, match="Cannot use median strategy with non-numeric data"):
        imputer.fit_transform(None, y)
    with pytest.raises(ValueError, match="Cannot use median strategy with non-numeric data"):
        imputer.fit(None, y)


@pytest.mark.parametrize("data_type", ['pd', 'ww'])
def test_target_imputer_all_bool_return_original(data_type, make_data_type):
    y = pd.Series([True, True, False, True, True], dtype=bool)
    y = make_data_type(data_type, y)
    y_expected = pd.Series([True, True, False, True, True], dtype='boolean')
    imputer = TargetImputer()
    imputer.fit(None, y)
    y_t = imputer.transform(None, y)
    assert_series_equal(y_expected, y_t.to_series())


@pytest.mark.parametrize("data_type", ['pd', 'ww'])
def test_target_imputer_boolean_dtype(data_type, make_data_type):
    y = pd.Series([True, np.nan, False, np.nan, True], dtype='boolean')
    y_expected = pd.Series([True, True, False, True, True], dtype='boolean')
    y = make_data_type(data_type, y)
    imputer = TargetImputer()
    imputer.fit(None, y)
    y_t = imputer.transform(None, y)
    assert_series_equal(y_expected, y_t.to_series())


def test_target_imputer_fit_transform_all_nan_empty():
    y = pd.Series([np.nan, np.nan])
    imputer = TargetImputer()
    y_expected = pd.Series([])
    y_t = imputer.fit_transform (None, y)
    assert_series_equal(y_expected, y_t.to_series())


def test_target_imputer_numpy_input():
    y = np.array([np.nan, 0, 2])
    imputer = TargetImputer(impute_strategy='mean')
    y_expected = np.array([1, 0, 2])
    assert np.allclose(y_expected, imputer.fit_transform(None, y).to_series())
    np.testing.assert_almost_equal(y, np.array([np.nan, 0, 2]))


# @pytest.mark.parametrize("data_type", ["numeric", "categorical"])
# def test_target_imputer_fill_value(data_type):
#     if data_type == "numeric":
#         y = pd.DataFrame({
#             "some numeric": [np.nan, 1, 0],
#             "another numeric": [0, np.nan, 2]
#         })
#         fill_value = -1
#         expected = pd.DataFrame({
#             "some numeric": [-1, 1, 0],
#             "another numeric": [0, -1, 2]
#         })
#     else:
#         y = pd.DataFrame({
#             "categorical with nan": pd.Series([np.nan, "1", np.nan, "0", "3"], dtype='category'),
#             "object with nan": ["b", "b", np.nan, "c", np.nan]
#         })
#         fill_value = "fill"
#         expected = pd.DataFrame({
#             "categorical with nan": pd.Series(["fill", "1", "fill", "0", "3"], dtype='category'),
#             "object with nan": pd.Series(["b", "b", "fill", "c", "fill"], dtype='category'),
#         })
#     y = pd.Series([0, 0, 1, 0, 1])
#     imputer = TargetImputer(impute_strategy="constant", fill_value=fill_value)
#     imputer.fit(y, y)
#     transformed = imputer.transform(y, y)
#     assert_series_equal(expected, transformed.to_dataframe(), check_dtype=False)

#     imputer = TargetImputer(impute_strategy="constant", fill_value=fill_value)
#     transformed = imputer.fit_transform(y, y)
#     assert_series_equal(expected, transformed.to_dataframe(), check_dtype=False)


# def test_target_imputer_does_not_reset_index():
#     y = pd.DataFrame({'input_val': np.arange(10), 'target': np.arange(10)})
#     y.loc[5, 'input_val'] = np.nan
#     assert y.index.tolist() == list(range(10))

#     y.drop(0, inplace=True)
#     y = y.pop('target')
#     pd.testing.assert_series_equal(pd.DataFrame({'input_val': [1.0, 2, 3, 4, np.nan, 6, 7, 8, 9]},
#                                                dtype=float, index=list(range(1, 10))), y)

#     imputer = TargetImputer(impute_strategy="mean")
#     imputer.fit(y, y=y)
#     transformed = imputer.transform(None, y)
#     pd.testing.assert_series_equal(pd.DataFrame({'input_val': [1.0, 2, 3, 4, 5, 6, 7, 8, 9]},
#                                                dtype=float,
#                                                index=list(range(1, 10))),
#                                   transformed.to_dataframe())


@pytest.mark.parametrize("y, y_expected", [(pd.Series([1, 0, 5, None]), pd.Series([1, 0, 5, 2])),
                                           (pd.Series([0.1, 0.0, 0.5, None]), pd.Series([0.1, 0.0, 0.5, 0.2])),
                                           (pd.Series([None, None, None, None]), pd.Series([]))])
def test_target_imputer_with_none(y, y_expected):
    imputer = TargetImputer(impute_strategy="mean")
    y_t = imputer.fit_transform(None, y)
    assert_series_equal(y_expected, y_t.to_series(), check_dtype=False)


@pytest.mark.parametrize("y, y_expected", [(pd.Series([1, 0, 5, None]), pd.Series([1, 0, 5, 2])),
                                           (pd.Series([0.1, 0.0, 0.5, None]), pd.Series([0.1, 0.0, 0.5, 0.2])),
                                           (pd.Series(None, None, None, None), pd.Series([]))])
def test_target_imputer_with_none_non_numeric(y, y_expected):
    imputer = TargetImputer(impute_strategy="mean")
    y_t = imputer.fit_transform(None, y)
    assert_series_equal(y_expected, y_t.to_series(), check_dtype=False)

    y = pd.DataFrame({"category with None": pd.Series(["b", "a", "a", None], dtype='category'),
                      "boolean with None": pd.Series([True, None, False, True], dtype='boolean'),
                      "object with None": ["b", "a", "a", None],
                      "all None": [None, None, None, None]})
    y = pd.Series([0, 0, 1, 0, 1])
    imputer = TargetImputer()
    imputer.fit(y, y)
    transformed = imputer.transform(y, y)
    expected = pd.DataFrame({"category with None": pd.Series(["b", "a", "a", "a"], dtype='category'),
                             "boolean with None": pd.Series([True, True, False, True], dtype='boolean'),
                             "object with None": pd.Series(["b", "a", "a", "a"], dtype='category')})
    assert_series_equal(expected, transformed.to_dataframe(), check_dtype=False)


# @pytest.mark.parametrize("y_df", [pd.Series([1, 2, 3], dtype="Int64"),
#                                   pd.Series([1., 2., 3.], dtype="float"),
#                                   pd.Series(['a', 'b', 'a'], dtype="category"),
#                                   pd.Series([True, False, True], dtype="boolean"),
#                                   pd.Series(['this will be a natural language column because length', 'yay', 'hay'], dtype="string")])
# @pytest.mark.parametrize("has_nan", [True, False])
# @pytest.mark.parametrize("impute_strategy", ["mean", "median", "most_frequent"])
# def test_target_imputer_woodwork_custom_overrides_returned_by_components(y_df, has_nan, impute_strategy):
#     y = pd.Series([1, 2, 1])
#     if has_nan:
#         y_df[len(y_df) - 1] = np.nan
#     override_types = [Integer, Double, Categorical, NaturalLanguage, Boolean]
#     for logical_type in override_types:
#         try:
#             y = ww.DataColumn(y_df, logical_type=logical_type)
#         except TypeError:
#             continue

#         impute_strategy_to_use = impute_strategy
#         if logical_type in [NaturalLanguage, Categorical]:
#             impute_strategy_to_use = "most_frequent"

#         imputer = TargetImputer(impute_strategy=impute_strategy_to_use)
#         imputer.fit(None, y)
#         transformed = imputer.transform(None, y)
#         assert isinstance(transformed, ww.DataColumn)
#         if impute_strategy_to_use == "most_frequent" or not has_nan:
#             assert transformed.logical_type == logical_type
#         else:
#             assert transformed.logical_type == Double
