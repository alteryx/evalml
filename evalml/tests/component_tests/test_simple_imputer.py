import numpy as np
import pandas as pd
import pytest
import woodwork as ww
from pandas.testing import assert_frame_equal
from woodwork.logical_types import (
    Boolean,
    Categorical,
    Double,
    Integer,
    NaturalLanguage
)

from evalml.pipelines.components import SimpleImputer


def test_simple_imputer_median():
    X = pd.DataFrame([[np.nan, 0, 1, np.nan],
                      [1, 2, 3, 2],
                      [10, 2, np.nan, 2],
                      [10, 2, 5, np.nan],
                      [6, 2, 7, 0]])
    transformer = SimpleImputer(impute_strategy='median')
    X_expected_arr = pd.DataFrame([[8, 0, 1, 2],
                                   [1, 2, 3, 2],
                                   [10, 2, 4, 2],
                                   [10, 2, 5, 2],
                                   [6, 2, 7, 0]])
    X_t = transformer.fit_transform(X)
    assert_frame_equal(X_expected_arr, X_t.to_dataframe(), check_dtype=False)


def test_simple_imputer_mean():
    X = pd.DataFrame([[np.nan, 0, 1, np.nan],
                      [1, 2, 3, 2],
                      [1, 2, 3, 0]])
    # test impute_strategy
    transformer = SimpleImputer(impute_strategy='mean')
    X_expected_arr = pd.DataFrame([[1, 0, 1, 1],
                                   [1, 2, 3, 2],
                                   [1, 2, 3, 0]])
    X_t = transformer.fit_transform(X)
    assert_frame_equal(X_expected_arr, X_t.to_dataframe(), check_dtype=False)


def test_simple_imputer_constant():
    # test impute strategy is constant and fill value is not specified
    X = pd.DataFrame([[np.nan, 0, 1, np.nan],
                      ["a", 2, np.nan, 3],
                      ["b", 2, 3, 0]])

    transformer = SimpleImputer(impute_strategy='constant', fill_value=3)
    X_expected_arr = pd.DataFrame([[3, 0, 1, 3],
                                   ["a", 2, 3, 3],
                                   ["b", 2, 3, 0]])
    X_expected_arr = X_expected_arr.astype({0: 'category'})
    X_t = transformer.fit_transform(X)
    assert_frame_equal(X_expected_arr, X_t.to_dataframe(), check_dtype=False)


def test_simple_imputer_most_frequent():
    X = pd.DataFrame([[np.nan, 0, 1, np.nan],
                      ["a", 2, np.nan, 3],
                      ["b", 2, 1, 0]])

    transformer = SimpleImputer(impute_strategy='most_frequent')
    X_expected_arr = pd.DataFrame([["a", 0, 1, 0],
                                   ["a", 2, 1, 3],
                                   ["b", 2, 1, 0]])
    X_expected_arr = X_expected_arr.astype({0: 'category'})
    X_t = transformer.fit_transform(X)
    assert_frame_equal(X_expected_arr, X_t.to_dataframe(), check_dtype=False)


def test_simple_imputer_col_with_non_numeric():
    # test col with all strings
    X = pd.DataFrame([["a", 0, 1, np.nan],
                      ["b", 2, 3, 3],
                      ["a", 2, 3, 1],
                      [np.nan, 2, 3, 0]])

    transformer = SimpleImputer(impute_strategy='mean')
    with pytest.raises(ValueError, match="Cannot use mean strategy with non-numeric data"):
        transformer.fit_transform(X)
    with pytest.raises(ValueError, match="Cannot use mean strategy with non-numeric data"):
        transformer.fit(X)

    transformer = SimpleImputer(impute_strategy='median')
    with pytest.raises(ValueError, match="Cannot use median strategy with non-numeric data"):
        transformer.fit_transform(X)
    with pytest.raises(ValueError, match="Cannot use median strategy with non-numeric data"):
        transformer.fit(X)

    transformer = SimpleImputer(impute_strategy='most_frequent')
    X_expected_arr = pd.DataFrame([["a", 0, 1, 0],
                                   ["b", 2, 3, 3],
                                   ["a", 2, 3, 1],
                                   ["a", 2, 3, 0]])
    X_expected_arr = X_expected_arr.astype({0: 'category'})
    X_t = transformer.fit_transform(X)
    assert_frame_equal(X_expected_arr, X_t.to_dataframe(), check_dtype=False)

    transformer = SimpleImputer(impute_strategy='constant', fill_value=2)
    X_expected_arr = pd.DataFrame([["a", 0, 1, 2],
                                   ["b", 2, 3, 3],
                                   ["a", 2, 3, 1],
                                   [2, 2, 3, 0]])
    X_expected_arr = X_expected_arr.astype({0: 'category'})
    X_t = transformer.fit_transform(X)
    assert_frame_equal(X_expected_arr, X_t.to_dataframe(), check_dtype=False)


@pytest.mark.parametrize("data_type", ['pd', 'ww'])
def test_simple_imputer_all_bool_return_original(data_type, make_data_type):
    X = pd.DataFrame([True, True, False, True, True], dtype=bool)
    y = pd.Series([1, 0, 0, 1, 0])
    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)
    X_expected_arr = pd.DataFrame([True, True, False, True, True], dtype='boolean')
    imputer = SimpleImputer()
    imputer.fit(X, y)
    X_t = imputer.transform(X)
    assert_frame_equal(X_expected_arr, X_t.to_dataframe())


@pytest.mark.parametrize("data_type", ['pd', 'ww'])
def test_simple_imputer_boolean_dtype(data_type, make_data_type):
    X = pd.DataFrame([True, np.nan, False, np.nan, True], dtype='boolean')
    y = pd.Series([1, 0, 0, 1, 0])
    X_expected_arr = pd.DataFrame([True, True, False, True, True], dtype='boolean')
    X = make_data_type(data_type, X)
    imputer = SimpleImputer()
    imputer.fit(X, y)
    X_t = imputer.transform(X)
    assert_frame_equal(X_expected_arr, X_t.to_dataframe())


@pytest.mark.parametrize("data_type", ['pd', 'ww'])
def test_simple_imputer_multitype_with_one_bool(data_type, make_data_type):
    X_multi = pd.DataFrame({
        "bool with nan": pd.Series([True, np.nan, False, np.nan, False], dtype='boolean'),
        "bool no nan": pd.Series([False, False, False, False, True], dtype=bool),
    })
    y = pd.Series([1, 0, 0, 1, 0])
    X_multi_expected_arr = pd.DataFrame({
        "bool with nan": pd.Series([True, False, False, False, False], dtype='boolean'),
        "bool no nan": pd.Series([False, False, False, False, True], dtype='boolean'),
    })
    X_multi = make_data_type(data_type, X_multi)

    imputer = SimpleImputer()
    imputer.fit(X_multi, y)
    X_multi_t = imputer.transform(X_multi)
    assert_frame_equal(X_multi_expected_arr, X_multi_t.to_dataframe())


def test_simple_imputer_fit_transform_drop_all_nan_columns():
    X = pd.DataFrame({"all_nan": [np.nan, np.nan, np.nan],
                      "some_nan": [np.nan, 1, 0],
                      "another_col": [0, 1, 2]})

    transformer = SimpleImputer(impute_strategy='most_frequent')
    X_expected_arr = pd.DataFrame({"some_nan": [0, 1, 0], "another_col": [0, 1, 2]})
    X_t = transformer.fit_transform(X)
    assert_frame_equal(X_expected_arr, X_t.to_dataframe(), check_dtype=False)
    assert_frame_equal(X, pd.DataFrame({"all_nan": [np.nan, np.nan, np.nan],
                                        "some_nan": [np.nan, 1, 0],
                                        "another_col": [0, 1, 2]}))


def test_simple_imputer_transform_drop_all_nan_columns():
    X = pd.DataFrame({"all_nan": [np.nan, np.nan, np.nan],
                      "some_nan": [np.nan, 1, 0],
                      "another_col": [0, 1, 2]})
    transformer = SimpleImputer(impute_strategy='most_frequent')
    transformer.fit(X)
    X_expected_arr = pd.DataFrame({"some_nan": [0, 1, 0], "another_col": [0, 1, 2]})
    assert_frame_equal(X_expected_arr, transformer.transform(X).to_dataframe(), check_dtype=False)
    assert_frame_equal(X, pd.DataFrame({"all_nan": [np.nan, np.nan, np.nan],
                                        "some_nan": [np.nan, 1, 0],
                                        "another_col": [0, 1, 2]}))


def test_simple_imputer_transform_drop_all_nan_columns_empty():
    X = pd.DataFrame([[np.nan, np.nan, np.nan]])
    transformer = SimpleImputer(impute_strategy='most_frequent')
    assert transformer.fit_transform(X).to_dataframe().empty
    assert_frame_equal(X, pd.DataFrame([[np.nan, np.nan, np.nan]]))

    transformer = SimpleImputer(impute_strategy='most_frequent')
    transformer.fit(X)
    assert transformer.transform(X).to_dataframe().empty
    assert_frame_equal(X, pd.DataFrame([[np.nan, np.nan, np.nan]]))


def test_simple_imputer_numpy_input():
    X = np.array([[np.nan, 0, 1, np.nan],
                  [np.nan, 2, 3, 2],
                  [np.nan, 2, 3, 0]])
    transformer = SimpleImputer(impute_strategy='mean')
    X_expected_arr = np.array([[0, 1, 1],
                               [2, 3, 2],
                               [2, 3, 0]])
    assert np.allclose(X_expected_arr, transformer.fit_transform(X).to_dataframe())
    np.testing.assert_almost_equal(X, np.array([[np.nan, 0, 1, np.nan],
                                                [np.nan, 2, 3, 2],
                                                [np.nan, 2, 3, 0]]))


@pytest.mark.parametrize("data_type", ["numeric", "categorical"])
def test_simple_imputer_fill_value(data_type):
    if data_type == "numeric":
        X = pd.DataFrame({
            "some numeric": [np.nan, 1, 0],
            "another numeric": [0, np.nan, 2]
        })
        fill_value = -1
        expected = pd.DataFrame({
            "some numeric": [-1, 1, 0],
            "another numeric": [0, -1, 2]
        })
    else:
        X = pd.DataFrame({
            "categorical with nan": pd.Series([np.nan, "1", np.nan, "0", "3"], dtype='category'),
            "object with nan": ["b", "b", np.nan, "c", np.nan]
        })
        fill_value = "fill"
        expected = pd.DataFrame({
            "categorical with nan": pd.Series(["fill", "1", "fill", "0", "3"], dtype='category'),
            "object with nan": pd.Series(["b", "b", "fill", "c", "fill"], dtype='category'),
        })
    y = pd.Series([0, 0, 1, 0, 1])
    imputer = SimpleImputer(impute_strategy="constant", fill_value=fill_value)
    imputer.fit(X, y)
    transformed = imputer.transform(X, y)
    assert_frame_equal(expected, transformed.to_dataframe(), check_dtype=False)

    imputer = SimpleImputer(impute_strategy="constant", fill_value=fill_value)
    transformed = imputer.fit_transform(X, y)
    assert_frame_equal(expected, transformed.to_dataframe(), check_dtype=False)


def test_simple_imputer_does_not_reset_index():
    X = pd.DataFrame({'input_val': np.arange(10), 'target': np.arange(10)})
    X.loc[5, 'input_val'] = np.nan
    assert X.index.tolist() == list(range(10))

    X.drop(0, inplace=True)
    y = X.pop('target')
    pd.testing.assert_frame_equal(pd.DataFrame({'input_val': [1.0, 2, 3, 4, np.nan, 6, 7, 8, 9]},
                                               dtype=float, index=list(range(1, 10))), X)

    imputer = SimpleImputer(impute_strategy="mean")
    imputer.fit(X, y=y)
    transformed = imputer.transform(X)
    pd.testing.assert_frame_equal(pd.DataFrame({'input_val': [1, 2, 3, 4, 5, 6, 7, 8, 9]},
                                               dtype=float,
                                               index=list(range(1, 10))),
                                  transformed.to_dataframe())


def test_simple_imputer_with_none():
    X = pd.DataFrame({"int with None": [1, 0, 5, None],
                      "float with None": [0.1, 0.0, 0.5, None],
                      "all None": [None, None, None, None]})
    y = pd.Series([0, 0, 1, 0, 1])
    imputer = SimpleImputer(impute_strategy="mean")
    imputer.fit(X, y)
    transformed = imputer.transform(X, y)
    expected = pd.DataFrame({"int with None": [1, 0, 5, 2],
                             "float with None": [0.1, 0.0, 0.5, 0.2]})
    assert_frame_equal(expected, transformed.to_dataframe(), check_dtype=False)

    X = pd.DataFrame({"category with None": pd.Series(["b", "a", "a", None], dtype='category'),
                      "boolean with None": pd.Series([True, None, False, True], dtype='boolean'),
                      "object with None": ["b", "a", "a", None],
                      "all None": [None, None, None, None]})
    y = pd.Series([0, 0, 1, 0, 1])
    imputer = SimpleImputer()
    imputer.fit(X, y)
    transformed = imputer.transform(X, y)
    expected = pd.DataFrame({"category with None": pd.Series(["b", "a", "a", "a"], dtype='category'),
                             "boolean with None": pd.Series([True, True, False, True], dtype='boolean'),
                             "object with None": pd.Series(["b", "a", "a", "a"], dtype='category')})
    assert_frame_equal(expected, transformed.to_dataframe(), check_dtype=False)


@pytest.mark.parametrize("X_df", [pd.DataFrame(pd.Series([1, 2, 3], dtype="Int64")),
                                  pd.DataFrame(pd.Series([1., 2., 3.], dtype="float")),
                                  pd.DataFrame(pd.Series(['a', 'b', 'a'], dtype="category")),
                                  pd.DataFrame(pd.Series([True, False, True], dtype="boolean")),
                                  pd.DataFrame(pd.Series(['this will be a natural language column because length', 'yay', 'hay'], dtype="string"))])
@pytest.mark.parametrize("has_nan", [True, False])
@pytest.mark.parametrize("impute_strategy", ["mean", "median", "most_frequent"])
def test_simple_imputer_woodwork_custom_overrides_returned_by_components(X_df, has_nan, impute_strategy):
    y = pd.Series([1, 2, 1])
    if has_nan:
        X_df.iloc[len(X_df) - 1, 0] = np.nan
    override_types = [Integer, Double, Categorical, NaturalLanguage, Boolean]
    for logical_type in override_types:
        try:
            X = ww.DataTable(X_df, logical_types={0: logical_type})
        except TypeError:
            continue

        impute_strategy_to_use = impute_strategy
        if logical_type in [NaturalLanguage, Categorical]:
            impute_strategy_to_use = "most_frequent"

        imputer = SimpleImputer(impute_strategy=impute_strategy_to_use)
        imputer.fit(X, y)
        transformed = imputer.transform(X, y)
        assert isinstance(transformed, ww.DataTable)
        if impute_strategy_to_use == "most_frequent" or not has_nan:
            assert transformed.logical_types == {0: logical_type}
        else:
            assert transformed.logical_types == {0: Double}
