import numpy as np
import pandas as pd
import pytest
import woodwork as ww
from pandas.testing import assert_frame_equal

from evalml.pipelines.components import Imputer


@pytest.fixture
def imputer_test_data():
    return pd.DataFrame({
        "categorical col": pd.Series(["zero", "one", "two", "zero", "three"], dtype='category'),
        "int col": [0, 1, 2, 0, 3],
        "object col": ["b", "b", "a", "c", "d"],
        "float col": [0.0, 1.0, 0.0, -2.0, 5.],
        "bool col": [True, False, False, True, True],
        "categorical with nan": pd.Series([np.nan, "1", np.nan, "0", "3"], dtype='category'),
        "int with nan": [np.nan, 1, 0, 0, 1],
        "float with nan": [0.0, 1.0, np.nan, -1.0, 0.],
        "object with nan": ["b", "b", np.nan, "c", np.nan],
        "bool col with nan": [True, np.nan, False, np.nan, True],
        "all nan": [np.nan, np.nan, np.nan, np.nan, np.nan],
        "all nan cat": pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan], dtype='category')
    })


def test_invalid_strategy_parameters():
    with pytest.raises(ValueError, match="Valid impute strategies are"):
        Imputer(numeric_impute_strategy="not a valid strategy")
    with pytest.raises(ValueError, match="Valid categorical impute strategies are"):
        Imputer(categorical_impute_strategy="mean")


def test_imputer_default_parameters():
    imputer = Imputer()
    expected_parameters = {
        'categorical_impute_strategy': 'most_frequent',
        'numeric_impute_strategy': 'mean',
        'categorical_fill_value': None,
        'numeric_fill_value': None
    }
    assert imputer.parameters == expected_parameters


@pytest.mark.parametrize("categorical_impute_strategy", ["most_frequent", "constant"])
@pytest.mark.parametrize("numeric_impute_strategy", ["mean", "median", "most_frequent", "constant"])
def test_imputer_init(categorical_impute_strategy, numeric_impute_strategy):

    imputer = Imputer(categorical_impute_strategy=categorical_impute_strategy,
                      numeric_impute_strategy=numeric_impute_strategy,
                      categorical_fill_value="str_fill_value",
                      numeric_fill_value=-1)
    expected_parameters = {
        'categorical_impute_strategy': categorical_impute_strategy,
        'numeric_impute_strategy': numeric_impute_strategy,
        'categorical_fill_value': 'str_fill_value',
        'numeric_fill_value': -1
    }
    expected_hyperparameters = {
        "categorical_impute_strategy": ["most_frequent"],
        "numeric_impute_strategy": ["mean", "median", "most_frequent"]
    }
    assert imputer.name == "Imputer"
    assert imputer.parameters == expected_parameters
    assert imputer.hyperparameter_ranges == expected_hyperparameters


def test_numeric_only_input(imputer_test_data):
    X = imputer_test_data[["int col", "float col",
                           "int with nan", "float with nan", "all nan"]]
    y = pd.Series([0, 0, 1, 0, 1])
    imputer = Imputer(numeric_impute_strategy="median")
    imputer.fit(X, y)
    transformed = imputer.transform(X, y)
    expected = pd.DataFrame({
        "int col": [0, 1, 2, 0, 3],
        "float col": [0.0, 1.0, 0.0, -2.0, 5.],
        "int with nan": [0.5, 1.0, 0.0, 0.0, 1.0],
        "float with nan": [0.0, 1.0, 0, -1.0, 0.]
    })
    assert_frame_equal(transformed, expected, check_dtype=False)

    imputer = Imputer()
    transformed = imputer.fit_transform(X, y)
    assert_frame_equal(transformed, expected, check_dtype=False)


def test_categorical_only_input(imputer_test_data):
    X = imputer_test_data[["categorical col", "object col", "bool col",
                           "categorical with nan", "object with nan",
                           "bool col with nan", "all nan cat"]]
    y = pd.Series([0, 0, 1, 0, 1])
    imputer = Imputer()
    imputer.fit(X, y)
    transformed = imputer.transform(X, y)
    expected = pd.DataFrame({
        "categorical col": pd.Series(["zero", "one", "two", "zero", "three"], dtype='category'),
        "object col": pd.Series(["b", "b", "a", "c", "d"], dtype='category'),
        "bool col": [True, False, False, True, True],
        "categorical with nan": pd.Series(["0", "1", "0", "0", "3"], dtype='category'),
        "object with nan": pd.Series(["b", "b", "b", "c", "b"], dtype='category'),
        "bool col with nan": [True, True, False, True, True]
    })

    imputer = Imputer()
    transformed = imputer.fit_transform(X, y)
    assert_frame_equal(transformed, expected, check_dtype=False)


def test_categorical_and_numeric_input(imputer_test_data):
    X = imputer_test_data
    y = pd.Series([0, 0, 1, 0, 1])
    imputer = Imputer()
    imputer.fit(X, y)
    transformed = imputer.transform(X, y)
    expected = pd.DataFrame({
        "categorical col": pd.Series(["zero", "one", "two", "zero", "three"], dtype='category'),
        "int col": [0, 1, 2, 0, 3],
        "object col": pd.Series(["b", "b", "a", "c", "d"], dtype='category'),
        "float col": [0.0, 1.0, 0.0, -2.0, 5.],
        "bool col": [True, False, False, True, True],
        "categorical with nan": pd.Series(["0", "1", "0", "0", "3"], dtype='category'),
        "int with nan": [0.5, 1.0, 0.0, 0.0, 1.0],
        "float with nan": [0.0, 1.0, 0, -1.0, 0.],
        "object with nan": pd.Series(["b", "b", "b", "c", "b"], dtype='category'),
        "bool col with nan": [True, True, False, True, True]
    })
    assert_frame_equal(transformed, expected, check_dtype=False)

    imputer = Imputer()
    transformed = imputer.fit_transform(X, y)
    assert_frame_equal(transformed, expected, check_dtype=False)


def test_drop_all_columns(imputer_test_data):
    X = imputer_test_data[["all nan cat", "all nan"]]
    y = pd.Series([0, 0, 1, 0, 1])
    imputer = Imputer()
    imputer.fit(X, y)
    transformed = imputer.transform(X, y)
    expected = X.drop(["all nan cat", "all nan"], axis=1)
    assert_frame_equal(transformed, expected, check_dtype=False)

    imputer = Imputer()
    transformed = imputer.fit_transform(X, y)
    assert_frame_equal(transformed, expected, check_dtype=False)


def test_typed_imputer_numpy_input():
    X = np.array([[1, 2, 2, 0],
                  [np.nan, 0, 0, 0],
                  [1, np.nan, np.nan, np.nan]])
    y = pd.Series([0, 0, 1])
    imputer = Imputer()
    imputer.fit(X, y)
    transformed = imputer.transform(X, y)
    expected = pd.DataFrame(np.array([[1, 2, 2, 0],
                                      [1, 0, 0, 0],
                                      [1, 1, 1, 0]]))
    assert_frame_equal(transformed, expected, check_dtype=False)

    imputer = Imputer()
    transformed = imputer.fit_transform(X, y)
    assert_frame_equal(transformed, expected, check_dtype=False)


def test_imputer_datetime_input():
    X = pd.DataFrame({'dates': ['20190902', '20200519', '20190607', np.nan],
                      'more dates': ['20190902', '20201010', '20190921', np.nan]})
    X['dates'] = pd.to_datetime(X['dates'], format='%Y%m%d')
    X['more dates'] = pd.to_datetime(X['more dates'], format='%Y%m%d')
    y = pd.Series()

    imputer = Imputer()
    imputer.fit(X, y)
    transformed = imputer.transform(X, y)
    assert_frame_equal(transformed, X, check_dtype=False)

    imputer = Imputer()
    transformed = imputer.fit_transform(X, y)
    assert_frame_equal(transformed, X, check_dtype=False)


@pytest.mark.parametrize("data_type", ['np', 'pd', 'ww'])
def test_imputer_empty_data(data_type):
    if data_type == 'pd':
        X = pd.DataFrame()
        y = pd.Series()
        expected = pd.DataFrame(index=pd.Index([]), columns=pd.Index([]))
    elif data_type == 'ww':
        X = ww.DataTable(pd.DataFrame())
        y = ww.DataColumn(pd.Series())
        expected = pd.DataFrame(index=pd.Index([]), columns=pd.Index([]))
    else:
        X = np.array([[]])
        y = np.array([])
        expected = pd.DataFrame(index=pd.Index([0]), columns=pd.Int64Index([]))
    imputer = Imputer()
    imputer.fit(X, y)
    transformed = imputer.transform(X, y)
    assert_frame_equal(transformed, expected, check_dtype=False)

    imputer = Imputer()
    transformed = imputer.fit_transform(X, y)
    assert_frame_equal(transformed, expected, check_dtype=False)


def test_imputer_does_not_reset_index():
    X = pd.DataFrame({'input_val': np.arange(10), 'target': np.arange(10),
                      'input_cat': ['a'] * 7 + ['b'] * 3})
    X.loc[5, 'input_val'] = np.nan
    X.loc[5, 'input_cat'] = np.nan
    assert X.index.tolist() == list(range(10))

    X.drop(0, inplace=True)
    y = X.pop('target')

    imputer = Imputer()
    imputer.fit(X, y=y)
    transformed = imputer.transform(X)
    pd.testing.assert_frame_equal(transformed,
                                  pd.DataFrame({'input_val': [1.0, 2, 3, 4, 5, 6, 7, 8, 9],
                                                'input_cat': pd.Categorical(['a'] * 6 + ['b'] * 3)},
                                               index=list(range(1, 10))))


def test_imputer_fill_value(imputer_test_data):
    X = imputer_test_data[["int with nan", "categorical with nan",
                           "float with nan", "object with nan", "bool col with nan"]]
    y = pd.Series([0, 0, 1, 0, 1])
    imputer = Imputer(categorical_impute_strategy="constant", numeric_impute_strategy="constant",
                      categorical_fill_value="fill", numeric_fill_value=-1)
    imputer.fit(X, y)
    transformed = imputer.transform(X, y)
    expected = pd.DataFrame({
        "int with nan": [-1, 1, 0, 0, 1],
        "categorical with nan": pd.Series(["fill", "1", "fill", "0", "3"], dtype='category'),
        "float with nan": [0.0, 1.0, -1, -1.0, 0.],
        "object with nan": pd.Series(["b", "b", "fill", "c", "fill"], dtype='category'),
        "bool col with nan": [True, "fill", False, "fill", True]
    })
    assert_frame_equal(transformed, expected, check_dtype=False)

    imputer = Imputer(categorical_impute_strategy="constant", numeric_impute_strategy="constant",
                      categorical_fill_value="fill", numeric_fill_value=-1)
    transformed = imputer.fit_transform(X, y)
    assert_frame_equal(transformed, expected, check_dtype=False)


def test_imputer_no_nans(imputer_test_data):
    X = imputer_test_data[["categorical col", "object col", "bool col"]]
    y = pd.Series([0, 0, 1, 0, 1])
    imputer = Imputer(categorical_impute_strategy="constant", numeric_impute_strategy="constant",
                      categorical_fill_value="fill", numeric_fill_value=-1)
    imputer.fit(X, y)
    transformed = imputer.transform(X, y)
    expected = pd.DataFrame({
        "categorical col": pd.Series(["zero", "one", "two", "zero", "three"], dtype='category'),
        "object col": pd.Series(["b", "b", "a", "c", "d"], dtype='category'),
        "bool col": [True, False, False, True, True],
    })
    assert_frame_equal(transformed, expected, check_dtype=False)

    imputer = Imputer(categorical_impute_strategy="constant", numeric_impute_strategy="constant",
                      categorical_fill_value="fill", numeric_fill_value=-1)
    transformed = imputer.fit_transform(X, y)
    assert_frame_equal(transformed, expected, check_dtype=False)


def test_imputer_with_none():
    X = pd.DataFrame({"int with None": [1, 0, 5, None],
                      "float with None": [0.1, 0.0, 0.5, None],
                      "category with None": pd.Series(["b", "a", "a", None], dtype='category'),
                      "boolean with None": [True, None, False, True],
                      "object with None": ["b", "a", "a", None],
                      "all None": [None, None, None, None]})
    y = pd.Series([0, 0, 1, 0, 1])
    imputer = Imputer()
    imputer.fit(X, y)
    transformed = imputer.transform(X, y)
    expected = pd.DataFrame({"int with None": [1, 0, 5, 2],
                             "float with None": [0.1, 0.0, 0.5, 0.2],
                             "category with None": pd.Series(["b", "a", "a", "a"], dtype='category'),
                             "boolean with None": [True, True, False, True],
                             "object with None": pd.Series(["b", "a", "a", "a"], dtype='category')})
    assert_frame_equal(transformed, expected, check_dtype=False)

    imputer = Imputer()
    transformed = imputer.fit_transform(X, y)
    assert_frame_equal(transformed, expected, check_dtype=False)


@pytest.mark.parametrize("data_type", ['pd', 'ww'])
def test_imputer_all_bool_return_original(data_type):
    X = pd.DataFrame([True, True, False, True, True], dtype=bool)
    X_expected_arr = pd.DataFrame([True, True, False, True, True], dtype=bool)
    y = pd.Series([1, 0, 0, 1, 0])
    if data_type == 'ww':
        X = ww.DataTable(X)
        y = ww.DataColumn(y)
    imputer = Imputer()
    imputer.fit(X, y)
    X_t = imputer.transform(X)
    assert_frame_equal(X_expected_arr, X_t)


@pytest.mark.parametrize("data_type", ['pd', 'ww'])
def test_imputer_bool_dtype_object(data_type, make_data_type):
    X = pd.DataFrame([True, np.nan, False, np.nan, True], dtype=object)
    y = pd.Series([1, 0, 0, 1, 0])
    X_expected_arr = pd.DataFrame([True, True, False, True, True], dtype=object)
    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)
    imputer = Imputer()
    imputer.fit(X, y)
    X_t = imputer.transform(X)
    assert_frame_equal(X_expected_arr, X_t)


@pytest.mark.parametrize("data_type", ['pd', 'ww'])
def test_imputer_multitype_with_one_bool(data_type):
    X_multi = pd.DataFrame({
        "bool with nan": pd.Series([True, np.nan, False, np.nan, False], dtype=object),
        "bool no nan": pd.Series([False, False, False, False, True], dtype=bool),
    })
    y = pd.Series([1, 0, 0, 1, 0])
    X_multi_expected_arr = pd.DataFrame({
        "bool with nan": pd.Series([True, False, False, False, False], dtype=object),
        "bool no nan": pd.Series([False, False, False, False, True], dtype=bool),
    })
    if data_type == 'ww':
        X_multi = ww.DataTable(X_multi)
        y = ww.DataColumn(y)
    imputer = Imputer()
    imputer.fit(X_multi, y)
    X_multi_t = imputer.transform(X_multi)
    assert_frame_equal(X_multi_expected_arr, X_multi_t)
