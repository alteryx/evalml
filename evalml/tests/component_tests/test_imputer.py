import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from evalml.pipelines.components import Imputer


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


def test_imputer_init():
    imputer = Imputer(categorical_impute_strategy="most_frequent",
                      numeric_impute_strategy="median",
                      categorical_fill_value="str_fill_value",
                      numeric_fill_value=-1)
    expected_parameters = {
        'categorical_impute_strategy': 'most_frequent',
        'numeric_impute_strategy': 'median',
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


def test_numeric_only_input():
    X = pd.DataFrame({
        "int col": [0, 1, 2, 0, 3],
        "float col": [0.0, 1.0, 0.0, -2.0, 5.],
        "int with nan": [np.nan, 1, 2, 1, 0],
        "float with nan": [0.0, 1.0, np.nan, -1.0, 0.],
        "all nan": [np.nan, np.nan, np.nan, np.nan, np.nan]
    })
    y = pd.Series([0, 0, 1, 0, 1])
    imputer = Imputer(numeric_impute_strategy="median")
    imputer.fit(X, y)
    transformed = imputer.transform(X, y)
    expected = pd.DataFrame({
        "int col": [0, 1, 2, 0, 3],
        "float col": [0.0, 1.0, 0.0, -2.0, 5.],
        "int with nan": [1, 1, 2, 1, 0],
        "float with nan": [0.0, 1.0, 0, -1.0, 0.]
    })
    assert_frame_equal(transformed, expected, check_dtype=False)

    imputer = Imputer()
    transformed = imputer.fit_transform(X, y)
    assert_frame_equal(transformed, expected, check_dtype=False)


def test_categorical_only_input():
    X = pd.DataFrame({
        "categorical col": pd.Series(["0", "1", "2", "0", "3"], dtype='category'),
        "object col": ["b", "b", "a", "c", "d"],
        "bool col": [True, False, False, True, True],
        "categorical with nan": pd.Series([np.nan, "1", np.nan, "0", "3"], dtype='category'),
        "object with nan": ["b", "b", np.nan, "c", np.nan],
        "bool col with nan": [True, np.nan, False, np.nan, True],
        "all nan": pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan], dtype='category')
    })
    y = pd.Series([0, 0, 1, 0, 1])
    imputer = Imputer()
    imputer.fit(X, y)
    transformed = imputer.transform(X, y)
    expected = pd.DataFrame({
        "categorical col": pd.Series(["0", "1", "2", "0", "3"], dtype='category'),
        "object col": ["b", "b", "a", "c", "d"],
        "bool col": [True, False, False, True, True],
        "categorical with nan": pd.Series(["0", "1", "0", "0", "3"], dtype='category'),
        "object with nan": ["b", "b", "b", "c", "b"],
        "bool col with nan": [True, True, False, True, True]
    })

    imputer = Imputer()
    transformed = imputer.fit_transform(X, y)
    assert_frame_equal(transformed, expected, check_dtype=False)


def test_categorical_and_numeric_input():
    X = pd.DataFrame({
        "categorical col": pd.Series([0, 1, 2, 0, 3], dtype='category'),
        "int col": [0, 1, 2, 0, 3],
        "object col": ["b", "b", "a", "c", "d"],
        "float col": [0.0, 1.0, 0.0, -2.0, 5.],
        "bool col": [True, False, False, True, True],
        "int with nan": [np.nan, 1, 2, 1, 0],
        "categorical with nan": pd.Series([np.nan, "1", np.nan, "0", "3"], dtype='category'),
        "float with nan": [0.0, 1.0, np.nan, -1.0, 0.],
        "object with nan": ["b", "b", np.nan, "c", np.nan],
        "bool col with nan": [True, np.nan, False, np.nan, True],
        "all nan": [np.nan, np.nan, np.nan, np.nan, np.nan]
    })
    y = pd.Series([0, 0, 1, 0, 1])
    imputer = Imputer()
    imputer.fit(X, y)
    transformed = imputer.transform(X, y)
    expected = pd.DataFrame({
        "categorical col": pd.Series(["0", "1", "2", "0", "3"], dtype='category'),
        "int col": [0, 1, 2, 0, 3],
        "object col": ["b", "b", "a", "c", "d"],
        "float col": [0.0, 1.0, 0.0, -2.0, 5.],
        "bool col": [True, False, False, True, True],
        "int with nan": [1, 1, 2, 1, 0],
        "categorical with nan": pd.Series(["0", "1", "0", "0", "3"], dtype='category'),
        "float with nan": [0.0, 1.0, 0, -1.0, 0.],
        "object with nan": ["b", "b", "b", "c", "b"],
        "bool col with nan": [True, True, False, True, True]
    })
    assert_frame_equal(transformed, expected, check_dtype=False)

    imputer = Imputer()
    transformed = imputer.fit_transform(X, y)
    assert_frame_equal(transformed, expected, check_dtype=False)


def test_drop_all_columns():
    X = pd.DataFrame({
        "all nan cat": pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan], dtype='category'),
        "all nan": [np.nan, np.nan, np.nan, np.nan, np.nan]
    })
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


@pytest.mark.parametrize("data_type", ['np', 'pd'])
def test_imputer_empty_data(data_type):
    if data_type == 'pd':
        X = pd.DataFrame()
        y = pd.Series()
        expected = pd.DataFrame(index=pd.Int64Index([]), columns=pd.Index([]))
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


def test_imputer_resets_index():
    X = pd.DataFrame({'input_val': np.arange(10), 'target': np.arange(10)})
    X.loc[5, 'input_val'] = np.nan
    assert X.index.tolist() == list(range(10))

    X.drop(0, inplace=True)
    y = X.pop('target')
    pd.testing.assert_frame_equal(X,
                                  pd.DataFrame({'input_val': [1.0, 2, 3, 4, np.nan, 6, 7, 8, 9]},
                                               dtype=float,
                                               index=list(range(1, 10))))

    imputer = Imputer()
    imputer.fit(X, y=y)
    transformed = imputer.transform(X)
    pd.testing.assert_frame_equal(transformed,
                                  pd.DataFrame({'input_val': [1.0, 2, 3, 4, 5, 6, 7, 8, 9]},
                                               dtype=float,
                                               index=list(range(0, 9))))


def test_imputer_fill_value():
    X = pd.DataFrame({
        "int with nan": [np.nan, 1, 2, 1, 0],
        "categorical with nan": pd.Series([np.nan, "1", np.nan, "0", "3"], dtype='category'),
        "float with nan": [0.0, 1.0, np.nan, -1.0, 0.],
        "object with nan": ["b", "b", np.nan, "c", np.nan],
        "bool col with nan": [True, np.nan, False, np.nan, True],
    })
    y = pd.Series([0, 0, 1, 0, 1])
    imputer = Imputer(categorical_impute_strategy="constant", numeric_impute_strategy="constant",
                      categorical_fill_value="fill", numeric_fill_value=-1)
    imputer.fit(X, y)
    transformed = imputer.transform(X, y)
    expected = pd.DataFrame({
        "int with nan": [-1, 1, 2, 1, 0],
        "categorical with nan": pd.Series(["fill", "1", "fill", "0", "3"], dtype='category'),
        "float with nan": [0.0, 1.0, -1, -1.0, 0.],
        "object with nan": ["b", "b", "fill", "c", "fill"],
        "bool col with nan": [True, "fill", False, "fill", True]
    })
    assert_frame_equal(transformed, expected, check_dtype=False)

    imputer = Imputer(categorical_impute_strategy="constant", numeric_impute_strategy="constant",
                      categorical_fill_value="fill", numeric_fill_value=-1)
    transformed = imputer.fit_transform(X, y)
    assert_frame_equal(transformed, expected, check_dtype=False)


def test_imputer_no_nans():
    X = pd.DataFrame({
        "categorical col": pd.Series(["0", "1", "2", "0", "3"], dtype='category'),
        "object col": ["b", "b", "a", "c", "d"],
        "bool col": [True, False, False, True, True],
    })
    y = pd.Series([0, 0, 1, 0, 1])
    imputer = Imputer(categorical_impute_strategy="constant", numeric_impute_strategy="constant",
                      categorical_fill_value="fill", numeric_fill_value=-1)
    imputer.fit(X, y)
    transformed = imputer.transform(X, y)
    expected = pd.DataFrame({
        "categorical col": pd.Series(["0", "1", "2", "0", "3"], dtype='category'),
        "object col": ["b", "b", "a", "c", "d"],
        "bool col": [True, False, False, True, True],
    })
    assert_frame_equal(transformed, expected, check_dtype=False)

    imputer = Imputer(categorical_impute_strategy="constant", numeric_impute_strategy="constant",
                      categorical_fill_value="fill", numeric_fill_value=-1)
    transformed = imputer.fit_transform(X, y)
    assert_frame_equal(transformed, expected, check_dtype=False)
