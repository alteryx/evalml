import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from evalml.pipelines.components import DropColumns, SelectColumns


@pytest.mark.parametrize("class_to_test", [DropColumns, SelectColumns])
def test_column_transformer_init(class_to_test):
    transformer = class_to_test(columns=None)
    assert transformer.parameters["columns"] is None

    transformer = class_to_test(columns=[])
    assert transformer.parameters["columns"] == []

    transformer = class_to_test(columns=["a", "b"])
    assert transformer.parameters["columns"] == ["a", "b"]

    with pytest.raises(ValueError, match="Parameter columns must be a list."):
        _ = class_to_test(columns="Column1")


@pytest.mark.parametrize("class_to_test", [DropColumns, SelectColumns])
def test_column_transformer_empty_X(class_to_test):
    X = pd.DataFrame()
    transformer = class_to_test(columns=[])
    assert_frame_equal(X, transformer.transform(X).to_dataframe())

    transformer = class_to_test(columns=[])
    assert_frame_equal(X, transformer.fit_transform(X).to_dataframe())

    transformer = class_to_test(columns=["not in data"])
    with pytest.raises(ValueError, match="'not in data' not found in input data"):
        transformer.fit(X)

    transformer = class_to_test(columns=list(X.columns))
    assert transformer.transform(X).to_dataframe().empty


@pytest.mark.parametrize("class_to_test,checking_functions",
                         [(DropColumns, [lambda X, X_t: X_t.equals(X.astype("Int64")),
                                         lambda X, X_t: X_t.equals(X.astype("Int64")),
                                         lambda X, X_t: X_t.equals(X.drop(columns=["one"]).astype("Int64")),
                                         lambda X, X_t: X_t.empty]),
                          (SelectColumns, [lambda X, X_t: X_t.empty,
                                           lambda X, X_t: X_t.empty,
                                           lambda X, X_t: X_t.equals(X[["one"]].astype("Int64")),
                                           lambda X, X_t: X_t.equals(X.astype("Int64"))])
                          ])
def test_column_transformer_transform(class_to_test, checking_functions):
    X = pd.DataFrame({'one': [1, 2, 3, 4], 'two': [2, 3, 4, 5], 'three': [1, 2, 3, 4]})
    check1, check2, check3, check4 = checking_functions

    transformer = class_to_test(columns=None)
    assert check1(X, transformer.transform(X).to_dataframe())

    transformer = class_to_test(columns=[])
    assert check2(X, transformer.transform(X).to_dataframe())

    transformer = class_to_test(columns=["one"])
    assert check3(X, transformer.transform(X).to_dataframe())

    transformer = class_to_test(columns=list(X.columns))
    assert check4(X, transformer.transform(X).to_dataframe())


@pytest.mark.parametrize("class_to_test,checking_functions",
                         [(DropColumns, [lambda X, X_t: X_t.equals(X.astype("Int64")),
                                         lambda X, X_t: X_t.equals(X.drop(columns=["one"]).astype("Int64")),
                                         lambda X, X_t: X_t.empty]),
                          (SelectColumns, [lambda X, X_t: X_t.empty,
                                           lambda X, X_t: X_t.equals(X[["one"]].astype("Int64")),
                                           lambda X, X_t: X_t.equals(X.astype("Int64"))])
                          ])
def test_column_transformer_fit_transform(class_to_test, checking_functions):
    X = pd.DataFrame({'one': [1, 2, 3, 4], 'two': [2, 3, 4, 5], 'three': [1, 2, 3, 4]})
    check1, check2, check3 = checking_functions

    assert check1(X, class_to_test(columns=[]).fit_transform(X).to_dataframe())

    assert check2(X, class_to_test(columns=["one"]).fit_transform(X).to_dataframe())

    assert check3(X, class_to_test(columns=list(X.columns)).fit_transform(X).to_dataframe())


@pytest.mark.parametrize("class_to_test", [DropColumns, SelectColumns])
def test_drop_column_transformer_input_invalid_col_name(class_to_test):
    X = pd.DataFrame({'one': [1, 2, 3, 4], 'two': [2, 3, 4, 5], 'three': [1, 2, 3, 4]})
    transformer = class_to_test(columns=["not in data"])
    with pytest.raises(ValueError, match="'not in data' not found in input data"):
        transformer.fit(X)
    with pytest.raises(ValueError, match="'not in data' not found in input data"):
        transformer.transform(X)
    with pytest.raises(ValueError, match="'not in data' not found in input data"):
        transformer.transform(X)

    X = np.arange(12).reshape(3, 4)
    transformer = class_to_test(columns=[5])
    with pytest.raises(ValueError, match="'5' not found in input data"):
        transformer.fit(X)
    with pytest.raises(ValueError, match="'5' not found in input data"):
        transformer.transform(X)
    with pytest.raises(ValueError, match="'5' not found in input data"):
        transformer.fit_transform(X)


@pytest.mark.parametrize("class_to_test,answers",
                         [(DropColumns, [pd.DataFrame([[0, 2, 3], [4, 6, 7], [8, 10, 11]], columns=[0, 2, 3], dtype="Int64"),
                                         pd.DataFrame([[], [], []], dtype="Int64"),
                                         pd.DataFrame(np.arange(12).reshape(3, 4), dtype="Int64")]),
                          (SelectColumns, [pd.DataFrame([[1], [5], [9]], columns=[1], dtype="Int64"),
                                           pd.DataFrame(np.arange(12).reshape(3, 4), dtype="Int64"),
                                           pd.DataFrame([[], [], []], dtype="Int64")])
                          ])
def test_column_transformer_int_col_names_np_array(class_to_test, answers):
    X = np.arange(12).reshape(3, 4)
    answer1, answer2, answer3 = answers

    transformer = class_to_test(columns=[1])
    assert_frame_equal(answer1, transformer.transform(X).to_dataframe())

    transformer = class_to_test(columns=[0, 1, 2, 3])
    assert_frame_equal(answer2, transformer.transform(X).to_dataframe())

    transformer = class_to_test(columns=[])
    assert_frame_equal(answer3, transformer.transform(X).to_dataframe())
