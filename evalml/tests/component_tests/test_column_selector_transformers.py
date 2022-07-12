import numpy as np
import pandas as pd
import pytest
import woodwork as ww
from pandas.testing import assert_frame_equal

from evalml.pipelines.components import DropColumns, SelectByType, SelectColumns


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


def test_select_by_type_init():
    transformer = SelectByType(column_types=None)
    assert transformer.parameters["column_types"] is None

    transformer = SelectByType(column_types=[])
    assert transformer.parameters["column_types"] == []

    transformer = SelectByType(column_types=["a", "b"])
    assert transformer.parameters["column_types"] == ["a", "b"]


def test_select_by_type_empty_X():
    X = pd.DataFrame()
    transformer = SelectByType(columns=[])
    assert_frame_equal(X, transformer.transform(X))

    transformer = SelectByType(columns=[])
    assert_frame_equal(X, transformer.fit_transform(X))

    transformer = SelectByType(columns=list(X.columns))
    assert transformer.transform(X).empty


@pytest.mark.parametrize(
    "class_to_test,checking_functions",
    [
        (
            DropColumns,
            [
                lambda X, X_t: X_t.equals(X.astype("int64")),
                lambda X, X_t: X_t.equals(X.astype("int64")),
                lambda X, X_t: X_t.equals(X.drop(columns=["one"]).astype("int64")),
                lambda X, X_t: X_t.empty,
            ],
        ),
        (
            SelectColumns,
            [
                lambda X, X_t: X_t.empty,
                lambda X, X_t: X_t.empty,
                lambda X, X_t: X_t.equals(X[["one"]].astype("int64")),
                lambda X, X_t: X_t.equals(X.astype("int64")),
            ],
        ),
        (
            SelectByType,
            [
                lambda X, X_t: X_t.empty,
                lambda X, X_t: X_t.empty,
                lambda X, X_t: X_t.equals(X[["three"]].astype("int64")),
                lambda X, X_t: X_t.astype(str).equals(X.astype(str)),
            ],
        ),
    ],
)
def test_column_transformer_transform(class_to_test, checking_functions):
    if class_to_test is SelectByType:
        X = pd.DataFrame(
            {
                "one": ["1", "2", "3", "4"],
                "two": [False, True, True, False],
                "three": [1, 2, 3, 4],
            },
        )
    else:
        X = pd.DataFrame(
            {"one": [1, 2, 3, 4], "two": [2, 3, 4, 5], "three": [1, 2, 3, 4]},
        )
    check1, check2, check3, check4 = checking_functions

    transformer = class_to_test(columns=None)
    assert check1(X, transformer.transform(X))

    transformer = class_to_test(columns=[])
    assert check2(X, transformer.transform(X))

    if class_to_test is SelectByType:
        transformer = class_to_test(column_types=["integer"])
    else:
        transformer = class_to_test(columns=["one"])
    assert check3(X, transformer.transform(X))

    if class_to_test is SelectByType:
        transformer = class_to_test(column_types=["categorical", "Boolean", "Integer"])
        X.ww.init(logical_types={"one": "categorical"})
    else:
        transformer = class_to_test(columns=list(X.columns))
    assert check4(X, transformer.transform(X))


@pytest.mark.parametrize(
    "class_to_test,checking_functions",
    [
        (
            DropColumns,
            [
                lambda X, X_t: X_t.equals(X.astype("int64")),
                lambda X, X_t: X_t.equals(X.drop(columns=["one"]).astype("int64")),
                lambda X, X_t: X_t.empty,
            ],
        ),
        (
            SelectColumns,
            [
                lambda X, X_t: X_t.empty,
                lambda X, X_t: X_t.equals(X[["one"]].astype("int64")),
                lambda X, X_t: X_t.equals(X.astype("int64")),
            ],
        ),
        (
            SelectByType,
            [
                lambda X, X_t: X_t.empty,
                lambda X, X_t: X_t.equals(X[["three"]].astype("int64")),
                lambda X, X_t: X_t.astype(str).equals(X.astype(str)),
            ],
        ),
    ],
)
def test_column_transformer_fit_transform(class_to_test, checking_functions):
    if class_to_test is SelectByType:
        X = pd.DataFrame(
            {
                "one": ["1", "2", "3", "4"],
                "two": [False, True, True, False],
                "three": [1, 2, 3, 4],
            },
        )
    else:
        X = pd.DataFrame(
            {"one": [1, 2, 3, 4], "two": [2, 3, 4, 5], "three": [1, 2, 3, 4]},
        )
    check1, check2, check3 = checking_functions

    assert check1(X, class_to_test(columns=[]).fit_transform(X))

    if class_to_test is SelectByType:
        assert check2(X, class_to_test(column_types=["integer"]).fit_transform(X))
    else:
        assert check2(X, class_to_test(columns=["one"]).fit_transform(X))

    if class_to_test is SelectByType:
        X.ww.init(logical_types={"one": "categorical"})
        assert check3(
            X,
            class_to_test(
                column_types=["categorical", "boolean", "integer"],
            ).fit_transform(X),
        )
    else:
        assert check3(X, class_to_test(columns=list(X.columns)).fit_transform(X))


@pytest.mark.parametrize(
    "class_to_test,answers",
    [
        (
            DropColumns,
            [
                pd.DataFrame(
                    [[0, 2, 3], [4, 6, 7], [8, 10, 11]],
                    columns=[0, 2, 3],
                    dtype="int64",
                ),
                pd.DataFrame([[], [], []], dtype="Int64"),
                pd.DataFrame(np.arange(12).reshape(3, 4), dtype="int64"),
            ],
        ),
        (
            SelectColumns,
            [
                pd.DataFrame([[1], [5], [9]], columns=[1], dtype="int64"),
                pd.DataFrame(np.arange(12).reshape(3, 4), dtype="int64"),
                pd.DataFrame([[], [], []], dtype="Int64"),
            ],
        ),
    ],
)
def test_column_transformer_int_col_names_np_array(class_to_test, answers):
    X = np.arange(12).reshape(3, 4)
    answer1, answer2, answer3 = answers

    transformer = class_to_test(columns=[1])
    assert_frame_equal(answer1, transformer.transform(X))

    transformer = class_to_test(columns=[0, 1, 2, 3])
    assert_frame_equal(answer2, transformer.transform(X))

    transformer = class_to_test(columns=[])
    assert_frame_equal(answer3, transformer.transform(X))


def test_typeortag_column_transformer_ww_logical_and_semantic_types():
    X = pd.DataFrame(
        {
            "one": ["1", "2", "3", "4"],
            "two": [False, True, True, False],
            "three": [1, 2, 3, 4],
            "four": [4.0, 2.3, 6.5, 2.6],
        },
    )
    X.ww.init(logical_types={"one": "categorical"})

    X_t = SelectByType(column_types=[ww.logical_types.Integer]).fit_transform(X)
    assert X_t.equals(X[["three"]].astype("int64"))

    X_t = SelectByType(column_types=["Double"]).fit_transform(X)
    assert X_t.equals(X[["four"]].astype("float64"))

    X_t = SelectByType(
        column_types=[
            ww.logical_types.Categorical,
            ww.logical_types.Boolean,
            ww.logical_types.Integer,
            ww.logical_types.Double,
        ],
    ).fit_transform(X)
    assert X_t.astype(str).equals(X.astype(str))

    X_t = SelectByType(column_types=["numeric"]).fit_transform(X)
    assert X_t.astype(str).equals(X[["three", "four"]].astype(str))


def test_column_selector_missing_columns():
    selector = SelectColumns(columns=["A", "B", "C", "D"])
    X = pd.DataFrame(columns=["A", "C", "F", "G"])

    X_t = selector.fit_transform(X)
    assert (X_t.columns == ["A", "C"]).all()


def test_select_by_type_exclude():
    selector = SelectByType(column_types="category", exclude=True)
    X = pd.DataFrame(
        {
            "one": ["1", "2", "3", "4"],
            "two": [1, 2, 3, 4],
            "three": [4.0, 2.3, 6.5, 2.6],
        },
    )
    X.ww.init(logical_types={"one": "categorical"})

    X_t = selector.fit_transform(X)
    assert list(X_t.columns) == ["two", "three"]
