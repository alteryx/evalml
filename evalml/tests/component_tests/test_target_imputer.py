import numpy as np
import pandas as pd
import pytest
import woodwork as ww
from pandas.testing import assert_frame_equal, assert_series_equal
from woodwork.logical_types import (
    Boolean,
    Categorical,
    Double,
    Integer,
    NaturalLanguage,
)

from evalml.pipelines.components import TargetImputer


def test_target_imputer_no_y(X_y_binary):
    X, y = X_y_binary
    imputer = TargetImputer()
    assert imputer.fit_transform(None, None) == (None, None)

    imputer = TargetImputer()
    imputer.fit(None, None)
    assert imputer.transform(None, None) == (None, None)


def test_target_imputer_with_X():
    X = pd.DataFrame({"some col": [1, 3, np.nan]})
    y = pd.Series([np.nan, 1, 3])
    imputer = TargetImputer(impute_strategy="median")
    y_expected = pd.Series([2, 1, 3])
    X_expected = pd.DataFrame({"some col": [1, 3, np.nan]})
    X_t, y_t = imputer.fit_transform(X, y)
    assert_series_equal(y_expected, y_t, check_dtype=False)
    assert_frame_equal(X_expected, X_t, check_dtype=False)


def test_target_imputer_median():
    y = pd.Series([np.nan, 1, 10, 10, 6])
    imputer = TargetImputer(impute_strategy="median")
    y_expected = pd.Series([8, 1, 10, 10, 6])
    _, y_t = imputer.fit_transform(None, y)
    assert_series_equal(y_expected, y_t, check_dtype=False)


def test_target_imputer_mean():
    y = pd.Series([np.nan, 2, 0])
    imputer = TargetImputer(impute_strategy="mean")
    y_expected = pd.Series([1, 2, 0])
    _, y_t = imputer.fit_transform(None, y)
    assert_series_equal(y_expected, y_t, check_dtype=False)


@pytest.mark.parametrize(
    "fill_value, y, y_expected",
    [
        (None, pd.Series([np.nan, 0, 5]), pd.Series([0, 0, 5])),
        (
            None,
            pd.Series([np.nan, "a", "b"] * 5),
            pd.Series(["missing_value", "a", "b"] * 5).astype("category"),
        ),
        (3, pd.Series([np.nan, 0, 5]), pd.Series([3, 0, 5])),
        (
            3,
            pd.Series([np.nan, "a", "b"] * 5),
            pd.Series([3, "a", "b"] * 5).astype("category"),
        ),
    ],
)
def test_target_imputer_constant(fill_value, y, y_expected):
    imputer = TargetImputer(impute_strategy="constant", fill_value=fill_value)
    _, y_t = imputer.fit_transform(None, y)
    assert_series_equal(y_expected, y_t, check_dtype=False)


def test_target_imputer_most_frequent():
    y = pd.Series([np.nan, "a", "b"] * 5)
    imputer = TargetImputer(impute_strategy="most_frequent")
    y_expected = pd.Series(["a", "a", "b"] * 5).astype("category")
    _, y_t = imputer.fit_transform(None, y)
    assert_series_equal(y_expected, y_t, check_dtype=False)

    y = pd.Series([np.nan, 1, 1, 2])
    imputer = TargetImputer(impute_strategy="most_frequent")
    y_expected = pd.Series([1, 1, 1, 2])
    _, y_t = imputer.fit_transform(None, y)
    assert_series_equal(y_expected, y_t, check_dtype=False)


def test_target_imputer_col_with_non_numeric_with_numeric_strategy():
    y = pd.Series([np.nan, "a", "b"] * 5)
    imputer = TargetImputer(impute_strategy="mean")
    with pytest.raises(
        ValueError, match="Cannot use mean strategy with non-numeric data"
    ):
        imputer.fit_transform(None, y)
    with pytest.raises(
        ValueError, match="Cannot use mean strategy with non-numeric data"
    ):
        imputer.fit(None, y)
    imputer = TargetImputer(impute_strategy="median")
    with pytest.raises(
        ValueError, match="Cannot use median strategy with non-numeric data"
    ):
        imputer.fit_transform(None, y)
    with pytest.raises(
        ValueError, match="Cannot use median strategy with non-numeric data"
    ):
        imputer.fit(None, y)


@pytest.mark.parametrize("data_type", ["pd", "ww"])
def test_target_imputer_all_bool_return_original(data_type, make_data_type):
    y = pd.Series([True, True, False, True, True], dtype=bool)
    y = make_data_type(data_type, y)
    y_expected = pd.Series([True, True, False, True, True], dtype=bool)
    imputer = TargetImputer()
    imputer.fit(None, y)
    _, y_t = imputer.transform(None, y)
    assert_series_equal(y_expected, y_t)


@pytest.mark.parametrize("data_type", ["pd", "ww"])
def test_target_imputer_boolean_dtype(data_type, make_data_type):
    y = pd.Series([True, np.nan, False, np.nan, True], dtype="category")
    y_expected = pd.Series([True, True, False, True, True], dtype="category")
    y = make_data_type(data_type, y)
    imputer = TargetImputer()
    imputer.fit(None, y)
    _, y_t = imputer.transform(None, y)
    assert_series_equal(y_expected, y_t)


@pytest.mark.parametrize("y", [[np.nan, np.nan], [pd.NA, pd.NA]])
def test_target_imputer_fit_transform_all_nan_empty(y):
    y = pd.Series(y)

    imputer = TargetImputer()

    with pytest.raises(TypeError, match="Provided target full of nulls."):
        imputer.fit(None, y)

    imputer = TargetImputer()
    with pytest.raises(TypeError, match="Provided target full of nulls."):
        imputer.fit_transform(None, y)


def test_target_imputer_numpy_input():
    y = np.array([np.nan, 0, 2])
    imputer = TargetImputer(impute_strategy="mean")
    y_expected = np.array([1, 0, 2])
    _, y_t = imputer.fit_transform(None, y)
    assert np.allclose(y_expected, y_t)
    np.testing.assert_almost_equal(y, np.array([np.nan, 0, 2]))


def test_target_imputer_does_not_reset_index():
    y = pd.Series(np.arange(10))
    y[5] = np.nan
    assert y.index.tolist() == list(range(10))

    y.drop(0, inplace=True)
    pd.testing.assert_series_equal(
        pd.Series(
            [1, 2, 3, 4, np.nan, 6, 7, 8, 9], dtype=float, index=list(range(1, 10))
        ),
        y,
    )

    imputer = TargetImputer(impute_strategy="mean")
    imputer.fit(None, y=y)
    _, y_t = imputer.transform(None, y)
    pd.testing.assert_series_equal(
        pd.Series([1.0, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float, index=list(range(1, 10))),
        y_t,
    )


@pytest.mark.parametrize(
    "y, y_expected",
    [
        (pd.Series([1, 0, 5, None]), pd.Series([1, 0, 5, 2])),
        (pd.Series([0.1, 0.0, 0.5, None]), pd.Series([0.1, 0.0, 0.5, 0.2])),
    ],
)
def test_target_imputer_with_none(y, y_expected):
    imputer = TargetImputer(impute_strategy="mean")
    _, y_t = imputer.fit_transform(None, y)
    assert_series_equal(y_expected, y_t, check_dtype=False)


@pytest.mark.parametrize(
    "y, y_expected",
    [
        (
            pd.Series(["b", "a", "a", None] * 4, dtype="category"),
            pd.Series(["b", "a", "a", "a"] * 4, dtype="category"),
        ),
        (
            pd.Series([True, None, False, True], dtype="category"),
            pd.Series([True, True, False, True], dtype="category"),
        ),
        (
            pd.Series(["b", "a", "a", None] * 4),
            pd.Series(["b", "a", "a", "a"] * 4, dtype="category"),
        ),
    ],
)
def test_target_imputer_with_none_non_numeric(y, y_expected):
    imputer = TargetImputer()
    _, y_t = imputer.fit_transform(None, y)
    assert_series_equal(y_expected, y_t, check_dtype=False)


@pytest.mark.parametrize(
    "y_pd",
    [
        pd.Series([1, 2, 3], dtype="int64"),
        pd.Series([1.0, 2.0, 3.0], dtype="float"),
        pd.Series(["a", "b", "a"], dtype="category"),
        pd.Series([True, False, True], dtype=bool),
    ],
)
@pytest.mark.parametrize("has_nan", [True, False])
@pytest.mark.parametrize("impute_strategy", ["mean", "median", "most_frequent"])
def test_target_imputer_woodwork_custom_overrides_returned_by_components(
    y_pd, has_nan, impute_strategy
):
    y_to_use = y_pd.copy()
    if has_nan:
        y_to_use[len(y_pd) - 1] = np.nan
    override_types = [Integer, Double, Categorical, Boolean]
    for logical_type in override_types:
        # Converting a column with NaNs to Boolean will impute NaNs.
        if has_nan and logical_type == Boolean:
            continue
        try:
            y = ww.init_series(y_to_use.copy(), logical_type=logical_type)
        except (ww.exceptions.TypeConversionError, ValueError):
            continue

        impute_strategy_to_use = impute_strategy
        if logical_type in [Categorical, NaturalLanguage]:
            impute_strategy_to_use = "most_frequent"

        imputer = TargetImputer(impute_strategy=impute_strategy_to_use)
        imputer.fit(None, y)
        _, y_t = imputer.transform(None, y)

        if impute_strategy_to_use == "most_frequent" or not has_nan:
            assert type(y_t.ww.logical_type) == logical_type
        else:
            assert type(y_t.ww.logical_type) == Double
