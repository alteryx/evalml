from datetime import datetime

import matplotlib
import numpy as np
import pandas as pd
import pytest

from evalml.pipelines.components.transformers.preprocessing import (
    PolynomialDecomposer,
    STLDecomposer,
)


def test_set_time_index():
    x = np.arange(0, 2 * np.pi, 0.01)
    dts = pd.date_range(datetime.today(), periods=len(x))
    X = pd.DataFrame({"x": x})
    X = X.set_index(dts)
    y = pd.Series(np.sin(x))

    assert isinstance(y.index, pd.RangeIndex)

    # Use the PolynomialDecomposer since we can't use a Decomposer class as it
    # has abstract methods.
    decomposer = PolynomialDecomposer()
    y_time_index = decomposer._set_time_index(X, y)
    assert isinstance(y_time_index.index, pd.DatetimeIndex)


@pytest.mark.parametrize(
    "decomposer_child_class",
    [PolynomialDecomposer, STLDecomposer],
)
@pytest.mark.parametrize(
    "y_has_time_index",
    ["y_has_time_index", "y_doesnt_have_time_index"],
)
def test_decomposer_plot_decomposition(
    decomposer_child_class,
    y_has_time_index,
    generate_seasonal_data,
):
    step = 0.01
    period = 9
    X, y = generate_seasonal_data(real_or_synthetic="synthetic")(period, step)
    if y_has_time_index == "y_has_time_index":
        y = y.set_axis(X.index)

    pdc = decomposer_child_class(degree=1, seasonal_period=period)
    pdc.fit_transform(X, y)
    fig, axs = pdc.plot_decomposition(X, y, show=False)
    assert isinstance(fig, matplotlib.pyplot.Figure)
    assert isinstance(axs, np.ndarray)
    assert all([isinstance(ax, matplotlib.pyplot.Axes) for ax in axs])


@pytest.mark.parametrize(
    "decomposer_child_class",
    [PolynomialDecomposer, STLDecomposer],
)
@pytest.mark.parametrize("X_num_time_columns", [0, 1, 2, 3])
@pytest.mark.parametrize(
    "X_has_time_index",
    ["X_has_time_index", "X_doesnt_have_time_index"],
)
@pytest.mark.parametrize(
    "y_has_time_index",
    ["y_has_time_index", "y_doesnt_have_time_index"],
)
@pytest.mark.parametrize(
    "time_index_specified",
    [
        "time_index_is_specified",
        "time_index_not_specified",
        "time_index_is_specified_but_wrong",
    ],
)
def test_polynomial_decomposer_uses_time_index(
    decomposer_child_class,
    ts_data,
    X_has_time_index,
    X_num_time_columns,
    y_has_time_index,
    time_index_specified,
):
    X, _, y = ts_data()

    time_index_col_name = "date"
    assert isinstance(X.index, pd.DatetimeIndex)
    assert isinstance(y.index, pd.DatetimeIndex)

    # Modify time series data to match testing conditions
    if X_has_time_index == "X_doesnt_have_time_index":
        X = X.ww.reset_index(drop=True)
    if y_has_time_index == "y_doesnt_have_time_index":
        y = y.reset_index(drop=True)
    if X_num_time_columns == 0:
        X = X.ww.drop(columns=[time_index_col_name])
    elif X_num_time_columns > 1:
        for addn_col in range(X_num_time_columns - 1):
            X.ww[time_index_col_name + str(addn_col + 1)] = X.ww[time_index_col_name]
    time_index = {
        "time_index_is_specified": "date",
        "time_index_not_specified": None,
        "time_index_is_specified_but_wrong": "d4t3s",
    }[time_index_specified]
    decomposer = decomposer_child_class(time_index=time_index)

    err_msg = None

    # The time series data has no time data
    if (
        X_num_time_columns == 0
        and X_has_time_index == "X_doesnt_have_time_index"
        and y_has_time_index == "y_doesnt_have_time_index"
    ):
        err_msg = "There are no Datetime features in the feature data and neither the feature nor the target data have a DateTime index."

    # The time series data has too much time data
    elif (
        X_num_time_columns > 1
        and time_index_specified == "time_index_not_specified"
        and y_has_time_index == "y_doesnt_have_time_index"
        and X_has_time_index != "X_has_time_index"
    ):
        err_msg = "Too many Datetime features provided in data but no time_index column specified during __init__."

    # If the wrong time_index column is specified with multiple datetime columns
    elif (
        time_index_specified == "time_index_is_specified_but_wrong"
        and X_num_time_columns > 1
        and X_has_time_index != "X_has_time_index"
        and y_has_time_index != "y_has_time_index"
    ):
        err_msg = "Too many Datetime features provided in data and provided time_index column d4t3s not present in data."

    if err_msg is not None:
        with pytest.raises(
            ValueError,
            match=err_msg,
        ):
            decomposer.fit_transform(X, y)
    else:
        X_t, y_t = decomposer.fit_transform(X, y)

        # If the fit_transform() succeeds, assert the original X and y
        # have unchanged indices.
        if X_has_time_index == "X_doesnt_have_time_index":
            assert not isinstance(X.index, pd.DatetimeIndex)
        else:
            assert isinstance(X.index, pd.DatetimeIndex)
        if y_has_time_index == "y_doesnt_have_time_index":
            assert not isinstance(y.index, pd.DatetimeIndex)
        else:
            assert isinstance(y.index, pd.DatetimeIndex)


@pytest.mark.parametrize(
    "decomposer_child_class",
    [PolynomialDecomposer, STLDecomposer],
)
@pytest.mark.parametrize(
    "dataframe_has_datatime_index",
    ["df_has_datetime_index", "df_doesnt_have_datetime_index"],
)
@pytest.mark.parametrize(
    "multiple_time_features",
    ["has_another_time_feature", "doesnt_have_another_time_feature"],
)
@pytest.mark.parametrize(
    "time_index_exists",
    ["time_index_does_exist", "time_index_does_not_exist"],
)
def test_decomposer_prefers_users_time_index(
    decomposer_child_class,
    dataframe_has_datatime_index,
    multiple_time_features,
    time_index_exists,
    ts_data,
):
    periods = 30
    dates_1 = pd.date_range("2020-01-01", periods=periods)
    dates_2 = pd.date_range("2021-01-01", periods=periods)
    dates_3 = pd.date_range("2022-01-01", periods=periods)
    vals = np.arange(0, periods)
    y = pd.Series(vals)
    X = pd.DataFrame({"values": vals})

    if time_index_exists == "time_index_does_exist":
        X["dates"] = dates_1

    if multiple_time_features == "has_another_time_feature":
        X["more_dates"] = dates_2

    if dataframe_has_datatime_index == "df_has_datetime_index":
        X.set_axis(pd.DatetimeIndex(dates_3))

    pdc = decomposer_child_class(time_index="dates")

    err_msg = None
    if time_index_exists == "time_index_does_not_exist":
        if multiple_time_features == "has_another_time_feature":
            expected_values = dates_2.values
        else:
            err_msg = "There are no Datetime features in the feature data and neither the feature nor the target data have a DateTime index."
    else:
        expected_values = dates_1.values

    if err_msg:
        with pytest.raises(ValueError, match=err_msg):
            X_t, y_t = pdc.fit_transform(X, y)
    else:
        X_t, y_t = pdc.fit_transform(X, y)
        assert all(y_t.index.values == expected_values)
