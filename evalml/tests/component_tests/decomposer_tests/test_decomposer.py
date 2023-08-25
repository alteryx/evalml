import itertools
from datetime import datetime, timedelta

import matplotlib
import numpy as np
import pandas as pd
import pytest
import woodwork as ww

import evalml.exceptions
from evalml.pipelines.components.transformers.preprocessing import (
    PolynomialDecomposer,
    STLDecomposer,
)

# All the decomposers to run common tests over.
decomposer_list = [STLDecomposer, PolynomialDecomposer]


def get_trend_dataframe_format_correct(df):
    return set(df.columns) == {"signal", "trend", "seasonality", "residual"}


@pytest.mark.parametrize(
    "decomposer_child_class",
    decomposer_list,
)
def test_set_time_index(decomposer_child_class):
    x = np.arange(0, 2 * np.pi, 0.01)
    dts = pd.date_range(datetime.today(), periods=len(x))
    X = pd.DataFrame({"x": x})
    X = X.set_index(dts)
    y = pd.Series(np.sin(x))

    assert isinstance(y.index, pd.RangeIndex)

    decomposer = decomposer_child_class()
    y_time_index = decomposer._set_time_index(X, y)
    assert isinstance(y_time_index.index, pd.DatetimeIndex)


@pytest.mark.parametrize(
    "decomposer_child_class",
    decomposer_list,
)
def test_decomposer_init_raises_error_if_degree_not_int(decomposer_child_class):
    with pytest.raises(TypeError, match="Received str"):
        decomposer_child_class(degree="1")

    with pytest.raises(TypeError, match="Received float"):
        decomposer_child_class(degree=3.4)

    decomposer_child_class(degree=3.0)


@pytest.mark.parametrize(
    "decomposer_child_class",
    decomposer_list,
)
@pytest.mark.parametrize(
    "y_has_time_index",
    ["y_has_time_index", "y_doesnt_have_time_index"],
)
@pytest.mark.parametrize(
    "variateness",
    [
        "univariate",
        "multivariate",
    ],
)
def test_decomposer_plot_decomposition(
    decomposer_child_class,
    y_has_time_index,
    generate_seasonal_data,
    variateness,
):
    if variateness == "multivariate" and isinstance(
        decomposer_child_class(),
        PolynomialDecomposer,
    ):
        pytest.skip(
            "Skipping Decomposer because multiseries is not implemented for Polynomial Decomposer",
        )

    step = 0.01
    period = 9
    X, y = generate_seasonal_data(
        real_or_synthetic="synthetic",
        univariate_or_multivariate=variateness,
    )(period, step)

    if y_has_time_index == "y_has_time_index":
        y = y.set_axis(X.index)

    dec = decomposer_child_class(degree=1, period=period)
    dec.fit_transform(X, y)

    if variateness == "univariate":
        fig, axs = dec.plot_decomposition(X, y, show=False)
        assert isinstance(fig, matplotlib.pyplot.Figure)
        assert isinstance(axs, np.ndarray)
        assert all([isinstance(ax, matplotlib.pyplot.Axes) for ax in axs])
    elif variateness == "multivariate":
        result_plots = dec.plot_decomposition(X, y, show=False)
        for id in y.columns:
            fig, axs = result_plots[id]
            assert isinstance(fig, matplotlib.pyplot.Figure)
            assert isinstance(axs, np.ndarray)
            assert all([isinstance(ax, matplotlib.pyplot.Axes) for ax in axs])


@pytest.mark.parametrize(
    "decomposer_child_class",
    decomposer_list,
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
@pytest.mark.parametrize(
    "variateness",
    [
        "univariate",
        "multivariate",
    ],
)
def test_decomposer_uses_time_index(
    decomposer_child_class,
    ts_data,
    ts_multiseries_data,
    variateness,
    X_has_time_index,
    X_num_time_columns,
    y_has_time_index,
    time_index_specified,
):
    if variateness == "univariate":
        X, _, y = ts_data()
    elif variateness == "multivariate":
        if isinstance(decomposer_child_class(), PolynomialDecomposer):
            pytest.skip(
                "Skipping Decomposer because multiseries is not implemented for Polynomial Decomposer",
            )
        X, _, y = ts_multiseries_data()

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
    decomposer_list,
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

    dec = decomposer_child_class(time_index="dates")

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
            X_t, y_t = dec.fit_transform(X, y)
    else:
        X_t, y_t = dec.fit_transform(X, y)
        if isinstance(dec, STLDecomposer):
            assert all(dec.trends[0].index.values == expected_values)
        elif isinstance(dec, PolynomialDecomposer):
            assert all(dec.trend.index.values == expected_values)


@pytest.mark.parametrize(
    "decomposer_child_class",
    decomposer_list,
)
@pytest.mark.parametrize(
    "frequency",
    [
        "D",
        "W",
        "S",
        "h",
        "T",
        pytest.param(
            "m",
            marks=pytest.mark.xfail(reason="Frequency considered ambiguous by pandas"),
        ),
        pytest.param(
            "M",
            marks=pytest.mark.xfail(reason="Frequency considered ambiguous by pandas"),
        ),
        pytest.param(
            "Y",
            marks=pytest.mark.xfail(reason="Frequency considered ambiguous by pandas"),
        ),
        pytest.param(
            "MS",
            marks=pytest.mark.xfail(reason="Frequency considered ambiguous by pandas"),
        ),
    ],
)
@pytest.mark.parametrize(
    "test_first_index",
    ["on period", "before period", "just after period", "mid period"],
)
def test_decomposer_build_seasonal_signal(
    decomposer_child_class,
    ts_data,
    test_first_index,
    frequency,
):
    period = 10
    test_first_index = {
        "on period": 3 * period,
        "before period": 3 * period - 1,
        "just after period": 3 * period + 1,
        "mid period": 3 * period + 4,
    }[test_first_index]

    # Data spanning 2021-01-01 to 2021-02-09
    X, _, y = ts_data()

    # Change the date time index to start at the same time but have different frequency
    y = y.set_axis(
        pd.date_range(start="2021-01-01", periods=len(y), freq=frequency),
    )

    decomposer = decomposer_child_class(degree=2)

    # Synthesize a one-week long cyclic signal
    single_period_seasonal_signal = np.sin(y[0:period] * 2 * np.pi / len(y[0:period]))
    full_seasonal_signal = np.sin(y * 2 * np.pi / len(y[0:period]))

    # Split the target data.  Since the period of this data is 7 days, we'll test
    # when the cycle begins, an index before it begins, an index after it begins
    # and in the middle of a cycle
    y_test = y[test_first_index:]

    # Set the decomposer's trend attribute since the function uses it to select the
    # proper integer/datetime index.  The actual value doesn't matter, just that
    # something with an index exists there.
    decomposer.in_sample_datetime_index = full_seasonal_signal.index

    projected_seasonality = decomposer._project_seasonal(
        y_test,
        single_period_seasonal_signal,
        period,
        frequency,
    )

    # Make sure that the function extracted the correct portion of the repeating, full seasonal signal
    assert np.allclose(
        full_seasonal_signal[projected_seasonality.index],
        projected_seasonality,
    )


@pytest.mark.parametrize("has_freq", [True, False])
@pytest.mark.parametrize(
    "test_first_index",
    ["on period", "before period", "just after period", "mid period"],
)
def test_decomposer_projected_seasonality_integer_and_datetime(
    ts_data,
    test_first_index,
    has_freq,
):
    period = 10
    test_first_index = {
        "on period": 3 * period,
        "before period": 3 * period - 1,
        "just after period": 3 * period + 1,
        "mid period": 3 * period + 4,
    }[test_first_index]

    X, _, y = ts_data()

    datetime_index = pd.date_range(start="01-01-2002", periods=len(X), freq="M")
    if not has_freq:
        datetime_index.freq = None

    y_integer = y.set_axis(pd.RangeIndex(len(X)))
    y_datetime = y.set_axis(datetime_index)

    int_decomposer = STLDecomposer()
    int_decomposer.fit(X, y_integer)

    date_decomposer = STLDecomposer()
    date_decomposer.fit(X, y_datetime)

    # Synthesize a one-week long cyclic signal
    single_period_seasonal_signal = np.sin(y[0:period] * 2 * np.pi / len(y[0:period]))

    # Split the target data.  Since the period of this data is 7 days, we'll test
    # when the cycle begins, an index before it begins, an index after it begins
    # and in the middle of a cycle
    y_int_test = y_integer[test_first_index:]
    y_date_test = y_datetime[test_first_index:]

    integer_projected_seasonality = int_decomposer._project_seasonal(
        y_int_test,
        single_period_seasonal_signal,
        period,
        int_decomposer.frequency,
    )
    datetime_projected_seasonality = date_decomposer._project_seasonal(
        y_date_test,
        single_period_seasonal_signal,
        period,
        date_decomposer.frequency,
    )

    # Make sure that the function extracted the correct portion of the repeating, full seasonal signal

    pd.testing.assert_series_equal(
        integer_projected_seasonality,
        datetime_projected_seasonality,
        check_index=False,
    )


@pytest.mark.parametrize(
    "decomposer_child_class",
    decomposer_list,
)
@pytest.mark.parametrize("period", [7, 30, 365])
def test_decomposer_set_period(decomposer_child_class, period, generate_seasonal_data):
    X, y = generate_seasonal_data(real_or_synthetic="synthetic")(period)
    dec = decomposer_child_class()

    if isinstance(dec, STLDecomposer):
        assert dec.period is None
        assert dec.parameters["period"] is None
    elif isinstance(dec, PolynomialDecomposer):
        assert dec.period == -1
        assert dec.parameters["period"] == -1

    dec.set_period(X, y)

    assert 0.95 * period <= dec.period <= 1.05 * period
    assert dec.parameters["period"] == dec.period


@pytest.mark.parametrize(
    "y_logical_type",
    ["Double", "Integer", "Age", "IntegerNullable", "AgeNullable"],
)
@pytest.mark.parametrize(
    "decomposer_child_class",
    decomposer_list,
)
@pytest.mark.parametrize("decomposer_picked_correct_degree", [True, False])
@pytest.mark.parametrize(
    "synthetic_data,trend_degree,period",
    [
        *itertools.product(["synthetic"], [1, 2, 3], [None, 7, 30, 365]),
        ("real", 1, 365),
    ],
)
def test_decomposer_determine_periodicity(
    y_logical_type,
    decomposer_child_class,
    period,
    trend_degree,
    decomposer_picked_correct_degree,
    synthetic_data,
    generate_seasonal_data,
):
    X, y = generate_seasonal_data(real_or_synthetic=synthetic_data)(
        period,
        trend_degree=trend_degree,
    )
    y = ww.init_series(y.astype(int), logical_type=y_logical_type)

    # Test that the seasonality can be determined if trend guess isn't spot on.
    if not decomposer_picked_correct_degree:
        trend_degree = 1 if trend_degree in [2, 3] else 2

    dec = decomposer_child_class(degree=trend_degree, period=period)
    ac = dec.determine_periodicity(X, y)

    if period is None:
        assert ac is None
    else:
        assert 0.95 * period <= ac <= 1.05 * period


@pytest.mark.parametrize(
    "decomposer_child_class",
    decomposer_list,
)
@pytest.mark.parametrize(
    "nullable_ltype",
    ["IntegerNullable", "AgeNullable"],
)
@pytest.mark.parametrize(
    "handle_incompatibility",
    [
        True,
        pytest.param(
            False,
            marks=pytest.mark.xfail(
                condition=int(pd.__version__.split(".")[0]) < 2,
                strict=True,
                raises=AssertionError,
                reason="pandas 1.x does not recognize np.Nan in Float64 subtracted_floats.",
            ),
        ),
    ],
)
def test_decomposer_determine_periodicity_nullable_type_incompatibility(
    decomposer_child_class,
    handle_incompatibility,
    nullable_ltype,
    generate_seasonal_data,
):
    """Testing that the nullable type incompatibility that caused us to add handling for the Decomposer
    is still present in pandas. If this test is causing the test suite to fail
    because the code below no longer raises the expected AssertionError, we should confirm that the nullable
    types now work for our use case and remove the nullable type handling logic from Decomposer.determine_periodicity.
    """
    trend_degree = 2
    period = 7
    X, y = generate_seasonal_data(real_or_synthetic="synthetic")(
        period,
        trend_degree=trend_degree,
    )

    # Convert to Integer, truncating the rest of the value
    y = ww.init_series(y.astype(int), logical_type=nullable_ltype)

    if handle_incompatibility:
        dec = decomposer_child_class(degree=trend_degree, period=period)
        X, y = dec._handle_nullable_types(X, y)

    # Introduce nans like we do in _detrend_on_fly by rolling y
    moving_avg = 10
    y_trend_estimate = y.rolling(moving_avg).mean().dropna()
    subtracted_floats = y - y_trend_estimate

    # Pandas will not recognize the np.NaN value in a Float64 subtracted_floats
    # and will not drop those null values, so calling _handle_nullable_types ensures
    # that we stay in float64 and properly drop the null values
    dropped_nans = subtracted_floats.dropna()
    assert len(dropped_nans) == len(y) - moving_avg + 1


@pytest.mark.parametrize(
    "decomposer_child_class",
    decomposer_list,
)
@pytest.mark.parametrize(
    "variateness",
    [
        "univariate",
        "multivariate",
    ],
)
@pytest.mark.parametrize("fit_before_decompose", [True, False])
def test_decomposer_get_trend_dataframe_error_not_fit(
    decomposer_child_class,
    ts_data,
    ts_multiseries_data,
    variateness,
    fit_before_decompose,
):
    if variateness == "univariate":
        X, _, y = ts_data()
    elif variateness == "multivariate":
        if isinstance(decomposer_child_class(), PolynomialDecomposer):
            pytest.skip(
                "Skipping Decomposer because multiseries is not implemented for Polynomial Decomposer",
            )
        X, _, y = ts_multiseries_data()
    dec = decomposer_child_class(time_index="date")
    if fit_before_decompose:
        dec.fit_transform(X, y)
        dec.get_trend_dataframe(X, y)
    else:
        with pytest.raises(evalml.exceptions.ComponentNotYetFittedError):
            dec.transform(X, y)
        with pytest.raises(evalml.exceptions.ComponentNotYetFittedError):
            dec.get_trend_dataframe(X, y)


@pytest.mark.parametrize(
    "decomposer_child_class",
    decomposer_list,
)
@pytest.mark.parametrize(
    "variateness",
    [
        "univariate",
        "multivariate",
    ],
)
def test_decomposer_transform_returns_same_when_y_none(
    decomposer_child_class,
    ts_data,
    ts_multiseries_data,
    variateness,
):
    if variateness == "univariate":
        X, _, y = ts_data()
    elif variateness == "multivariate":
        if isinstance(decomposer_child_class(), PolynomialDecomposer):
            pytest.skip(
                "Skipping Decomposer because multiseries is not implemented for Polynomial Decomposer",
            )
        X, _, y = ts_multiseries_data()

    dec = decomposer_child_class().fit(X, y)
    X_t, y_t = dec.transform(X, None)
    pd.testing.assert_frame_equal(X, X_t)
    assert y_t is None


@pytest.mark.parametrize(
    "decomposer_child_class",
    decomposer_list,
)
@pytest.mark.parametrize(
    "variateness",
    [
        "univariate",
        "multivariate",
    ],
)
def test_decomposer_raises_value_error_target_is_none(
    decomposer_child_class,
    ts_data,
    ts_multiseries_data,
    variateness,
):
    if variateness == "univariate":
        X, _, y = ts_data()
    elif variateness == "multivariate":
        if isinstance(decomposer_child_class(), PolynomialDecomposer):
            pytest.skip(
                "Skipping Decomposer because multiseries is not implemented for Polynomial Decomposer",
            )
        X, _, y = ts_multiseries_data()

    with pytest.raises(ValueError, match="cannot be None for Decomposer!"):
        decomposer_child_class(degree=3).fit_transform(X, None)

    with pytest.raises(ValueError, match="cannot be None for Decomposer!"):
        decomposer_child_class(degree=3).fit(X, None)

    dec = decomposer_child_class(degree=3).fit(X, y)

    with pytest.raises(ValueError, match="cannot be None for Decomposer!"):
        dec.inverse_transform(None)


@pytest.mark.parametrize(
    "decomposer_child_class",
    decomposer_list,
)
@pytest.mark.parametrize(
    "variateness",
    [
        "univariate",
        "multivariate",
    ],
)
def test_decomposer_bad_target_index(
    decomposer_child_class,
    ts_data,
    ts_multiseries_data,
    variateness,
):
    if variateness == "univariate":
        X, _, y = ts_data()
    elif variateness == "multivariate":
        if isinstance(decomposer_child_class(), PolynomialDecomposer):
            pytest.skip(
                "Skipping Decomposer because multiseries is not implemented for Polynomial Decomposer",
            )
        X, _, y = ts_multiseries_data()

    dec = decomposer_child_class()
    y.index = pd.CategoricalIndex(["cat_index" for x in range(len(y))])
    with pytest.raises(
        ValueError,
        match="doesn't support target data with index of type",
    ):
        dec._choose_proper_index(y)


@pytest.mark.parametrize(
    "decomposer_child_class",
    decomposer_list,
)
@pytest.mark.parametrize(
    "transformer_fit_on_data",
    [
        "in-sample",
        "in-sample-less-than-sample",
        "wholly-out-of-sample",
        "wholly-out-of-sample-no-gap",
        "partially-out-of-sample",
        "out-of-sample-in-past",
        "partially-out-of-sample-in-past",
    ],
)
@pytest.mark.parametrize(
    "variateness",
    [
        "univariate",
        "multivariate",
    ],
)
def test_decomposer_fit_transform_out_of_sample(
    decomposer_child_class,
    variateness,
    generate_seasonal_data,
    transformer_fit_on_data,
):
    if variateness == "multivariate" and isinstance(
        decomposer_child_class(),
        PolynomialDecomposer,
    ):
        pytest.skip(
            "Skipping Decomposer because multiseries is not implemented for Polynomial Decomposer",
        )

    # Generate 10 periods (the default) of synthetic seasonal data
    period = 7
    X, y = generate_seasonal_data(
        real_or_synthetic="synthetic",
        univariate_or_multivariate=variateness,
    )(
        period=period,
        freq_str="D",
        set_time_index=True,
        seasonal_scale=0.05,  # Increasing this value causes the decomposer to miscalculate trend
    )

    subset_y = y.loc[y.index[2 * period : 7 * period]]
    subset_X = X[2 * period : 7 * period]

    decomposer = decomposer_child_class(period=period)
    decomposer.fit(subset_X, subset_y)

    if transformer_fit_on_data == "in-sample":
        output_X, output_y = decomposer.transform(subset_X, subset_y)
        if variateness == "multivariate":
            assert_function = pd.testing.assert_frame_equal
            y_expected = y_expected = pd.DataFrame(
                [np.zeros(len(output_y)), np.zeros(len(output_y))],
            ).T.set_axis(subset_y.index)
        else:
            assert_function = pd.testing.assert_series_equal
            y_expected = pd.Series(np.zeros(len(output_y))).set_axis(subset_y.index)
        assert_function(
            y_expected,
            output_y,
            check_dtype=False,
            check_names=False,
            atol=0.2,
        )

    if transformer_fit_on_data != "in-sample":
        y_new = build_test_target(
            subset_y,
            period,
            transformer_fit_on_data,
            to_test="transform",
        )
        if transformer_fit_on_data in [
            "out-of-sample-in-past",
            "partially-out-of-sample-in-past",
        ]:
            with pytest.raises(
                ValueError,
                match="STLDecomposer cannot transform/inverse transform data out of sample",
            ):
                output_X, output_inverse_y = decomposer.transform(None, y_new)
        else:
            output_X, output_y_t = decomposer.transform(None, y.loc[y_new.index])
            if variateness == "multivariate":
                assert_function = pd.testing.assert_frame_equal
                y_new = pd.DataFrame([y_new, y_new]).T
                y_expected = pd.DataFrame(
                    [np.zeros(len(output_y_t)), np.zeros(len(output_y_t))],
                ).T.set_axis(y_new.index)
            else:
                assert_function = pd.testing.assert_series_equal
                y_expected = pd.Series(np.zeros(len(output_y_t))).set_axis(y_new.index)

            assert_function(
                y_expected,
                output_y_t,
                check_exact=False,
                atol=0.1,  # STLDecomposer is within atol=5.0e-4
            )


@pytest.mark.parametrize(
    "decomposer_child_class",
    decomposer_list,
)
@pytest.mark.parametrize("index_type", ["integer_index", "datetime_index"])
@pytest.mark.parametrize(
    "transformer_fit_on_data",
    [
        "in-sample",
        "in-sample-less-than-sample",
        "wholly-out-of-sample",
        "wholly-out-of-sample-no-gap",
        "partially-out-of-sample",
        "out-of-sample-in-past",
        "partially-out-of-sample-in-past",
    ],
)
@pytest.mark.parametrize(
    "variateness",
    [
        "univariate",
        "multivariate",
    ],
)
def test_decomposer_inverse_transform(
    decomposer_child_class,
    index_type,
    generate_seasonal_data,
    variateness,
    transformer_fit_on_data,
):
    if variateness == "multivariate" and isinstance(
        decomposer_child_class(),
        PolynomialDecomposer,
    ):
        pytest.skip(
            "Skipping Decomposer because multiseries is not implemented for Polynomial Decomposer",
        )

    # Generate 10 periods (the default) of synthetic seasonal data
    period = 7
    X, y = generate_seasonal_data(
        real_or_synthetic="synthetic",
        univariate_or_multivariate=variateness,
    )(
        period=period,
        freq_str="D",
        set_time_index=True,
        seasonal_scale=0.05,  # Increasing this value causes the decomposer to miscalculate trend
    )
    if index_type == "integer_index":
        y = y.reset_index(drop=True)

    subset_X = X[: 5 * period]
    subset_y = y.loc[y.index[: 5 * period]]

    decomposer = decomposer_child_class(period=period)
    output_X, output_y = decomposer.fit_transform(subset_X, subset_y)

    if transformer_fit_on_data == "in-sample":
        output_inverse_y = decomposer.inverse_transform(output_y)
        if variateness == "multivariate":
            assert_function = pd.testing.assert_frame_equal
            y_expected = pd.DataFrame(subset_y)
        else:
            assert_function = pd.testing.assert_series_equal
            y_expected = pd.Series(subset_y)
        assert_function(
            y_expected,
            output_inverse_y,
            check_dtype=False,
        )

    if transformer_fit_on_data != "in-sample":
        y_t_new = build_test_target(
            subset_y,
            period,
            transformer_fit_on_data,
            to_test="inverse_transform",
        )
        if variateness == "multivariate":
            y_t_new = pd.DataFrame([y_t_new, y_t_new]).T
        if transformer_fit_on_data in [
            "out-of-sample-in-past",
            "partially-out-of-sample-in-past",
        ]:
            with pytest.raises(
                ValueError,
                match="STLDecomposer cannot transform/inverse transform data out of sample",
            ):
                output_inverse_y = decomposer.inverse_transform(y_t_new)
        else:
            output_inverse_y = decomposer.inverse_transform(y_t_new)
            # Because output_inverse_y.index is int32 and y[y_t_new.index].index is int64 in windows,
            # we need to test the indices equivalence separately.

            if variateness == "multivariate":
                assert_function = pd.testing.assert_frame_equal
                y_expected = pd.DataFrame(y.loc[y_t_new.index])
            else:
                assert_function = pd.testing.assert_series_equal
                y_expected = pd.Series(y[y_t_new.index])
            assert_function(
                y_expected,
                output_inverse_y,
                check_exact=False,
                rtol=1.0e-1,
            )

            pd.testing.assert_index_equal(
                y.loc[y_t_new.index].index,
                output_inverse_y.index,
                exact=False,
            )


@pytest.mark.parametrize(
    "decomposer_child_class",
    decomposer_list,
)
def test_decomposer_doesnt_modify_target_index(
    decomposer_child_class,
    generate_seasonal_data,
):
    X, y = generate_seasonal_data(real_or_synthetic="synthetic")(
        period=7,
        set_time_index=True,
    )
    original_X_index = X.index
    original_y_index = y.index

    dec = decomposer_child_class()
    dec.fit(X, y)
    pd.testing.assert_index_equal(X.index, original_X_index)
    pd.testing.assert_index_equal(y.index, original_y_index)

    X_t, y_t = dec.transform(X, y)
    pd.testing.assert_index_equal(X_t.index, original_X_index)
    pd.testing.assert_index_equal(y_t.index, original_y_index)

    y_new = dec.inverse_transform(y_t)
    pd.testing.assert_index_equal(y_new.index, original_y_index)


@pytest.mark.parametrize(
    "decomposer_child_class",
    decomposer_list,
)
@pytest.mark.parametrize(
    "variateness",
    [
        "univariate",
        "multivariate",
    ],
)
def test_decomposer_monthly_begin_data(
    decomposer_child_class,
    ts_data,
    ts_multiseries_data,
    variateness,
):
    if variateness == "univariate":
        X, _, y = ts_data()
    elif variateness == "multivariate":
        if isinstance(decomposer_child_class(), PolynomialDecomposer):
            pytest.skip(
                "Skipping Decomposer because multiseries is not implemented for Polynomial Decomposer",
            )
        X, _, y = ts_multiseries_data()

    dts = pd.date_range("01-01-2000", periods=len(X), freq="MS")
    datetime_index = pd.DatetimeIndex(dts)
    X.index = datetime_index
    y.index = datetime_index
    X["date"] = dts
    assert (
        X.index.freqstr == "MS"
    ), "The frequency string that was causing this problem in statsmodels decompose has changed."

    pdc = decomposer_child_class(degree=1, time_index="date")

    if isinstance(pdc, PolynomialDecomposer):
        with pytest.raises(NotImplementedError, match="statsmodels decompose"):
            pdc.fit(X, y)
    else:
        pdc.fit(X, y)


def build_test_target(subset_y, period, transformer_fit_on_data, to_test):
    """Function to build a sample target.  Based on subset_y being daily data containing 5 periods of a periodic signal."""
    if transformer_fit_on_data == "in-sample-less-than-sample":
        # Re-compose 14-days worth of data within, but not spanning the entire sample
        delta = -3
    if transformer_fit_on_data == "wholly-out-of-sample":
        # Re-compose 14-days worth of data with a one period gap between end of
        # fit data and start of data to inverse-transform
        delta = period
    elif transformer_fit_on_data == "wholly-out-of-sample-no-gap":
        # Re-compose 14-days worth of data with no gap between end of
        # fit data and start of data to inverse-transform
        delta = 1
    elif transformer_fit_on_data == "partially-out-of-sample":
        # Re-compose 14-days worth of data overlapping the in and out-of
        # sample data.
        delta = -1
    elif transformer_fit_on_data == "out-of-sample-in-past":
        # Re-compose 14-days worth of data both out of sample and in the
        # past.
        delta = -12
    elif transformer_fit_on_data == "partially-out-of-sample-in-past":
        # Re-compose 14-days worth of data partially out of sample and in the
        # past.
        delta = -6

    if isinstance(subset_y.index, pd.DatetimeIndex):
        delta = timedelta(days=delta * period)

        new_index = pd.date_range(
            subset_y.index[-1] + delta,
            periods=2 * period,
            freq="D",
        )
    else:
        delta = delta * period
        new_index = np.arange(
            subset_y.index[-1] + delta,
            subset_y.index[-1] + delta + 2 * period,
        )

    if to_test == "inverse_transform":
        y_t_new = pd.Series(np.zeros(len(new_index))).set_axis(new_index)
    elif to_test == "transform":
        y_t_new = pd.Series(np.sin([x for x in range(len(new_index))])).set_axis(
            new_index,
        )
    return y_t_new
