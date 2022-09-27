import matplotlib.pyplot
import numpy as np
import pandas as pd
import pytest
import woodwork as ww
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from evalml.pipelines.components import PolynomialDecomposer


def test_polynomial_decomposer_init():
    delayed_features = PolynomialDecomposer(degree=3, time_index="dates")
    assert delayed_features.parameters == {
        "degree": 3,
        "seasonal_period": -1,
        "time_index": "dates",
    }


def test_polynomial_decomposer_init_raises_error_if_degree_not_int():

    with pytest.raises(TypeError, match="Received str"):
        PolynomialDecomposer(degree="1")

    with pytest.raises(TypeError, match="Received float"):
        PolynomialDecomposer(degree=3.4)

    PolynomialDecomposer(degree=3.0)


def test_polynomial_decomposer_raises_value_error_target_is_none(ts_data):
    X, _, y = ts_data()

    with pytest.raises(ValueError, match="y cannot be None for PolynomialDecomposer!"):
        PolynomialDecomposer(degree=3).fit_transform(X, None)

    with pytest.raises(ValueError, match="y cannot be None for PolynomialDecomposer!"):
        PolynomialDecomposer(degree=3).fit(X, None)

    pdt = PolynomialDecomposer(degree=3).fit(X, y)

    with pytest.raises(ValueError, match="y cannot be None for PolynomialDecomposer!"):
        pdt.inverse_transform(None)


def test_polynomial_decomposer_get_trend_dataframe_raises_errors(ts_data):
    X, _, y = ts_data()
    pdt = PolynomialDecomposer(degree=3)
    pdt.fit_transform(X, y)

    with pytest.raises(
        TypeError,
        match="Provided X should have datetimes in the index.",
    ):
        X_int_index = X.reset_index()
        pdt.get_trend_dataframe(X_int_index, y)

    with pytest.raises(
        ValueError,
        match="Provided DatetimeIndex of X should have an inferred frequency.",
    ):
        X.index.freq = None
        pdt.get_trend_dataframe(X, y)


def test_polynomial_decomposer_transform_returns_same_when_y_none(
    ts_data,
):
    X, _, y = ts_data()
    pdc = PolynomialDecomposer().fit(X, y)
    X_t, y_t = pdc.transform(X, None)
    pd.testing.assert_frame_equal(X, X_t)
    assert y_t is None


@pytest.mark.parametrize("index_type", ["datetime", "integer"])
@pytest.mark.parametrize("input_type", ["pd", "ww"])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_polynomial_decomposer_fit_transform(
    degree,
    input_type,
    index_type,
    ts_data,
):

    X, _, y = ts_data()

    # Get the expected answer
    lin_reg = LinearRegression(fit_intercept=True)
    features = PolynomialFeatures(degree=degree).fit_transform(
        np.arange(X.shape[0]).reshape(-1, 1),
    )
    lin_reg.fit(features, y)
    detrended_values = y.values - lin_reg.predict(features)
    expected_index = y.index if input_type != "np" else range(y.shape[0])
    expected_answer = pd.Series(detrended_values, index=expected_index)

    if input_type == "ww":
        X = X.copy()
        X.ww.init()
        y = ww.init_series(y.copy())

    if index_type == "integer":
        y = y.reset_index(drop=True)
        X = X.reset_index(drop=True)
        X.ww.init()

    output_X, output_y = PolynomialDecomposer(degree=degree).fit_transform(X, y)
    pd.testing.assert_series_equal(
        expected_answer,
        output_y,
        check_exact=False,
        rtol=1.9,
    )

    # Verify the X is not changed
    pd.testing.assert_frame_equal(X, output_X)


@pytest.mark.parametrize(
    "variateness",
    [
        "univariate",
        "multivariate",
    ],
)
@pytest.mark.parametrize("fit_before_decompose", [True, False])
@pytest.mark.parametrize("input_type", ["pd", "ww"])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_polynomial_decomposer_get_trend_dataframe(
    degree,
    input_type,
    fit_before_decompose,
    variateness,
    ts_data,
    ts_data_quadratic_trend,
    ts_data_cubic_trend,
):

    if degree == 1:
        X_input, _, y_input = ts_data()
    elif degree == 2:
        X_input, y_input = ts_data_quadratic_trend
    elif degree == 3:
        X_input, y_input = ts_data_cubic_trend

    # Get the expected answer
    lin_reg = LinearRegression(fit_intercept=True)
    features = PolynomialFeatures(degree=degree).fit_transform(
        np.arange(X_input.shape[0]).reshape(-1, 1),
    )
    lin_reg.fit(features, y_input)

    X, y = X_input, y_input

    if input_type == "ww":
        X = X_input.copy()
        X.ww.init()
        y = ww.init_series(y_input.copy())

    pdt = PolynomialDecomposer(degree=degree)
    pdt.fit_transform(X, y)

    # get_trend_dataframe() is only expected to work with datetime indices
    if variateness == "multivariate":
        y = pd.concat([y, y], axis=1)
    result_dfs = pdt.get_trend_dataframe(X, y)

    def get_trend_dataframe_format_correct(df):
        return set(df.columns) == {"signal", "trend", "seasonality", "residual"}

    def get_trend_dataframe_values_correct(df, y):
        np.testing.assert_array_almost_equal(
            (df["trend"] + df["seasonality"] + df["residual"]).values,
            y.values,
        )

    assert isinstance(result_dfs, list)
    assert all(isinstance(x, pd.DataFrame) for x in result_dfs)
    assert all(get_trend_dataframe_format_correct(x) for x in result_dfs)
    if variateness == "univariate":
        assert len(result_dfs) == 1
        [get_trend_dataframe_values_correct(x, y) for x in result_dfs]
    elif variateness == "multivariate":
        assert len(result_dfs) == 2
        [
            get_trend_dataframe_values_correct(x, y[idx])
            for idx, x in enumerate(result_dfs)
        ]


@pytest.mark.parametrize("fit_before_decompose", [True, False])
def test_polynomial_decomposer_get_trend_dataframe_error_not_fit(
    ts_data,
    fit_before_decompose,
):
    X, _, y = ts_data()

    pdt = PolynomialDecomposer(degree=3)
    if fit_before_decompose:
        pdt.fit_transform(X, y)
        pdt.get_trend_dataframe(X, y)
    else:
        with pytest.raises(ValueError):
            pdt.get_trend_dataframe(X, y)


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_polynomial_decomposer_inverse_transform(degree, ts_data):
    X, _, y = ts_data()

    decomposer = PolynomialDecomposer(degree=degree)
    output_X, output_y = decomposer.fit_transform(X, y)
    output_inverse_y = decomposer.inverse_transform(output_y)
    pd.testing.assert_series_equal(y, output_inverse_y, check_dtype=False)


def test_polynomial_decomposer_needs_monotonic_index(ts_data):
    X, _, y = ts_data()
    decomposer = PolynomialDecomposer(degree=2)

    with pytest.raises(Exception) as exec_info:
        y_shuffled = y.sample(frac=1, replace=False)
        decomposer.fit_transform(X, y_shuffled)
    expected_errors = ["monotonically", "X must be in an sktime compatible format"]
    assert any([error in str(exec_info.value) for error in expected_errors])


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
    ],
)
@pytest.mark.parametrize(
    "test_first_index",
    ["on period", "before period", "just after period", "mid period"],
)
def test_polynomial_decomposer_build_seasonal_signal(
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
    y.set_axis(
        pd.date_range(start="2021-01-01", periods=len(y), freq=frequency),
        inplace=True,
    )

    decomposer = PolynomialDecomposer(degree=2)

    # Synthesize a one-week long cyclic signal
    single_period_seasonal_signal = np.sin(y[0:period] * 2 * np.pi / len(y[0:period]))
    full_seasonal_signal = np.sin(y * 2 * np.pi / len(y[0:period]))

    # Split the target data.  Since the period of this data is 7 days, we'll test
    # when the cycle begins, an index before it begins, an index after it begins
    # and in the middle of a cycle
    y_test = y[test_first_index:]

    projected_seasonality = decomposer._build_seasonal_signal(
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
    decomposer = PolynomialDecomposer(time_index=time_index)

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
    "y_has_time_index",
    ["y_has_time_index", "y_doesnt_have_time_index"],
)
def test_polynomial_decomposer_plot_decomposition(
    y_has_time_index,
    generate_seasonal_data,
):
    step = 0.01
    period = 9
    X, y = generate_seasonal_data(real_or_synthetic="synthetic")(period, step)
    if y_has_time_index == "y_has_time_index":
        y = y.set_axis(X.index)

    pdc = PolynomialDecomposer(degree=1, seasonal_period=period)
    pdc.fit_transform(X, y)
    fig, axs = pdc.plot_decomposition(X, y, show=False)
    assert isinstance(fig, matplotlib.pyplot.Figure)
    assert isinstance(axs, np.ndarray)
    assert all([isinstance(ax, matplotlib.pyplot.Axes) for ax in axs])


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
def test_polynomial_decomposer_prefers_users_time_index(
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

    pdc = PolynomialDecomposer(time_index="dates")

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


@pytest.mark.parametrize(
    "periodicity_determination_method",
    [
        "autocorrelation",
        pytest.param(
            "partial-autocorrelation",
            marks=pytest.mark.xfail(reason="Partial Autocorrelation not working yet."),
        ),
    ],
)
@pytest.mark.parametrize(
    "period",
    [
        7,
        30,
        365,
        pytest.param(
            None,
            marks=pytest.mark.xfail(
                reason="Don't have a good heuristic to distinguish bad period guess.",
            ),
        ),
    ],
)
@pytest.mark.parametrize("trend_degree", [1, 2, 3])
@pytest.mark.parametrize("decomposer_picked_correct_degree", [True, False])
@pytest.mark.parametrize("synthetic_data", ["synthetic", "real"])
def test_polynomial_decomposer_determine_periodicity(
    period,
    trend_degree,
    decomposer_picked_correct_degree,
    periodicity_determination_method,
    synthetic_data,
    generate_seasonal_data,
):

    X, y = generate_seasonal_data(real_or_synthetic=synthetic_data)(
        period,
        trend_degree=trend_degree,
    )

    if period is None:
        component_period = 1
    else:
        component_period = period

    # Test that the seasonality can be determined if trend guess isn't spot on.
    if not decomposer_picked_correct_degree:
        trend_degree = 1 if trend_degree in [2, 3] else 2

    pdc = PolynomialDecomposer(degree=trend_degree, seasonal_period=component_period)
    ac = pdc.determine_periodicity(X, y, method=periodicity_determination_method)

    if synthetic_data == "synthetic":
        if period is None:
            assert ac is None
        else:
            assert period - 1 <= ac <= period + 1
    else:
        if period is None:
            assert ac is None
        else:
            assert 360 < ac < 370


@pytest.mark.parametrize("period", [7, 30, 365])
def test_polynomial_decomposer_set_period(period, generate_seasonal_data):
    X, y = generate_seasonal_data(real_or_synthetic="synthetic")(period)
    pdc = PolynomialDecomposer()

    assert pdc.seasonal_period == -1
    assert pdc.parameters["seasonal_period"] == -1

    pdc.set_seasonal_period(X, y)

    assert period - 1 <= pdc.seasonal_period <= period + 1
    assert pdc.parameters["seasonal_period"]
