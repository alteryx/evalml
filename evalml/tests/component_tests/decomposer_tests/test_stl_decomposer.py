import datetime

import matplotlib.pyplot
import numpy as np
import pandas as pd
import pytest
import woodwork as ww
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from evalml.pipelines.components import STLDecomposer


def test_stl_decomposer_init():
    delayed_features = STLDecomposer(degree=3, time_index="dates")
    assert delayed_features.parameters == {
        "degree": 3,
        "seasonal_period": 3,
        "time_index": "dates",
    }


def test_stl_decomposer_init_raises_error_if_degree_not_int():

    with pytest.raises(TypeError, match="Received str"):
        STLDecomposer(degree="1")

    with pytest.raises(TypeError, match="Received float"):
        STLDecomposer(degree=3.4)

    STLDecomposer(degree=3.0)


def test_stl_decomposer_raises_value_error_target_is_none(ts_data):
    X, _, y = ts_data()

    with pytest.raises(ValueError, match="y cannot be None for STLDecomposer!"):
        STLDecomposer(degree=3).fit_transform(X, None)

    with pytest.raises(ValueError, match="y cannot be None for STLDecomposer!"):
        STLDecomposer(degree=3).fit(X, None)

    pdt = STLDecomposer(degree=3).fit(X, y)

    with pytest.raises(ValueError, match="y cannot be None for STLDecomposer!"):
        pdt.inverse_transform(None)


# def test_polynomial_decomposer_get_trend_dataframe_raises_errors(ts_data):
#     X, _, y = ts_data()
#     stl = STLDecomposer()
#     stl.fit_transform(X, y)
#
#     with pytest.raises(
#         TypeError,
#         match="Provided X should have datetimes in the index.",
#     ):
#         X_int_index = X.reset_index()
#         pdt.get_trend_dataframe(X_int_index, y)
#
#     with pytest.raises(
#         ValueError,
#         match="Provided DatetimeIndex of X should have an inferred frequency.",
#     ):
#         X.index.freq = None
#         pdt.get_trend_dataframe(X, y)


def test_polynomial_decomposer_transform_returns_same_when_y_none(
    ts_data,
):
    X, _, y = ts_data()
    stl = STLDecomposer().fit(X, y)
    X_t, y_t = stl.transform(X, None)
    pd.testing.assert_frame_equal(X, X_t)
    assert y_t is None


@pytest.mark.parametrize("index_type", ["datetime", "integer"])
@pytest.mark.parametrize("input_type", ["pd", "ww"])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_stl_decomposer_fit_transform(
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

    stl = STLDecomposer(degree=degree)
    output_X, output_y = stl.fit_transform(X, y)

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(4)
    fig.set_size_inches(18.5, 14.5)
    axs[0].plot(y, "r")
    axs[0].set_title("signal")
    axs[1].plot(stl.trend, "b")
    axs[1].set_title("trend")
    axs[2].plot(stl.seasonal, "g")
    axs[2].set_title("seasonality")
    axs[3].plot(stl.residual, "y")
    axs[3].set_title("residual")
    plt.show()

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


@pytest.mark.parametrize("transformer_fit_on_data", ["in-sample", "out-of-sample"])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_stl_decomposer_inverse_transform(
    degree,
    generate_seasonal_data,
    transformer_fit_on_data,
):
    X, y = generate_seasonal_data(real_or_synthetic="synthetic")(
        period=7,
        freq_str="D",
        set_time_index=True,
    )

    decomposer = STLDecomposer(degree=degree, seasonal_period=7)
    output_X, output_y = decomposer.fit_transform(X, y)
    import matplotlib.pyplot as plt

    if transformer_fit_on_data == "in-sample":
        output_inverse_y = decomposer.inverse_transform(output_y)
    elif transformer_fit_on_data == "out-of-sample":
        import datetime

        delta = datetime.timedelta(days=7)
        new_index = y.index + delta
        y_t_new = pd.Series(np.zeros(len(y))).set_axis(new_index)
        output_inverse_y = decomposer.inverse_transform(y_t_new)
        plt.plot(output_inverse_y)
        plt.show()
        print("hi")
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


@pytest.mark.parametrize(
    "period,freq",
    [
        (7, "D"),  # Weekly season
        pytest.param(
            30,
            "D",
            marks=pytest.mark.xfail(
                reason="STL doesn't perform well on seasonal data with high periods.",
            ),
        ),
        pytest.param(
            365,
            "D",
            marks=pytest.mark.xfail(
                reason="STL doesn't perform well on seasonal data with high periods.",
            ),
        ),
        (12, "M"),  # Annual season
        (4, "M"),  # Quarterly season
    ],
)
@pytest.mark.parametrize("trend_degree", [1, 2, 3])
@pytest.mark.parametrize("synthetic_data", ["synthetic"])  # , "real"])
def test_stl_fit_transform(
    period,
    freq,
    trend_degree,
    synthetic_data,
    generate_seasonal_data,
):

    X, y = generate_seasonal_data(real_or_synthetic=synthetic_data)(
        period,
        freq_str=freq,
        trend_degree=trend_degree,
    )

    # Get the expected answer
    lin_reg = LinearRegression(fit_intercept=True)
    features = PolynomialFeatures(degree=trend_degree).fit_transform(
        np.arange(X.shape[0]).reshape(-1, 1),
    )
    lin_reg.fit(features, y)
    expected_trend = lin_reg.predict(features)
    detrended_values = y.values - expected_trend
    expected_answer = pd.Series(detrended_values)

    if period is None:
        component_period = 1
    else:
        component_period = period

    stl = STLDecomposer(seasonal_period=component_period)

    X_t, y_t = stl.fit_transform(X, y)

    # Check to make sure STL detrended/deseasoned
    pd.testing.assert_series_equal(
        pd.Series(np.zeros(len(y_t))),
        y_t,
        check_exact=False,
        check_index=False,
        check_names=False,
        atol=0.1,
    )

    # Check the trend to make sure STL worked properly
    pd.testing.assert_series_equal(
        pd.Series(expected_trend),
        pd.Series(stl.trend),
        check_exact=False,
        check_index=False,
        check_names=False,
        atol=0.3,
    )

    # Verify the X is not changed
    pd.testing.assert_frame_equal(X, X_t)


@pytest.mark.parametrize("period", [7, 30, 365])
def test_polynomial_decomposer_set_period(period, generate_seasonal_data):
    X, y = generate_seasonal_data(real_or_synthetic="synthetic")(period)
    pdc = PolynomialDecomposer()

    assert pdc.seasonal_period == -1
    assert pdc.parameters["seasonal_period"] == -1

    pdc.set_seasonal_period(X, y)

    assert period - 1 <= pdc.seasonal_period <= period + 1
    assert pdc.parameters["seasonal_period"]


def test_thing():
    import matplotlib.pyplot as plt
    from statsmodels.datasets import elec_equip as ds
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.forecasting.stl import STLForecast

    elec_equip = ds.load().data
    elec_equip.index.freq = elec_equip.index.inferred_freq
    stlf = STLForecast(elec_equip, ARIMA, model_kwargs=dict(order=(1, 1, 0), trend="t"))
    stlf_res = stlf.fit()

    forecast = stlf_res.forecast(24)
    plt.plot(elec_equip)
    plt.plot(forecast)

    seasonal = stlf_res._result.seasonal
    trend = stlf_res._result.trend
    plt.plot(seasonal)
    plt.plot(trend)
    plt.plot(forecast)
    plt.show()
