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


def test_polynomial_decomposer_transform_returns_same_when_y_none(
    ts_data,
):
    X, _, y = ts_data()
    stl = STLDecomposer().fit(X, y)
    X_t, y_t = stl.transform(X, None)
    pd.testing.assert_frame_equal(X, X_t)
    assert y_t is None


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


@pytest.mark.parametrize(
    "transformer_fit_on_data",
    [
        "in-sample",
        "wholly-out-of-sample",
        "wholly-out-of-sample-no-gap",
        "partially-out-of-sample",
        "out-of-sample-in-past",
    ],
)
def test_stl_decomposer_inverse_transform(
    generate_seasonal_data,
    transformer_fit_on_data,
):
    # Generate 10 periods (the default) of synthetic seasonal data
    seasonal_period = 7
    X, y = generate_seasonal_data(real_or_synthetic="synthetic")(
        period=seasonal_period,
        freq_str="D",
        set_time_index=True,
    )
    subset_X = X[: 5 * seasonal_period]
    subset_y = y[: 5 * seasonal_period]

    decomposer = STLDecomposer(seasonal_period=seasonal_period)
    output_X, output_y = decomposer.fit_transform(subset_X, subset_y)
    import datetime

    import matplotlib.pyplot as plt

    def plot_things():
        plt.plot(y)
        plt.plot(y[new_index], "bo")
        plt.plot(output_inverse_y, "rx")
        plt.plot(y_t_new, "gx")
        plt.plot(subset_y, "yx")
        plt.show()

    if transformer_fit_on_data == "in-sample":
        output_inverse_y = decomposer.inverse_transform(output_y)
        pd.testing.assert_series_equal(subset_y, output_inverse_y, check_dtype=False)
    elif transformer_fit_on_data == "wholly-out-of-sample":
        # Re-compose 14-days worth of data with a 7 day gap between end of
        # fit data and start of data to inverse-transform
        delta = datetime.timedelta(days=seasonal_period)
    elif transformer_fit_on_data == "wholly-out-of-sample-no-gap":
        # Re-compose 14-days worth of data with no gap between end of
        # fit data and start of data to inverse-transform
        delta = datetime.timedelta(days=1)
    elif transformer_fit_on_data == "partially-out-of-sample":
        # Re-compose 14-days worth of data overlapping the in and out-of
        # sample data.
        delta = datetime.timedelta(days=-1 * seasonal_period)
    elif transformer_fit_on_data == "out-of-sample-in-past":
        # Re-compose 14-days worth of data both out of sample and in the
        # past.
        delta = datetime.timedelta(days=-12 * seasonal_period)

    if transformer_fit_on_data != "in-sample":
        new_index = pd.date_range(
            subset_y.index[-1] + delta,
            periods=2 * seasonal_period,
            freq="D",
        )
        y_t_new = pd.Series(np.zeros(len(new_index))).set_axis(new_index)
        if transformer_fit_on_data in [
            "partially-out-of-sample",
            "out-of-sample-in-past",
        ]:
            with pytest.raises(
                ValueError,
                match="STLDecomposer cannot recompose/inverse transform data out of sample",
            ):
                output_inverse_y = decomposer.inverse_transform(y_t_new)
        else:
            output_inverse_y = decomposer.inverse_transform(y_t_new)
            pd.testing.assert_series_equal(
                y[new_index],
                output_inverse_y,
                check_exact=False,
                rtol=1.0e-3,
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
def test_stl_decomposer_set_period(period, generate_seasonal_data):
    X, y = generate_seasonal_data(real_or_synthetic="synthetic")(period)
    pdc = STLDecomposer()

    assert pdc.seasonal_period == 7
    assert pdc.parameters["seasonal_period"] == 7

    pdc.set_seasonal_period(X, y)

    assert period - 1 <= pdc.seasonal_period <= period + 1
    assert pdc.parameters["seasonal_period"]


def test_polynomial_decomposer_get_trend_dataframe_raises_errors(ts_data):
    X, _, y = ts_data()
    stl = STLDecomposer()
    stl.fit_transform(X, y)

    with pytest.raises(
        TypeError,
        match="Provided X should have datetimes in the index.",
    ):
        X_int_index = X.reset_index()
        stl.get_trend_dataframe(X_int_index, y)

    with pytest.raises(
        ValueError,
        match="Provided DatetimeIndex of X should have an inferred frequency.",
    ):
        X.index.freq = None
        stl.get_trend_dataframe(X, y)
