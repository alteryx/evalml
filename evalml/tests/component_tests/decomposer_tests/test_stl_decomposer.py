import datetime

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from evalml.pipelines.components import STLDecomposer


def test_stl_decomposer_init():
    delayed_features = STLDecomposer(degree=3, time_index="dates")
    assert delayed_features.parameters == {
        "degree": 3,
        "seasonal_period": 7,
        "time_index": "dates",
    }


def test_stl_decomposer_init_raises_error_if_degree_not_int():

    with pytest.raises(TypeError, match="Received str"):
        STLDecomposer(degree="1")

    with pytest.raises(TypeError, match="Received float"):
        STLDecomposer(degree=3.4)

    STLDecomposer(degree=3.0)


def test_stl_decomposer_auto_sets_seasonal_period_to_odd(ts_data):
    X, _, y = ts_data()

    stl = STLDecomposer(seasonal_period=3)
    assert stl.seasonal_period == 3

    stl = STLDecomposer(seasonal_period=4)
    assert stl.seasonal_period == 5


def build_test_target(subset_y, seasonal_period, transformer_fit_on_data, to_test):
    if transformer_fit_on_data == "in-sample-less-than-sample":
        # Re-compose 14-days worth of data within, but not spanning the entire sample
        delta = datetime.timedelta(days=-3 * seasonal_period)
    if transformer_fit_on_data == "wholly-out-of-sample":
        # Re-compose 14-days worth of data with a one period gap between end of
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

    new_index = pd.date_range(
        subset_y.index[-1] + delta,
        periods=2 * seasonal_period,
        freq="D",
    )
    if to_test == "inverse_transform":
        y_t_new = pd.Series(np.zeros(len(new_index))).set_axis(new_index)
    elif to_test == "transform":
        y_t_new = pd.Series(np.sin([x for x in range(len(new_index))])).set_axis(
            new_index,
        )
    return y_t_new


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
def test_stl_fit_transform_in_sample(
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

    stl = STLDecomposer(seasonal_period=period)

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


@pytest.mark.parametrize(
    "transformer_fit_on_data",
    [
        "in-sample",
        "in-sample-less-than-sample",
        "wholly-out-of-sample",
        "wholly-out-of-sample-no-gap",
        "partially-out-of-sample",
        "out-of-sample-in-past",
    ],
)
def test_stl_decomposer_fit_transform_out_of_sample(
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
    decomposer.fit(subset_X, subset_y)

    if transformer_fit_on_data == "in-sample":
        output_X, output_y = decomposer.transform(subset_X, subset_y)
        pd.testing.assert_series_equal(
            pd.Series(np.zeros(len(output_y))).set_axis(subset_y.index),
            output_y,
            check_dtype=False,
            check_names=False,
        )

    if transformer_fit_on_data != "in-sample":
        y_new = build_test_target(
            subset_y,
            seasonal_period,
            transformer_fit_on_data,
            to_test="transform",
        )
        if transformer_fit_on_data in [
            "out-of-sample-in-past",
        ]:
            with pytest.raises(
                ValueError,
                match="STLDecomposer cannot transform/inverse transform data out of sample",
            ):
                output_X, output_inverse_y = decomposer.transform(None, y_new)
        else:
            output_X, output_y_t = decomposer.transform(None, y[y_new.index])

            pd.testing.assert_series_equal(
                pd.Series(np.zeros(len(output_y_t))).set_axis(y_new.index),
                output_y_t,
                check_exact=False,
                atol=1.0e-4,
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

    if transformer_fit_on_data == "in-sample":
        output_inverse_y = decomposer.inverse_transform(output_y)
        pd.testing.assert_series_equal(subset_y, output_inverse_y, check_dtype=False)

    if transformer_fit_on_data != "in-sample":
        y_t_new = build_test_target(
            subset_y,
            seasonal_period,
            transformer_fit_on_data,
            to_test="inverse_transform",
        )
        if transformer_fit_on_data in [
            "out-of-sample-in-past",
        ]:
            with pytest.raises(
                ValueError,
                match="STLDecomposer cannot transform/inverse transform data out of sample",
            ):
                output_inverse_y = decomposer.inverse_transform(y_t_new)
        else:
            output_inverse_y = decomposer.inverse_transform(y_t_new)
            pd.testing.assert_series_equal(
                y[y_t_new.index],
                output_inverse_y,
                check_exact=False,
                rtol=1.0e-3,
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
    ],
)
@pytest.mark.parametrize(
    "variateness",
    [
        "univariate",
        "multivariate",
    ],
)
@pytest.mark.parametrize("fit_before_decompose", [True, False])
def test_stl_decomposer_get_trend_dataframe(
    generate_seasonal_data,
    transformer_fit_on_data,
    fit_before_decompose,
    variateness,
):
    def get_trend_dataframe_format_correct(df):
        return set(df.columns) == {"signal", "trend", "seasonality", "residual"}

    seasonal_period = 7
    X, y = generate_seasonal_data(real_or_synthetic="synthetic")(
        period=seasonal_period,
        freq_str="D",
        set_time_index=True,
    )
    subset_X = X[: 5 * seasonal_period]
    subset_y = y[: 5 * seasonal_period]

    if transformer_fit_on_data == "in-sample":
        dec = STLDecomposer()
        dec.fit(subset_X, subset_y)

        # get_trend_dataframe() is only expected to work with datetime indices
        if variateness == "multivariate":
            subset_y = pd.concat([subset_y, subset_y], axis=1)

        result_dfs = dec.get_trend_dataframe(subset_X, subset_y)

        assert isinstance(result_dfs, list)
        assert all(isinstance(x, pd.DataFrame) for x in result_dfs)
        assert all(get_trend_dataframe_format_correct(x) for x in result_dfs)
        if variateness == "univariate":
            assert len(result_dfs) == 1
            [get_trend_dataframe_format_correct(x) for x in result_dfs]
        elif variateness == "multivariate":
            assert len(result_dfs) == 2
            [get_trend_dataframe_format_correct(x) for idx, x in enumerate(result_dfs)]

    elif transformer_fit_on_data != "in-sample":

        y_t_new = build_test_target(
            subset_y,
            seasonal_period,
            transformer_fit_on_data,
            to_test="transform",
        )
        dec = STLDecomposer()
        dec.fit(subset_X, subset_y)

        # get_trend_dataframe() is only expected to work with datetime indices
        if variateness == "multivariate":
            y_t_new = pd.concat([y_t_new, y_t_new], axis=1)

        if transformer_fit_on_data in [
            "out-of-sample-in-past",
        ]:
            with pytest.raises(
                ValueError,
                match="STLDecomposer cannot transform/inverse transform data out of sample",
            ):
                result_dfs = dec.get_trend_dataframe(X, y_t_new)

        else:
            result_dfs = dec.get_trend_dataframe(X.loc[y_t_new.index], y_t_new)

            assert isinstance(result_dfs, list)
            assert all(isinstance(x, pd.DataFrame) for x in result_dfs)
            assert all(get_trend_dataframe_format_correct(x) for x in result_dfs)
            if variateness == "univariate":
                assert len(result_dfs) == 1
                [get_trend_dataframe_format_correct(x) for x in result_dfs]
            elif variateness == "multivariate":
                assert len(result_dfs) == 2
                [
                    get_trend_dataframe_format_correct(x)
                    for idx, x in enumerate(result_dfs)
                ]


def test_stl_decomposer_get_trend_dataframe_sets_time_index_internally(
    generate_seasonal_data,
):
    def get_trend_dataframe_format_correct(df):
        return set(df.columns) == {"signal", "trend", "seasonality", "residual"}

    X, y = generate_seasonal_data(real_or_synthetic="synthetic")(
        period=7,
        set_time_index=False,
    )
    assert not isinstance(y.index, pd.DatetimeIndex)

    stl = STLDecomposer()
    stl.fit(X, y)
    result_dfs = stl.get_trend_dataframe(X, y)

    assert isinstance(result_dfs, list)
    assert all(isinstance(x, pd.DataFrame) for x in result_dfs)
    assert all(get_trend_dataframe_format_correct(x) for x in result_dfs)
