import matplotlib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from evalml.pipelines.components import STLDecomposer
from evalml.tests.component_tests.decomposer_tests.test_decomposer import (
    build_test_target,
    get_trend_dataframe_format_correct,
)


def test_stl_decomposer_init():
    decomp = STLDecomposer(degree=3, time_index="dates")
    assert decomp.parameters == {
        "degree": 3,
        "period": None,
        "seasonal_smoother": 7,
        "time_index": "dates",
        "series_id": None,
    }


def test_stl_decomposer_multiseries_init():
    decomp = STLDecomposer(degree=3, time_index="dates", series_id="ids")
    assert decomp.parameters == {
        "degree": 3,
        "period": None,
        "seasonal_smoother": 7,
        "time_index": "dates",
        "series_id": "ids",
    }


def test_stl_decomposer_auto_sets_seasonal_smoother_to_odd():
    stl = STLDecomposer(seasonal_smoother=3)
    assert stl.seasonal_smoother == 3

    stl = STLDecomposer(seasonal_smoother=4)
    assert stl.seasonal_smoother == 5


@pytest.mark.parametrize(
    "variateness",
    [
        "univariate",
        "multivariate",
    ],
)
def test_stl_raises_warning_high_smoother(
    caplog,
    ts_data,
    multiseries_ts_data_unstacked,
    variateness,
):
    if variateness == "univariate":
        X, _, y = ts_data()
    elif variateness == "multivariate":
        X, y = multiseries_ts_data_unstacked
    stl = STLDecomposer(seasonal_smoother=101)
    stl.fit(X, y)
    assert "STLDecomposer may perform poorly" in caplog.text


@pytest.mark.parametrize(
    "period,freq",
    [
        (7, "D"),
        (30, "D"),
        (365, "D"),
        (12, "M"),
        (40, "M"),
    ],
)
def test_stl_sets_determined_period(
    period,
    freq,
    generate_seasonal_data,
):
    X, y = generate_seasonal_data(real_or_synthetic="synthetic")(
        period,
        freq_str=freq,
    )

    stl = STLDecomposer()
    stl.fit(X, y)
    # Allow for a slight margin of error with detection
    assert period * 0.99 <= stl.period <= period * 1.01


@pytest.mark.parametrize(
    "period,freq",
    [
        (7, "D"),  # Weekly season
        (30, "D"),
        pytest.param(
            365,
            "D",
            marks=pytest.mark.xfail(
                reason="STL is less precise with larger periods.",
            ),
        ),
        (12, "M"),  # Annual season
        (4, "M"),  # Quarterly season
    ],
)
@pytest.mark.parametrize("trend_degree", [1, 2, 3])
@pytest.mark.parametrize(
    "variateness",
    [
        "univariate",
        "multivariate",
    ],
)
def test_stl_fit_transform_in_sample(
    period,
    freq,
    trend_degree,
    generate_seasonal_data,
    generate_multiseries_seasonal_data,
    variateness,
):
    if variateness == "univariate":
        X, y = generate_seasonal_data(real_or_synthetic="synthetic")(
            period,
            freq_str=freq,
            trend_degree=trend_degree,
        )
    elif variateness == "multivariate":
        X, y = generate_multiseries_seasonal_data(real_or_synthetic="synthetic")(
            period,
            freq_str=freq,
            trend_degree=trend_degree,
        )

    stl = STLDecomposer(period=period)

    X_t, y_t = stl.fit_transform(X, y)

    if variateness == "univariate":
        # Get the expected answer
        lin_reg = LinearRegression(fit_intercept=True)
        features = PolynomialFeatures(degree=trend_degree).fit_transform(
            np.arange(X.shape[0]).reshape(-1, 1),
        )
        lin_reg.fit(features, y)
        expected_trend = lin_reg.predict(features)

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
    elif variateness == "multivariate":
        # Get the expected answer
        for id in y.columns:
            y_series = y[id]
            lin_reg = LinearRegression(fit_intercept=True)
            features = PolynomialFeatures(degree=trend_degree).fit_transform(
                np.arange(X.shape[0]).reshape(-1, 1),
            )
            lin_reg.fit(features, y_series)
            expected_trend = lin_reg.predict(features)
            # Check the trend to make sure STL worked properly
            pd.testing.assert_series_equal(
                pd.Series(expected_trend),
                pd.Series(stl.decompositions[id]["trend"]),
                check_exact=False,
                check_index=False,
                check_names=False,
                atol=0.3,
            )

    # Verify the X is not changed
    pd.testing.assert_frame_equal(X, X_t)


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
def test_stl_decomposer_inverse_transform(
    index_type,
    generate_seasonal_data,
    generate_multiseries_seasonal_data,
    variateness,
    transformer_fit_on_data,
):
    # Generate 10 periods (the default) of synthetic seasonal data
    period = 7
    if variateness == "univariate":
        X, y = generate_seasonal_data(real_or_synthetic="synthetic")(
            period=period,
            freq_str="D",
            set_time_index=True,
        )
        if index_type == "integer_index":
            y = y.reset_index(drop=True)
        subset_X = X[: 5 * period]
        subset_y = y[: 5 * period]
    elif variateness == "multivariate":
        X, y = generate_multiseries_seasonal_data(real_or_synthetic="synthetic")(
            period=period,
            freq_str="D",
            set_time_index=True,
        )
        if index_type == "integer_index":
            y = y.reset_index(drop=True)
        subset_y = y.loc[y.index[: 5 * period]]

    subset_X = X[: 5 * period]
    decomposer = STLDecomposer(period=period)
    output_X, output_y = decomposer.fit_transform(subset_X, subset_y)

    if transformer_fit_on_data == "in-sample":
        output_inverse_y = decomposer.inverse_transform(output_y)
        if variateness == "univariate":
            pd.testing.assert_series_equal(
                subset_y,
                output_inverse_y,
                check_dtype=False,
            )
        elif variateness == "multivariate":
            pd.testing.assert_frame_equal(
                pd.DataFrame(subset_y),
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
            # Because output_inverse_y.index is int32 and y[y_t_new.index].index is int64 in windows,
            # we need to test the indices equivalence separately.
            output_inverse_y = decomposer.inverse_transform(y_t_new)

            if variateness == "univariate":
                pd.testing.assert_series_equal(
                    y[y_t_new.index],
                    output_inverse_y,
                    check_index=False,
                    rtol=1.0e-2,
                )
            elif variateness == "multivariate":
                pd.testing.assert_frame_equal(
                    pd.DataFrame(y.loc[y_t_new.index]),
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
@pytest.mark.parametrize("fit_before_decompose", [True, False])
def test_stl_decomposer_get_trend_dataframe(
    generate_seasonal_data,
    generate_multiseries_seasonal_data,
    transformer_fit_on_data,
    fit_before_decompose,
    variateness,
):
    period = 7

    if variateness == "univariate":
        X, y = generate_seasonal_data(real_or_synthetic="synthetic")(
            period=period,
            freq_str="D",
            set_time_index=True,
        )
        subset_y = y[: 5 * period]
    elif variateness == "multivariate":
        X, y = generate_multiseries_seasonal_data(real_or_synthetic="synthetic")(
            period=period,
            freq_str="D",
            set_time_index=True,
        )
        subset_y = y.loc[y.index[: 5 * period]]

    subset_X = X[: 5 * period]

    if transformer_fit_on_data == "in-sample":
        dec = STLDecomposer()
        dec.fit(subset_X, subset_y)

        # get_trend_dataframe() is only expected to work with datetime indices

        result_dfs = dec.get_trend_dataframe(subset_X, subset_y)

        if variateness == "univariate":
            assert isinstance(result_dfs, list)
            assert all(isinstance(x, pd.DataFrame) for x in result_dfs)
            assert all(get_trend_dataframe_format_correct(x) for x in result_dfs)
            assert len(result_dfs) == 1
            [get_trend_dataframe_format_correct(x) for x in result_dfs]

        elif variateness == "multivariate":
            assert isinstance(result_dfs, dict)
            assert all(isinstance(result_dfs[x], list) for x in result_dfs)
            assert all(
                all(isinstance(x, pd.DataFrame) for x in result_dfs[df])
                for df in result_dfs
            )
            assert len(result_dfs) == 2
            [
                (get_trend_dataframe_format_correct(x) for x in result_dfs[df])
                for df in result_dfs
            ]

    elif transformer_fit_on_data != "in-sample":
        y_t_new = build_test_target(
            subset_y,
            period,
            transformer_fit_on_data,
            to_test="transform",
        )
        if variateness == "multivariate":
            y_t_new = pd.DataFrame([y_t_new, y_t_new]).T
        dec = STLDecomposer()
        dec.fit(subset_X, subset_y)

        # get_trend_dataframe() is only expected to work with datetime indices

        if transformer_fit_on_data in [
            "out-of-sample-in-past",
            "partially-out-of-sample-in-past",
        ]:
            with pytest.raises(
                ValueError,
                match="STLDecomposer cannot transform/inverse transform data out of sample",
            ):
                result_dfs = dec.get_trend_dataframe(X, y_t_new)

        else:
            result_dfs = dec.get_trend_dataframe(X.loc[y_t_new.index], y_t_new)

            if variateness == "univariate":
                assert isinstance(result_dfs, list)
                assert all(isinstance(x, pd.DataFrame) for x in result_dfs)
                assert all(get_trend_dataframe_format_correct(x) for x in result_dfs)
                assert len(result_dfs) == 1
                [get_trend_dataframe_format_correct(x) for x in result_dfs]
            elif variateness == "multivariate":
                assert isinstance(result_dfs, dict)
                assert all(isinstance(result_dfs[x], list) for x in result_dfs)
                assert all(
                    all(isinstance(x, pd.DataFrame) for x in result_dfs[df])
                    for df in result_dfs
                )
                assert all(
                    (get_trend_dataframe_format_correct(x) for x in result_dfs[df])
                    for df in result_dfs
                )
                assert len(result_dfs) == 2
                [
                    (get_trend_dataframe_format_correct(x) for x in result_dfs[df])
                    for df in result_dfs
                ]


@pytest.mark.parametrize(
    "variateness",
    [
        "univariate",
        "multivariate",
    ],
)
def test_stl_decomposer_get_trend_dataframe_sets_time_index_internally(
    generate_seasonal_data,
    generate_multiseries_seasonal_data,
    variateness,
):
    if variateness == "univariate":
        X, y = generate_seasonal_data(real_or_synthetic="synthetic")(
            period=7,
            set_time_index=False,
        )
    elif variateness == "multivariate":
        X, y = generate_multiseries_seasonal_data(real_or_synthetic="synthetic")(
            period=7,
            set_time_index=False,
        )

    assert not isinstance(y.index, pd.DatetimeIndex)

    stl = STLDecomposer()
    stl.fit(X, y)
    result_dfs = stl.get_trend_dataframe(X, y)

    if variateness == "univariate":
        assert isinstance(result_dfs, list)
        assert all(isinstance(x, pd.DataFrame) for x in result_dfs)
        assert all(get_trend_dataframe_format_correct(x) for x in result_dfs)
    elif variateness == "multivariate":
        assert isinstance(result_dfs, dict)
        assert all(isinstance(result_dfs[x], list) for x in result_dfs)
        assert all(
            all(isinstance(x, pd.DataFrame) for x in result_dfs[df])
            for df in result_dfs
        )


@pytest.mark.parametrize(
    "bad_frequency",
    ["T", "A"],
)
@pytest.mark.parametrize(
    "variateness",
    [
        "univariate",
        "multivariate",
    ],
)
def test_unsupported_frequencies(
    bad_frequency,
    generate_seasonal_data,
    generate_multiseries_seasonal_data,
    variateness,
):
    """This test exists to highlight that even though the underlying statsmodels STL component won't work
    for minute or annual frequencies, we can still run these frequencies with automatic period detection.
    """
    if variateness == "univariate":
        X, y = generate_seasonal_data(real_or_synthetic="synthetic")(
            period=7,
            freq_str=bad_frequency,
        )
    elif variateness == "multivariate":
        X, y = generate_multiseries_seasonal_data(real_or_synthetic="synthetic")(
            period=7,
            freq_str=bad_frequency,
        )
    stl = STLDecomposer()
    X_t, y_t = stl.fit_transform(X, y)
    assert stl.period is not None


@pytest.mark.parametrize(
    "variateness",
    [
        "univariate",
        "multivariate",
    ],
)
def test_stl_decomposer_doesnt_modify_target_index(
    generate_seasonal_data,
    generate_multiseries_seasonal_data,
    variateness,
):
    if variateness == "univariate":
        X, y = generate_seasonal_data(real_or_synthetic="synthetic")(
            period=7,
            set_time_index=False,
        )
    elif variateness == "multivariate":
        X, y = generate_multiseries_seasonal_data(real_or_synthetic="synthetic")(
            period=7,
            set_time_index=False,
        )

    original_X_index = X.index
    original_y_index = y.index

    stl = STLDecomposer()
    stl.fit(X, y)
    pd.testing.assert_index_equal(X.index, original_X_index)
    pd.testing.assert_index_equal(y.index, original_y_index)

    X_t, y_t = stl.transform(X, y)
    pd.testing.assert_index_equal(X_t.index, original_X_index)
    pd.testing.assert_index_equal(y_t.index, original_y_index)

    y_new = stl.inverse_transform(y_t)
    pd.testing.assert_index_equal(y_new.index, original_y_index)


@pytest.mark.parametrize("index_type", ["datetime", "int"])
@pytest.mark.parametrize("set_coverage", [True, False])
@pytest.mark.parametrize(
    "variateness",
    [
        "univariate",
        "multivariate",
    ],
)
def test_stl_decomposer_get_trend_prediction_intervals(
    set_coverage,
    index_type,
    generate_seasonal_data,
    generate_multiseries_seasonal_data,
    variateness,
):
    coverage = [0.75, 0.85, 0.95] if set_coverage else None
    period = 7
    if variateness == "univariate":
        X, y = generate_seasonal_data(real_or_synthetic="synthetic")(
            period=period,
            freq_str="D",
            set_time_index=True,
        )
        y_train = y[: 15 * period]
        y_validate = y[15 * period :]
    elif variateness == "multivariate":
        X, y = generate_multiseries_seasonal_data(real_or_synthetic="synthetic")(
            period=period,
            freq_str="D",
            set_time_index=True,
        )
        y_train = y.loc[y.index[: 15 * period]]
        y_validate = y.loc[y.index[15 * period :]]
    X_train = X[: 15 * period]

    stl = STLDecomposer()
    stl.fit(X_train, y_train)

    def assert_pred_interval_coverage(pred_interval):
        expected_coverage = [0.95] if coverage is None else coverage
        for cover_value in expected_coverage:
            for key in [f"{cover_value}_lower", f"{cover_value}_upper"]:
                pd.testing.assert_index_equal(
                    pred_interval[key].index,
                    y_validate.index,
                )

    # Set y.index to be non-datetime
    if index_type == "int":
        y_validate.index = np.arange(len(X_train), len(y))

    trend_pred_intervals = stl.get_trend_prediction_intervals(
        y_validate,
        coverage=coverage,
    )

    if variateness == "univariate":
        assert_pred_interval_coverage(trend_pred_intervals)
    elif variateness == "multivariate":
        for id in y_validate:
            assert_pred_interval_coverage(trend_pred_intervals[id])


@pytest.mark.parametrize(
    "variateness",
    [
        "univariate",
        "multivariate",
    ],
)
def test_stl_decomposer_plot_decomposition(
    ts_data,
    multiseries_ts_data_unstacked,
    variateness,
):
    if variateness == "univariate":
        X, _, y = ts_data()
    elif variateness == "multivariate":
        X, y = multiseries_ts_data_unstacked
        X.index = X["date"]
        X.index.freq = "D"

    dec = STLDecomposer(time_index="date")
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
