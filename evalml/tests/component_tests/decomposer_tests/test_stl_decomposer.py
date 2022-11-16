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
    delayed_features = STLDecomposer(degree=3, time_index="dates")
    assert delayed_features.parameters == {
        "degree": 3,
        "seasonal_period": 7,
        "time_index": "dates",
    }


def test_stl_decomposer_auto_sets_seasonal_period_to_odd(ts_data):
    X, _, y = ts_data()

    stl = STLDecomposer(seasonal_period=3)
    assert stl.seasonal_period == 3

    stl = STLDecomposer(seasonal_period=4)
    assert stl.seasonal_period == 5


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
@pytest.mark.parametrize("synthetic_data", ["synthetic"])
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
def test_stl_decomposer_inverse_transform(
    index_type,
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
    if index_type == "integer_index":
        y = y.reset_index(drop=True)
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
            "partially-out-of-sample-in-past",
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
                rtol=1.0e-2,
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
    transformer_fit_on_data,
    fit_before_decompose,
    variateness,
):

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
            "partially-out-of-sample-in-past",
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


@pytest.mark.parametrize(
    "bad_frequency",
    [
        pytest.param(
            "T",
            marks=pytest.mark.xfail(
                reason="statsmodels freq_to_period doesn't support this frequency",
            ),
        ),
        pytest.param(
            "A",
            marks=pytest.mark.xfail(
                reason="statsmodels freq_to_period doesn't support this frequency",
            ),
        ),
    ],
)
def test_unsupported_frequencies(
    bad_frequency,
    generate_seasonal_data,
):
    """This test exists to highlight that the underlying statsmodels STL component won't work for minute or annual frequencies."""
    X, y = generate_seasonal_data(real_or_synthetic="synthetic")(
        period=7,
        freq_str=bad_frequency,
    )

    stl = STLDecomposer()
    X_t, y_t = stl.fit_transform(X, y)


def test_stl_decomposer_doesnt_modify_target_index(
    generate_seasonal_data,
):
    X, y = generate_seasonal_data(real_or_synthetic="synthetic")(
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
