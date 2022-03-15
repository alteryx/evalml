import numpy as np
import pandas as pd
import pytest

from woodwork.statistics_utils.frequency_inference import infer_frequency

from evalml.pipelines import TimeSeriesRegularizer


def get_df(dates):
    reg_X = pd.DataFrame()

    reg_X["dates"] = dates
    reg_X["ints"] = [int(i) for i in range(len(dates))]
    reg_X["doubles"] = [i / 0.25 ** (i / 100) for i in range(len(dates))]
    reg_X["bools"] = [bool(min(1, i % 3)) for i in range(len(dates))]
    reg_X["cats"] = np.random.choice(
        ["here", "there", "somewhere", "everywhere"], len(dates)
    )

    reg_y = pd.Series([i for i in range(len(dates))])

    return reg_X, reg_y


def assert_features_and_length_equal(X, y, X_out, y_out, error_dict):
    length_mismatch = (
        len(error_dict["duplicate"])
        + len(error_dict["extra"])
        - len(error_dict["missing"])
        + len(error_dict["nan"])
    )
    assert len(X) == len(X_out) + length_mismatch
    assert len(y) == len(y_out) + length_mismatch
    # Randomly test 5 shared dates in the output dataframe and make sure their features match those of the same dates
    # in the input dataframe
    non_nan_X = X_out.dropna()
    ignore_dates = set()
    for misaligned in error_dict[
        "misaligned"
    ].values():  # Don't include misaligned dates since they won't match
        ignore_dates.add(misaligned["correct"])
    dates_to_test = set(non_nan_X["dates"]) - ignore_dates
    rand_date = np.random.choice(list(dates_to_test), 5, replace=False)
    for each_date in rand_date:
        input_feat = X.loc[X["dates"] == each_date, set(X.columns) - {"dates"}].iloc[0]
        outout_feat = non_nan_X.loc[
            non_nan_X["dates"] == each_date, set(non_nan_X.columns) - {"dates"}
        ].iloc[0]
        pd.testing.assert_series_equal(
            pd.Series(input_feat.values), pd.Series(outout_feat.values)
        )


def test_ts_regularizer_no_time_index():
    dates_1 = pd.date_range("1/1/21", periods=10)
    dates_2 = pd.date_range("1/13/21", periods=10, freq="2D")
    dates = dates_1.append(dates_2)

    X, y = get_df(dates)

    ts_regularizer = TimeSeriesRegularizer()
    with pytest.raises(
        ValueError,
        match="The argument time_index cannot be None!",
    ):
        ts_regularizer.fit(X, y)


def test_ts_regularizer_mismatch_target_length():
    dates_1 = pd.date_range("1/1/21", periods=10)
    dates_2 = pd.date_range("1/13/21", periods=10, freq="2D")
    dates = dates_1.append(dates_2)

    X, _ = get_df(dates)
    y = pd.Series([i for i in range(25)])

    ts_regularizer = TimeSeriesRegularizer(time_index="dates")
    with pytest.raises(
        ValueError,
        match="If y has been passed, then it must be the same length as X.",
    ):
        ts_regularizer.fit(X, y)


def test_ts_regularizer_no_freq():
    dates_1 = pd.date_range("1/1/21", periods=10)
    dates_2 = pd.date_range("1/13/21", periods=10, freq="2D")
    dates = dates_1.append(dates_2)

    X, y = get_df(dates)

    ts_regularizer = TimeSeriesRegularizer(time_index="dates")
    with pytest.raises(
        ValueError,
        match="The column dates does not have a frequency that can be inferred.",
    ):
        ts_regularizer.fit(X, y)


@pytest.mark.parametrize(
    "dataset", ["beginning", "middle", "end", "scattered", "continuous"]
)
def test_ts_regularizer_duplicate(
    dataset,
    duplicate_beginning,
    duplicate_middle,
    duplicate_end,
    duplicate_scattered,
    duplicate_continuous,
):

    if dataset == "beginning":
        dates = duplicate_beginning
    elif dataset == "middle":
        dates = duplicate_middle
    elif dataset == "end":
        dates = duplicate_end
    elif dataset == "scattered":
        dates = duplicate_scattered
    else:
        dates = duplicate_continuous

    X, y = get_df(dates)

    ww_payload = infer_frequency(X["dates"], debug=True)

    ts_regularizer = TimeSeriesRegularizer(time_index="dates")
    ts_regularizer.fit(X, y)
    X_out, y_out = ts_regularizer.transform(X, y)

    assert pd.infer_freq(X_out["dates"]) == ww_payload[1]["estimated_freq"]

    error_dict = ts_regularizer.error_dict
    assert_features_and_length_equal(X, y, X_out, y_out, error_dict)


@pytest.mark.parametrize(
    "dataset", ["beginning", "middle", "end", "scattered", "continuous"]
)
def test_ts_regularizer_missing(
    dataset,
    missing_beginning,
    missing_middle,
    missing_end,
    missing_scattered,
    missing_continuous,
):

    if dataset == "beginning":
        dates = missing_beginning
    elif dataset == "middle":
        dates = missing_middle
    elif dataset == "end":
        dates = missing_end
    elif dataset == "scattered":
        dates = missing_scattered
    else:
        dates = missing_continuous

    X, y = get_df(dates)

    ww_payload = infer_frequency(X["dates"], debug=True)

    ts_regularizer = TimeSeriesRegularizer(time_index="dates")
    ts_regularizer.fit(X, y)
    X_out, y_out = ts_regularizer.transform(X, y)

    assert pd.infer_freq(X_out["dates"]) == ww_payload[1]["estimated_freq"]

    error_dict = ts_regularizer.error_dict
    assert_features_and_length_equal(X, y, X_out, y_out, error_dict)


@pytest.mark.parametrize(
    "dataset", ["beginning", "middle", "end", "scattered", "continuous"]
)
def test_ts_regularizer_uneven(
    dataset,
    uneven_beginning,
    uneven_middle,
    uneven_end,
    uneven_scattered,
    uneven_continuous,
):

    if dataset == "beginning":
        dates = uneven_beginning
    elif dataset == "middle":
        dates = uneven_middle
    elif dataset == "end":
        dates = uneven_end
    elif dataset == "scattered":
        dates = uneven_scattered
    else:
        dates = uneven_continuous

    X, y = get_df(dates)

    ww_payload = infer_frequency(X["dates"], debug=True)

    ts_regularizer = TimeSeriesRegularizer(time_index="dates")
    ts_regularizer.fit(X, y)
    X_out, y_out = ts_regularizer.transform(X, y)

    assert pd.infer_freq(X_out["dates"]) == ww_payload[1]["estimated_freq"]
    if dataset == "beginning":
        assert X.iloc[0]["dates"] not in X_out["dates"]
        assert y.iloc[0] not in y_out.values
    elif dataset == "end":
        assert X.iloc[-1]["dates"] not in X_out["dates"]
        assert y.iloc[-1] not in y_out.values

    error_dict = ts_regularizer.error_dict
    assert_features_and_length_equal(X, y, X_out, y_out, error_dict)


def test_ts_regularizer_combination(combination):
    X, y = get_df(combination)

    ww_payload = infer_frequency(X["dates"], debug=True)

    ts_regularizer = TimeSeriesRegularizer(time_index="dates")
    ts_regularizer.fit(X, y)
    X_out, y_out = ts_regularizer.transform(X, y)

    assert pd.infer_freq(X_out["dates"]) == ww_payload[1]["estimated_freq"]

    error_dict = ts_regularizer.error_dict
    assert_features_and_length_equal(X, y, X_out, y_out, error_dict)
