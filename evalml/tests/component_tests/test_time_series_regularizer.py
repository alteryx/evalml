import numpy as np
import pandas as pd
import pytest
from woodwork.statistics_utils import infer_frequency

from evalml.pipelines import TimeSeriesRegularizer


def get_df(dates):
    reg_X = pd.DataFrame()

    reg_X["dates"] = dates
    reg_X["ints"] = [int(i) for i in range(len(dates))]
    reg_X["doubles"] = [i / 0.25 ** (i / 100) for i in range(len(dates))]
    reg_X["bools"] = [bool(min(1, i % 3)) for i in range(len(dates))]
    reg_X["cats"] = np.random.choice(
        ["here", "there", "somewhere", "everywhere"],
        len(dates),
    )

    reg_y = pd.Series([i for i in range(len(dates))])

    return reg_X, reg_y


def assert_features_and_length_equal(
    X,
    y,
    X_output,
    y_output,
    error_dict,
    has_target=True,
):
    ww_payload = infer_frequency(X["dates"], debug=True, window_length=4, threshold=0.4)

    assert isinstance(X_output, pd.DataFrame)
    assert isinstance(y_output, pd.Series) if has_target else True
    assert pd.infer_freq(X_output["dates"]) == ww_payload[1]["estimated_freq"]

    length_mismatch = (
        len(error_dict["duplicate"])
        + len(error_dict["extra"])
        - len(error_dict["missing"])
        + len(error_dict["nan"])
    )
    assert len(X) == len(X_output) + length_mismatch
    assert len(y) == len(y_output) + length_mismatch if has_target else True
    # Randomly test 5 shared dates in the output dataframe and make sure their features match those of the same dates
    # in the input dataframe
    non_nan_X = X_output.dropna()
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
            non_nan_X["dates"] == each_date,
            set(non_nan_X.columns) - {"dates"},
        ].iloc[0]
        pd.testing.assert_series_equal(
            pd.Series(input_feat.values),
            pd.Series(outout_feat.values),
        )


def test_ts_regularizer_init():

    ts_regularizer = TimeSeriesRegularizer(time_index="dates")

    assert ts_regularizer.name == "Time Series Regularizer"
    assert ts_regularizer.parameters == {
        "time_index": "dates",
        "window_length": 4,
        "threshold": 0.4,
    }
    assert ts_regularizer.hyperparameter_ranges == {}
    assert ts_regularizer.modifies_target is True
    assert ts_regularizer.modifies_features is True
    assert ts_regularizer.training_only is True


def test_ts_regularizer_invalid_frequency_payload():
    with pytest.raises(
        ValueError,
        match="The frequency_payload parameter must be a tuple",
    ):
        _ = TimeSeriesRegularizer(time_index="ints", frequency_payload="This is wrong")


def test_ts_regularizer_time_index_not_datetime():
    dates_1 = pd.date_range("1/1/21", periods=10)
    dates_2 = pd.date_range("1/13/21", periods=10, freq="2D")
    dates = dates_1.append(dates_2)

    X, y = get_df(dates)

    ts_regularizer = TimeSeriesRegularizer(time_index="ints")
    with pytest.raises(
        TypeError,
        match="The time_index column `ints` must be of type Datetime.",
    ):
        ts_regularizer.fit(X, y)


def test_ts_regularizer_time_index_doesnt_exist(duplicate_beginning):
    X, y = get_df(duplicate_beginning)

    ts_regularizer = TimeSeriesRegularizer(time_index="blah")
    with pytest.raises(
        KeyError,
        match="The time_index column `blah` does not exist in X!",
    ):
        ts_regularizer.fit(X, y)


def test_ts_regularizer_time_index_is_None(duplicate_beginning):
    X, y = get_df(duplicate_beginning)

    ts_regularizer = TimeSeriesRegularizer(time_index=None)
    with pytest.raises(
        ValueError,
        match="The argument time_index cannot be None!",
    ):
        ts_regularizer.fit(X, y)


def test_ts_regularizer_mismatch_target_length(duplicate_beginning):
    X, _ = get_df(duplicate_beginning)
    y = pd.Series([i for i in range(25)])

    ts_regularizer = TimeSeriesRegularizer(time_index="dates")
    with pytest.raises(
        ValueError,
        match="If y has been passed, then it must be the same length as X.",
    ):
        ts_regularizer.fit(X, y)


def test_ts_regularizer_no_freq():
    dates_1 = pd.date_range("2015-01-01", periods=5, freq="D")
    dates_2 = pd.date_range("2015-01-08", periods=3, freq="D")
    dates_3 = pd.DatetimeIndex(["2015-01-12"])
    dates_4 = pd.date_range("2015-01-15", periods=5, freq="D")
    dates_5 = pd.date_range("2015-01-22", periods=5, freq="D")
    dates_6 = pd.date_range("2015-01-29", periods=11, freq="M")

    dates = (
        dates_1.append(dates_2)
        .append(dates_3)
        .append(dates_4)
        .append(dates_5)
        .append(dates_6)
    )

    X, y = get_df(dates)

    ts_regularizer = TimeSeriesRegularizer(time_index="dates")
    with pytest.raises(
        ValueError,
        match="The column dates does not have a frequency that can be inferred.",
    ):
        ts_regularizer.fit(X, y)


def test_ts_regularizer_no_issues(ts_data):
    X, _, y = ts_data()

    ts_regularizer = TimeSeriesRegularizer(time_index="date")
    X_output, y_output = ts_regularizer.fit_transform(X, y)

    assert ts_regularizer.inferred_freq is not None
    assert len(ts_regularizer.error_dict) == 0
    pd.testing.assert_frame_equal(X, X_output)
    pd.testing.assert_series_equal(y, y_output)


@pytest.mark.parametrize("y_passed", [True, False])
def test_ts_regularizer_X_only_equal_payload(y_passed, combination_of_faulty_datetime):
    X, y = get_df(combination_of_faulty_datetime)

    ww_payload = infer_frequency(
        X["dates"],
        debug=True,
        window_length=5,
        threshold=0.8,
    )

    ts_regularizer_with_payload = TimeSeriesRegularizer(
        time_index="dates",
        frequency_payload=ww_payload,
    )
    ts_regularizer = TimeSeriesRegularizer(time_index="dates")

    X_output_payload, y_output_payload = ts_regularizer_with_payload.fit_transform(
        X,
        y=y if y_passed else None,
    )
    X_output, y_output = ts_regularizer.fit_transform(X, y=y if y_passed else None)

    if not y_passed:
        assert y_output is None

    error_dict = ts_regularizer.error_dict
    assert_features_and_length_equal(
        X,
        y,
        X_output,
        y_output,
        error_dict,
        has_target=True if y_passed else False,
    )
    pd.testing.assert_frame_equal(X_output, X_output_payload)
    if y_passed:
        pd.testing.assert_series_equal(y_output, y_output_payload)


@pytest.mark.parametrize(
    "duplicate_location",
    ["beginning", "middle", "end", "scattered", "continuous"],
)
def test_ts_regularizer_duplicate(
    duplicate_location,
    duplicate_beginning,
    duplicate_middle,
    duplicate_end,
    duplicate_scattered,
    duplicate_continuous,
):

    if duplicate_location == "beginning":
        dates = duplicate_beginning
    elif duplicate_location == "middle":
        dates = duplicate_middle
    elif duplicate_location == "end":
        dates = duplicate_end
    elif duplicate_location == "scattered":
        dates = duplicate_scattered
    else:
        dates = duplicate_continuous

    X, y = get_df(dates)

    ts_regularizer = TimeSeriesRegularizer(time_index="dates")
    X_output, y_output = ts_regularizer.fit_transform(X, y)

    error_dict = ts_regularizer.error_dict
    assert_features_and_length_equal(X, y, X_output, y_output, error_dict)


@pytest.mark.parametrize(
    "missing_location",
    ["beginning", "middle", "end", "scattered", "continuous"],
)
def test_ts_regularizer_missing(
    missing_location,
    missing_beginning,
    missing_middle,
    missing_end,
    missing_scattered,
    missing_continuous,
):

    if missing_location == "beginning":
        dates = missing_beginning
    elif missing_location == "middle":
        dates = missing_middle
    elif missing_location == "end":
        dates = missing_end
    elif missing_location == "scattered":
        dates = missing_scattered
    else:
        dates = missing_continuous

    X, y = get_df(dates)

    ts_regularizer = TimeSeriesRegularizer(time_index="dates")
    X_output, y_output = ts_regularizer.fit_transform(X, y)

    error_dict = ts_regularizer.error_dict
    assert_features_and_length_equal(X, y, X_output, y_output, error_dict)


@pytest.mark.parametrize(
    "uneven_type",
    ["beginning", "middle", "end", "scattered", "continuous", "work week"],
)
def test_ts_regularizer_uneven(
    uneven_type,
    uneven_beginning,
    uneven_middle,
    uneven_end,
    uneven_scattered,
    uneven_continuous,
    uneven_work_week,
):

    if uneven_type == "beginning":
        dates = uneven_beginning
    elif uneven_type == "middle":
        dates = uneven_middle
    elif uneven_type == "end":
        dates = uneven_end
    elif uneven_type == "scattered":
        dates = uneven_scattered
    elif uneven_type == "continuous":
        dates = uneven_continuous
    else:
        dates = uneven_work_week

    X, y = get_df(dates)
    ts_regularizer = TimeSeriesRegularizer(time_index="dates")
    X_output, y_output = ts_regularizer.fit_transform(X, y)

    if uneven_type == "beginning":
        assert X.iloc[0]["dates"] not in X_output["dates"]
        assert X.iloc[1]["dates"] not in X_output["dates"]
        assert y.iloc[0] not in y_output.values
        assert y.iloc[1] not in y_output.values
    elif uneven_type == "end":
        assert X.iloc[-1]["dates"] not in X_output["dates"]
        assert y.iloc[-1] not in y_output.values

    error_dict = ts_regularizer.error_dict
    assert_features_and_length_equal(X, y, X_output, y_output, error_dict)
