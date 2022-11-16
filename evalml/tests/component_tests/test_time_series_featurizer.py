from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import woodwork as ww
from pandas.testing import assert_frame_equal
from woodwork.logical_types import (
    Boolean,
    Categorical,
    Datetime,
    Double,
    Integer,
    Ordinal,
)

from evalml.pipelines import TimeSeriesFeaturizer

ROLLING_TRANSFORM_METHOD_NAME = "_compute_rolling_transforms"
DELAYED_FEATURES_METHOD_NAME = "_compute_delays"


def _all_featurized_cols_double(df):
    ltypes = {}
    for col in df.columns:
        if not isinstance(col, str):
            continue
        if "delay" in col or "rolling_mean" in col:
            ltypes[col] = Double

    return ltypes


def test_delayed_features_transformer_init():
    delayed_features = TimeSeriesFeaturizer(
        max_delay=4,
        delay_features=True,
        delay_target=False,
        time_index="date",
        random_seed=1,
    )
    assert delayed_features.parameters == {
        "max_delay": 4,
        "delay_features": True,
        "delay_target": False,
        "gap": 0,
        "forecast_horizon": 1,
        "time_index": "date",
        "conf_level": 0.05,
        "rolling_window_size": 0.25,
    }


@pytest.mark.parametrize("conf_level", [-0.05, 0, 1.2])
def test_delayed_features_init_raises_if_conf_level_not_in_range(conf_level):
    with pytest.raises(
        ValueError,
        match="Parameter conf_level must be in range \\(0, 1\\]",
    ):
        TimeSeriesFeaturizer(conf_level=conf_level)


def test_delayed_features_init_raises_if_conf_level_None():
    with pytest.raises(ValueError, match="Parameter conf_level cannot be None"):
        TimeSeriesFeaturizer(conf_level=None)


def test_delayed_features_raises_if_time_index_None(ts_data):
    X, _, y = ts_data()
    with pytest.raises(ValueError, match=" cannot be None"):
        dft = TimeSeriesFeaturizer(time_index=None)
        dft.fit_transform(X, y)


def encode_y_as_string(y):
    y = y.astype("category")
    y_answer = y.astype(int) - 1
    y = y.map(lambda val: str(val).zfill(2))
    y.ww.init(logical_type="categorical")
    return y, y_answer


def encode_X_as_string(X):
    X_answer = X.copy()
    X_answer.feature = X.feature.astype(int) - 1
    # So that the encoder encodes the values in ascending order. This makes it easier to
    # specify the answer for each unit test
    X.feature = pd.Categorical(X.feature.map(lambda val: str(val).zfill(2)))
    X.ww.init(logical_types={"feature": "categorical"})
    return X, X_answer


def encode_X_y_as_strings(X, y, encode_X_as_str, encode_y_as_str):
    y_answer = y
    if encode_y_as_str:
        y, y_answer = encode_y_as_string(y)
    X_answer = X
    if encode_X_as_str:
        X, X_answer = encode_X_as_string(X)
    return X, X_answer, y, y_answer


@pytest.mark.parametrize("encode_X_as_str", [True, False])
@pytest.mark.parametrize("encode_y_as_str", [True, False])
@patch(
    f"evalml.pipelines.components.transformers.TimeSeriesFeaturizer.{ROLLING_TRANSFORM_METHOD_NAME}",
    return_value=pd.DataFrame(),
)
def test_delayed_feature_extractor_maxdelay3_forecasthorizon1_gap0(
    mock_roll,
    encode_X_as_str,
    encode_y_as_str,
    ts_data,
):
    X, _, y = ts_data()
    X, X_answer, y, y_answer = encode_X_y_as_strings(
        X,
        y,
        encode_X_as_str,
        encode_y_as_str,
    )
    answer = pd.DataFrame(
        {
            "date": X["date"],
            "feature_delay_1": X_answer.feature.shift(1),
            "feature_delay_2": X_answer.feature.shift(2),
            "feature_delay_3": X_answer.feature.shift(3),
            "feature_delay_4": X_answer.feature.shift(4),
            "target_delay_1": y_answer.shift(1),
            "target_delay_2": y_answer.shift(2),
            "target_delay_3": y_answer.shift(3),
            "target_delay_4": y_answer.shift(4),
        },
    )
    logical_types = _all_featurized_cols_double(answer)
    answer.ww.init(logical_types=logical_types)
    assert_frame_equal(
        answer,
        TimeSeriesFeaturizer(
            max_delay=3,
            gap=0,
            forecast_horizon=1,
            conf_level=1.0,
            time_index="date",
        ).fit_transform(X=X, y=y),
    )

    answer_only_y = pd.DataFrame(
        {
            "date": X["date"],
            "target_delay_1": y_answer.shift(1),
            "target_delay_2": y_answer.shift(2),
            "target_delay_3": y_answer.shift(3),
            "target_delay_4": y_answer.shift(4),
        },
    )
    logical_types = _all_featurized_cols_double(answer_only_y)
    answer_only_y.ww.init(logical_types=logical_types)
    assert_frame_equal(
        answer_only_y,
        TimeSeriesFeaturizer(
            max_delay=3,
            gap=0,
            forecast_horizon=1,
            conf_level=1.0,
            time_index="date",
            delay_features=False,
        ).fit_transform(X=X, y=y),
        check_dtype=False,
    )


@pytest.mark.parametrize("encode_X_as_str", [True, False])
@pytest.mark.parametrize("encode_y_as_str", [True, False])
@patch(
    f"evalml.pipelines.TimeSeriesFeaturizer.{ROLLING_TRANSFORM_METHOD_NAME}",
    return_value=pd.DataFrame(),
)
def test_delayed_feature_extractor_maxdelay5_forecasthorizon1_gap0(
    mock_roll,
    encode_X_as_str,
    encode_y_as_str,
    ts_data,
):
    X, _, y = ts_data()
    X, X_answer, y, y_answer = encode_X_y_as_strings(
        X,
        y,
        encode_X_as_str,
        encode_y_as_str,
    )
    answer = pd.DataFrame(
        {
            "date": X["date"],
            "feature_delay_1": X_answer.feature.shift(1),
            "feature_delay_2": X_answer.feature.shift(2),
            "feature_delay_3": X_answer.feature.shift(3),
            "feature_delay_4": X_answer.feature.shift(4),
            "feature_delay_5": X_answer.feature.shift(5),
            "feature_delay_6": X_answer.feature.shift(6),
            "target_delay_1": y_answer.shift(1),
            "target_delay_2": y_answer.shift(2),
            "target_delay_3": y_answer.shift(3),
            "target_delay_4": y_answer.shift(4),
            "target_delay_5": y_answer.shift(5),
            "target_delay_6": y_answer.shift(6),
        },
    )
    logical_types = _all_featurized_cols_double(answer)
    answer.ww.init(logical_types=logical_types)

    assert_frame_equal(
        answer,
        TimeSeriesFeaturizer(
            max_delay=5,
            gap=0,
            forecast_horizon=1,
            conf_level=1.0,
            time_index="date",
        ).fit_transform(X, y),
    )

    answer_only_y = pd.DataFrame(
        {
            "date": X["date"],
            "target_delay_1": y_answer.shift(1),
            "target_delay_2": y_answer.shift(2),
            "target_delay_3": y_answer.shift(3),
            "target_delay_4": y_answer.shift(4),
            "target_delay_5": y_answer.shift(5),
            "target_delay_6": y_answer.shift(6),
        },
    )
    logical_types = _all_featurized_cols_double(answer_only_y)
    answer_only_y.ww.init(logical_types=logical_types)
    assert_frame_equal(
        answer_only_y,
        TimeSeriesFeaturizer(
            max_delay=5,
            gap=0,
            forecast_horizon=1,
            conf_level=1.0,
            time_index="date",
            delay_features=False,
        ).fit_transform(X=X, y=y),
    )


@pytest.mark.parametrize("encode_X_as_str", [True, False])
@pytest.mark.parametrize("encode_y_as_str", [True, False])
@patch(
    f"evalml.pipelines.TimeSeriesFeaturizer.{ROLLING_TRANSFORM_METHOD_NAME}",
    return_value=pd.DataFrame(),
)
def test_delayed_feature_extractor_maxdelay2_forecasthorizon7_gap1(
    mock_roll,
    encode_X_as_str,
    encode_y_as_str,
    ts_data,
):
    X, _, y = ts_data()
    X, X_answer, y, y_answer = encode_X_y_as_strings(
        X,
        y,
        encode_X_as_str,
        encode_y_as_str,
    )
    answer = pd.DataFrame(
        {
            "date": X["date"],
            "feature_delay_8": X_answer.feature.shift(8),
            "feature_delay_9": X_answer.feature.shift(9),
            "feature_delay_10": X_answer.feature.shift(10),
            "target_delay_8": y_answer.shift(8),
            "target_delay_9": y_answer.shift(9),
            "target_delay_10": y_answer.shift(10),
        },
    )
    logical_types = _all_featurized_cols_double(answer)
    answer.ww.init(logical_types=logical_types)
    assert_frame_equal(
        answer,
        TimeSeriesFeaturizer(
            max_delay=2,
            forecast_horizon=7,
            gap=1,
            conf_level=1.0,
            time_index="date",
        ).fit_transform(X, y),
    )

    answer_only_y = pd.DataFrame(
        {
            "date": X["date"],
            "target_delay_8": y_answer.shift(8),
            "target_delay_9": y_answer.shift(9),
            "target_delay_10": y_answer.shift(10),
        },
    )
    logical_types = _all_featurized_cols_double(answer_only_y)
    answer_only_y.ww.init(logical_types=logical_types)
    assert_frame_equal(
        answer_only_y,
        TimeSeriesFeaturizer(
            max_delay=2,
            forecast_horizon=7,
            gap=1,
            conf_level=1.0,
            time_index="date",
            delay_features=False,
        ).fit_transform(X=X, y=y),
    )


@patch(
    f"evalml.pipelines.TimeSeriesFeaturizer.{ROLLING_TRANSFORM_METHOD_NAME}",
    return_value=pd.DataFrame(),
)
def test_delayed_feature_extractor_numpy(mock_roll):
    X = pd.DataFrame(
        {"feature": range(1, 32), "date": pd.date_range("2021-01-01", periods=31)},
    )
    y = pd.Series(range(1, 32))
    X, X_answer, y, y_answer = encode_X_y_as_strings(X, y, False, False)
    X_np = X.values
    y_np = y.values
    answer = pd.DataFrame(
        {
            1: X["date"],
            "target_delay_8": y_answer.shift(8),
            "target_delay_9": y_answer.shift(9),
        },
    )
    logical_types = _all_featurized_cols_double(answer)
    answer.ww.init(logical_types=logical_types)
    res = TimeSeriesFeaturizer(
        max_delay=2,
        forecast_horizon=7,
        gap=1,
        conf_level=1.0,
        time_index=1,
        delay_features=False,
    ).fit_transform(X_np, y_np)
    assert_frame_equal(
        answer,
        res,
    )
    answer_only_y = pd.DataFrame(
        {
            "date": X["date"],
            "target_delay_8": y_answer.shift(8),
            "target_delay_9": y_answer.shift(9),
        },
    )
    logical_types = _all_featurized_cols_double(answer_only_y)
    answer_only_y.ww.init(logical_types=logical_types)
    assert_frame_equal(
        answer_only_y,
        TimeSeriesFeaturizer(
            max_delay=2,
            forecast_horizon=7,
            gap=1,
            conf_level=1.0,
            time_index="date",
            delay_features=False,
        ).fit_transform(X=X, y=y_np),
    )


@pytest.mark.parametrize(
    "delay_features,delay_target",
    [(False, True), (True, False), (False, False)],
)
@pytest.mark.parametrize("encode_X_as_str", [True, False])
@pytest.mark.parametrize("encode_y_as_str", [True, False])
@patch(
    f"evalml.pipelines.TimeSeriesFeaturizer.{ROLLING_TRANSFORM_METHOD_NAME}",
    return_value=pd.DataFrame(),
)
def test_lagged_feature_extractor_delay_features_delay_target(
    mock_roll,
    encode_y_as_str,
    encode_X_as_str,
    delay_features,
    delay_target,
    ts_data,
):
    X, _, y = ts_data()
    X, X_answer, y, y_answer = encode_X_y_as_strings(
        X,
        y,
        encode_X_as_str,
        encode_y_as_str,
    )
    all_delays = pd.DataFrame(
        {
            "date": X["date"],
            "feature_delay_1": X_answer.feature.shift(1),
            "feature_delay_2": X_answer.feature.shift(2),
            "feature_delay_3": X_answer.feature.shift(3),
            "feature_delay_4": X_answer.feature.shift(4),
            "target_delay_1": y_answer.shift(1),
            "target_delay_2": y_answer.shift(2),
            "target_delay_3": y_answer.shift(3),
            "target_delay_4": y_answer.shift(4),
        },
    )

    if not delay_features:
        all_delays = all_delays.drop(
            columns=[c for c in all_delays.columns if "feature_" in c],
        )
    if not delay_target:
        all_delays = all_delays.drop(
            columns=[c for c in all_delays.columns if "target" in c],
        )

    transformer = TimeSeriesFeaturizer(
        max_delay=3,
        forecast_horizon=1,
        delay_features=delay_features,
        delay_target=delay_target,
        conf_level=1.0,
        time_index="date",
    )
    logical_types = _all_featurized_cols_double(all_delays)
    all_delays.ww.init(logical_types=logical_types)
    assert_frame_equal(all_delays, transformer.fit_transform(X, y))


@pytest.mark.parametrize(
    "delay_features,delay_target",
    [(False, True), (True, False), (False, False)],
)
@pytest.mark.parametrize("encode_X_as_str", [True, False])
@pytest.mark.parametrize("encode_y_as_str", [True, False])
@patch(
    f"evalml.pipelines.TimeSeriesFeaturizer.{ROLLING_TRANSFORM_METHOD_NAME}",
    return_value=pd.DataFrame(),
)
def test_lagged_feature_extractor_delay_target(
    mock_roll,
    encode_y_as_str,
    encode_X_as_str,
    delay_features,
    delay_target,
    ts_data,
):
    X, _, y = ts_data()
    X, X_answer, y, y_answer = encode_X_y_as_strings(
        X,
        y,
        encode_X_as_str,
        encode_y_as_str,
    )
    answer = pd.DataFrame({"date": X["date"]})
    if delay_features:
        delayed_features = pd.DataFrame(
            {
                "feature_delay_1": X_answer.feature.shift(1),
                "feature_delay_2": X_answer.feature.shift(2),
                "feature_delay_3": X_answer.feature.shift(3),
                "feature_delay_4": X_answer.feature.shift(4),
            },
        )
        answer = pd.concat([answer, delayed_features], axis=1)
    if delay_target:
        delayed_target = pd.DataFrame(
            {
                "target_delay_1": y_answer.shift(1),
                "target_delay_2": y_answer.shift(2),
                "target_delay_3": y_answer.shift(3),
                "target_delay_4": y_answer.shift(4),
            },
        )
        answer = pd.concat([answer, delayed_target], axis=1)

    transformer = TimeSeriesFeaturizer(
        max_delay=3,
        forecast_horizon=1,
        delay_features=delay_features,
        delay_target=delay_target,
        conf_level=1.0,
        time_index="date",
    )
    logical_types = _all_featurized_cols_double(answer)
    answer.ww.init(logical_types=logical_types)
    assert_frame_equal(answer, transformer.fit_transform(X, y))


@pytest.mark.parametrize("encode_X_as_str", [True, False])
@pytest.mark.parametrize("encode_y_as_str", [True, False])
@pytest.mark.parametrize("data_type", ["ww", "pd"])
def test_delay_feature_transformer_supports_custom_index(
    encode_X_as_str,
    encode_y_as_str,
    data_type,
    make_data_type,
    ts_data,
):
    X, _, y = ts_data()
    X, X_answer, y, y_answer = encode_X_y_as_strings(
        X,
        y,
        encode_X_as_str,
        encode_y_as_str,
    )
    X.index = pd.RangeIndex(50, 90)
    X_answer.index = pd.RangeIndex(50, 90)
    y.index = pd.RangeIndex(50, 90)
    y_answer.index = pd.RangeIndex(50, 90)
    answer = pd.DataFrame(
        {
            "date": X["date"],
            "feature_delay_7": X_answer.feature.shift(7),
            "feature_delay_8": X_answer.feature.shift(8),
            "feature_delay_9": X_answer.feature.shift(9),
            "feature_delay_10": X_answer.feature.shift(10),
            "target_delay_7": y_answer.shift(7),
            "target_delay_8": y_answer.shift(8),
            "target_delay_9": y_answer.shift(9),
            "target_delay_10": y_answer.shift(10),
        },
        index=pd.RangeIndex(50, 90),
    )
    answer.ww.init(
        logical_types={
            col: "Double" if col != "date" else "datetime" for col in answer.columns
        },
    )
    logical_types = _all_featurized_cols_double(answer)
    answer.ww.init(logical_types=logical_types)
    if not encode_y_as_str and not encode_X_as_str:
        rolling_features = pd.DataFrame(
            {
                "feature_rolling_mean": X_answer.feature.shift(7).rolling(4, 4).mean(),
                "target_rolling_mean": y_answer.shift(7).rolling(4, 4).mean(),
            },
            index=pd.RangeIndex(50, 90),
        )
        rolling_features_target_only = rolling_features
    elif encode_y_as_str and not encode_X_as_str:
        rolling_features = pd.DataFrame(
            {
                "feature_rolling_mean": X_answer.feature.shift(7).rolling(4, 4).mean(),
            },
            index=pd.RangeIndex(50, 90),
        )
        rolling_features_target_only = rolling_features
    elif not encode_y_as_str and encode_X_as_str:
        rolling_features = pd.DataFrame(
            {
                "target_rolling_mean": y_answer.shift(7).rolling(4, 4).mean(),
            },
            index=pd.RangeIndex(50, 90),
        )
        rolling_features_target_only = pd.DataFrame(
            {
                "target_rolling_mean": y_answer.shift(7).rolling(4, 4).mean(),
            },
            index=pd.RangeIndex(50, 90),
        )
    else:
        rolling_features = pd.DataFrame()
        rolling_features_target_only = pd.DataFrame()
    rolling_features_target_only.ww.init(
        logical_types={col: "Double" for col in rolling_features_target_only.columns},
    )

    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)

    concatted = ww.concat_columns([answer, rolling_features])
    res = TimeSeriesFeaturizer(
        max_delay=3,
        forecast_horizon=7,
        conf_level=1.0,
        rolling_window_size=1.0,
        time_index="date",
    ).fit_transform(X, y)
    assert_frame_equal(
        concatted,
        res,
    )

    answer_only_y = pd.DataFrame(
        {
            "date": X["date"],
            "target_delay_7": y_answer.shift(7),
            "target_delay_8": y_answer.shift(8),
            "target_delay_9": y_answer.shift(9),
            "target_delay_10": y_answer.shift(10),
        },
        index=pd.RangeIndex(50, 90),
    )
    answer_only_y.ww.init(
        logical_types={
            col: "Double" if col != "date" else "datetime"
            for col in answer_only_y.columns
        },
    )
    answer_only_y = ww.concat_columns([answer_only_y, rolling_features_target_only])
    assert_frame_equal(
        answer_only_y,
        TimeSeriesFeaturizer(
            max_delay=3,
            forecast_horizon=7,
            conf_level=1.0,
            time_index="date",
            rolling_window_size=1.0,
            delay_features=False,
        ).fit_transform(X=X, y=y),
    )


def test_delay_feature_transformer_multiple_categorical_columns(ts_data):
    X, _, y = ts_data()
    X, X_answer, y, y_answer = encode_X_y_as_strings(X, y, True, True)
    logical_types = X.ww.logical_types

    X["feature_2"] = pd.Categorical(
        ["a"] * 10 + ["aa"] * 10 + ["aaa"] * 10 + ["aaaa"] * 10,
    )
    logical_types["feature_2"] = "categorical"
    X.ww.init(logical_types=logical_types)

    X_answer["feature_2"] = pd.Series(
        [0] * 10 + [1] * 10 + [2] * 10 + [3] * 10,
        index=X.index,
    )
    answer = pd.DataFrame(
        {
            "date": X["date"],
            "feature_delay_11": X_answer.feature.shift(11),
            "feature_2_delay_11": X_answer.feature_2.shift(11),
            "target_delay_11": y_answer.shift(11),
        },
    )
    logical_types = _all_featurized_cols_double(answer)
    answer.ww.init(logical_types=logical_types)
    expected = TimeSeriesFeaturizer(
        max_delay=1,
        forecast_horizon=9,
        gap=2,
        conf_level=1.0,
        time_index="date",
    ).fit_transform(X, y)
    assert_frame_equal(answer, expected)


def test_delay_feature_transformer_y_is_none(ts_data):
    X, _, _ = ts_data()
    answer = pd.DataFrame(
        {
            "date": X["date"],
            "feature_delay_11": X.feature.shift(11),
            "feature_delay_12": X.feature.shift(12),
            "feature_rolling_mean": X.feature.shift(11).rolling(2, 2).mean(),
        },
    )
    logical_types = _all_featurized_cols_double(answer)
    answer.ww.init(logical_types=logical_types)
    assert_frame_equal(
        answer,
        TimeSeriesFeaturizer(
            max_delay=1,
            forecast_horizon=11,
            conf_level=1.0,
            rolling_window_size=1.0,
            time_index="date",
        ).fit_transform(X, y=None),
    )


def test_delayed_feature_transformer_does_not_modify_input_data(ts_data):
    X, _, _ = ts_data()
    expected = X.copy()
    _ = TimeSeriesFeaturizer(
        max_delay=1,
        forecast_horizon=11,
        conf_level=1.0,
        time_index="date",
    ).fit_transform(X, y=None)

    assert_frame_equal(X, expected)


@pytest.mark.parametrize("rolling_window_size", [0.1, 0.2, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("gap", [0, 1, 2])
@pytest.mark.parametrize("forecast_horizon", [1, 5, 10])
@pytest.mark.parametrize("max_delay", [1, 3, 9])
@patch(
    f"evalml.pipelines.TimeSeriesFeaturizer.{DELAYED_FEATURES_METHOD_NAME}",
)
def test_time_series_featurizer_rolling_mean(
    mock_delay,
    max_delay,
    forecast_horizon,
    rolling_window_size,
    gap,
    ts_data,
):
    X, _, y = ts_data()
    mock_delay.return_value = X
    output = TimeSeriesFeaturizer(
        max_delay=max_delay,
        forecast_horizon=forecast_horizon,
        gap=gap,
        rolling_window_size=rolling_window_size,
        time_index="date",
    ).fit_transform(X, y)

    size = int(rolling_window_size * max_delay)
    rolling_means = (
        X.feature.shift(forecast_horizon + gap).rolling(size + 1, size + 1).mean()
    )
    rolling_means_target = (
        y.shift(forecast_horizon + gap).rolling(size + 1, size + 1).mean()
    )
    expected = pd.DataFrame(
        {
            "date": X["date"],
            "feature_rolling_mean": rolling_means,
            "target_rolling_mean": rolling_means_target,
        },
    )
    logical_types = _all_featurized_cols_double(expected)
    expected.ww.init(logical_types=logical_types)
    assert_frame_equal(output, expected)


@pytest.mark.parametrize("rolling_window_size", [0.1, 0.2, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("gap", [0, 1, 2])
@pytest.mark.parametrize("forecast_horizon", [1, 5, 10])
@pytest.mark.parametrize("max_delay", [1, 3, 9])
def test_time_series_featurizer_does_not_need_to_delay_to_compute_means(
    max_delay,
    forecast_horizon,
    gap,
    rolling_window_size,
    ts_data,
):
    X, _, y = ts_data()
    output = TimeSeriesFeaturizer(
        max_delay=max_delay,
        forecast_horizon=forecast_horizon,
        gap=gap,
        delay_features=False,
        delay_target=False,
        time_index="date",
        rolling_window_size=rolling_window_size,
    ).fit_transform(X, y)

    size = int(rolling_window_size * max_delay)
    rolling_means = (
        X.feature.shift(forecast_horizon + gap).rolling(size + 1, size + 1).mean()
    )
    rolling_means_target = (
        y.shift(forecast_horizon + gap).rolling(size + 1, size + 1).mean()
    )
    expected = pd.DataFrame(
        {
            "date": X["date"],
            "feature_rolling_mean": rolling_means,
            "target_rolling_mean": rolling_means_target,
        },
    )
    logical_types = _all_featurized_cols_double(expected)
    expected.ww.init(logical_types=logical_types)
    assert_frame_equal(output, expected)


@pytest.mark.parametrize("peaks", [[14, 21, 28], [32, 45], [18, 29, 56], [5, 8, 12]])
@pytest.mark.parametrize(
    "significant_lags",
    [[1, 2, 8, 14, 21], [1, 19, 20, 32], [14, 21, 28, 56, 18], [13, 25, 8]],
)
@patch(
    "evalml.pipelines.components.transformers.preprocessing.time_series_featurizer.find_peaks",
)
@patch(
    "evalml.pipelines.components.transformers.preprocessing.time_series_featurizer.acf",
)
@patch(
    f"evalml.pipelines.TimeSeriesFeaturizer.{ROLLING_TRANSFORM_METHOD_NAME}",
    return_value=pd.DataFrame(),
)
def test_delayed_feature_transformer_conf_level(
    mock_roll,
    mock_acf,
    mock_peaks,
    peaks,
    significant_lags,
):
    X = pd.DataFrame(
        {
            "feature": np.arange(10000),
            "date": pd.date_range("2021-01-01", periods=10000),
        },
    )
    y = pd.Series(np.arange(10000))

    def create_acf_return_value(y, significant_lags, peaks):
        """Create ci intervals such that significant_lags and peaks are significant."""
        acf_series = np.arange(len(y))
        ci = np.ones((len(y), 2))
        ci[:, 0] = -1
        ci[significant_lags + peaks, 0] = 0.1
        ci[significant_lags + peaks, 1] = 0.3
        return acf_series, ci

    mock_acf.return_value = create_acf_return_value(y, significant_lags, peaks)
    mock_peaks.return_value = peaks, None

    MAX_DELAY = 50
    FORECAST_HORIZON = 10

    # Although the conf_level is hard-coded we mock the return values of
    # find_peaks and acf so that we simulate different significant lags.
    dft = TimeSeriesFeaturizer(
        max_delay=MAX_DELAY,
        forecast_horizon=FORECAST_HORIZON,
        conf_level=0.05,
        gap=0,
        time_index="date",
    )
    new_X = dft.fit_transform(X, y)

    first_significant_10 = [l for l in significant_lags if l < 10]
    expected_lags = set(peaks).union(first_significant_10)
    expected_lags = sorted(
        expected_lags.intersection(
            np.arange(FORECAST_HORIZON, FORECAST_HORIZON + MAX_DELAY + 1),
        ),
    )
    answer = pd.DataFrame({"date": X["date"]})
    answer = answer.assign(
        **{
            f"feature_delay_{t}": X["feature"].shift(
                t,
            )
            for t in expected_lags
        }
    )
    answer = answer.assign(**{f"target_delay_{t}": y.shift(t) for t in expected_lags})
    # Sort columns in alphabetical order
    answer = answer.sort_index(axis=1)
    logical_types = _all_featurized_cols_double(answer)
    answer.ww.init(logical_types=logical_types)
    assert_frame_equal(new_X, answer)


@patch(
    "evalml.pipelines.components.transformers.preprocessing.time_series_featurizer.find_peaks",
)
@patch(
    "evalml.pipelines.components.transformers.preprocessing.time_series_featurizer.acf",
)
@patch(
    f"evalml.pipelines.TimeSeriesFeaturizer.{ROLLING_TRANSFORM_METHOD_NAME}",
    return_value=pd.DataFrame(),
)
def test_delayed_feature_transformer_selects_first_lag_if_none_significant(
    mock_roll,
    mock_acf,
    mock_peaks,
):
    X = pd.DataFrame(
        {
            "feature": np.arange(10000),
            "date": pd.date_range("2021-01-01", periods=10000),
        },
    )
    y = pd.Series(np.arange(10000))
    X.ww.init(logical_types={"feature": "IntegerNullable", "date": "datetime"})

    acf_series = np.arange(len(y))
    ci = np.ones((len(y), 2))
    ci[:, 0] = -1

    mock_acf.return_value = acf_series, ci
    mock_peaks.return_value = [], None

    MAX_DELAY = 50
    FORECAST_HORIZON = 10

    dft = TimeSeriesFeaturizer(
        max_delay=MAX_DELAY,
        forecast_horizon=FORECAST_HORIZON,
        conf_level=0.1,
        gap=0,
        time_index="date",
    )
    new_X = dft.fit_transform(X, y)
    answer = pd.DataFrame(
        {
            "date": X["date"],
            f"feature_delay_{FORECAST_HORIZON}": X["feature"].shift(
                FORECAST_HORIZON,
            ),
            f"target_delay_{FORECAST_HORIZON}": y.shift(FORECAST_HORIZON),
        },
    )
    logical_types = _all_featurized_cols_double(answer)
    answer.ww.init(logical_types=logical_types)
    assert_frame_equal(new_X, answer)


@pytest.mark.parametrize(
    "X_df",
    [
        pd.DataFrame(pd.Series([0, 0, 3, 1] * 5, dtype="int64")),
        pd.DataFrame(pd.Series([0, 0, 3.0, 2] * 5, dtype="float")),
        pd.DataFrame(pd.Series(["a", "b", "a"] * 5, dtype="category")),
    ],
)
@pytest.mark.parametrize("fit_transform", [True, False])
@patch(
    f"evalml.pipelines.TimeSeriesFeaturizer.{ROLLING_TRANSFORM_METHOD_NAME}",
    return_value=pd.DataFrame(),
)
def test_delay_feature_transformer_woodwork_custom_overrides_returned_by_components(
    mock_roll,
    X_df,
    fit_transform,
):
    y = pd.Series(range(X_df.shape[0]))
    override_types = [Integer, Double, Categorical, Boolean]

    for logical_type in override_types:
        try:
            X = X_df.copy()
            X["date"] = pd.date_range("2021-01-01", periods=X.shape[0])
            X.ww.init(logical_types={0: logical_type})
        except (ww.exceptions.TypeConversionError, ValueError):
            continue
        if X.loc[:, 0].isna().all():
            # Casting the fourth and fifth dataframes to datetime will produce all NaNs
            continue
        dft = TimeSeriesFeaturizer(
            max_delay=1,
            forecast_horizon=1,
            conf_level=1.0,
            time_index="date",
        )
        if fit_transform:
            transformed = dft.fit_transform(X, y)
        else:
            dft.fit(X, y)
            transformed = dft.transform(X, y)
        assert isinstance(transformed, pd.DataFrame)
        transformed_logical_types = {
            k: type(v) for k, v in transformed.ww.logical_types.items()
        }
        if logical_type in [Integer, Double, Categorical, Boolean]:
            assert transformed_logical_types == {
                "date": Datetime,
                "0_delay_1": Double,
                "0_delay_2": Double,
                "target_delay_1": Double,
                "target_delay_2": Double,
            }


def test_delay_feature_transformer_works_for_non_numeric_ordinal_categories(ts_data):
    X, _, y = ts_data()
    X.ww.set_time_index("date")
    X.ww["cats"] = pd.Series(
        np.random.choice(["a", "b", "c"], size=len(X)),
        index=X.index,
    )
    X.ww.set_types(logical_types={"cats": Ordinal(order=["a", "b", "c"])})

    output = TimeSeriesFeaturizer(
        max_delay=3,
        forecast_horizon=1,
        gap=0,
        conf_level=1,
        time_index="date",
    )
    output.fit(X, y)
    X_t = output.transform(X, y)
    assert set(X_t["cats_delay_1"].value_counts().to_dict().keys()) == {2.0, 0.0, 1.0}
